# Hebbian Learning Checkpoint Manager with Delta Tracking
# Copyright 2026 - Kharagpur Data Science Hackathon

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import copy
import zlib
import pickle

from config import HebbianConfig


@dataclass
class Checkpoint:
    """Single checkpoint storing layer states"""
    step: int
    timestamp: float
    layer_states: Dict[str, torch.Tensor]
    metadata: Dict = field(default_factory=dict)


@dataclass
class DeltaCheckpoint:
    """Compressed delta between two checkpoints"""
    from_step: int
    to_step: int
    deltas: Dict[str, bytes]  # Compressed delta tensors
    metadata: Dict = field(default_factory=dict)


class HebbianCheckpointManager:
    """
    Manages Hebbian learning checkpoints with delta compression.
    
    Key Features:
    - Stores layer-wise changes between checkpoints
    - Compressed delta storage for memory efficiency
    - Fast state retrieval via accumulated deltas
    - Tracks how layers change over time for analysis
    """
    
    def __init__(self, config: HebbianConfig):
        self.config = config
        self.checkpoints: List[Checkpoint] = []
        self.delta_checkpoints: List[DeltaCheckpoint] = []
        self.current_step = 0
        
    def save_checkpoint(
        self, 
        layer_states: Dict[str, Dict[str, torch.Tensor]], 
        step: int,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save a new checkpoint with layer states.
        
        Args:
            layer_states: Dict mapping layer names to their parameter dicts
            step: Current training step
            metadata: Optional metadata (loss, accuracy, etc.)
        """
        import time
        
        # Flatten layer states for storage
        flat_states = {}
        for layer_name, params in layer_states.items():
            for param_name, tensor in params.items():
                key = f"{layer_name}/{param_name}"
                flat_states[key] = tensor.detach().cpu().clone()
        
        checkpoint = Checkpoint(
            step=step,
            timestamp=time.time(),
            layer_states=flat_states,
            metadata=metadata or {}
        )
        
        # Compute delta from previous checkpoint if exists
        if len(self.checkpoints) > 0:
            prev_checkpoint = self.checkpoints[-1]
            delta = self._compute_delta(prev_checkpoint, checkpoint)
            self.delta_checkpoints.append(delta)
        
        self.checkpoints.append(checkpoint)
        self.current_step = step
        
        # Prune old checkpoints if exceeding max
        self._prune_checkpoints()
    
    def _compute_delta(
        self, 
        prev: Checkpoint, 
        curr: Checkpoint
    ) -> DeltaCheckpoint:
        """Compute compressed delta between two checkpoints"""
        deltas = {}
        
        for key in curr.layer_states:
            if key in prev.layer_states:
                delta_tensor = curr.layer_states[key] - prev.layer_states[key]
                # Compress the delta
                if self.config.delta_compression:
                    delta_bytes = pickle.dumps(delta_tensor)
                    compressed = zlib.compress(delta_bytes, level=6)
                    deltas[key] = compressed
                else:
                    deltas[key] = pickle.dumps(delta_tensor)
            else:
                # New parameter, store full state
                if self.config.delta_compression:
                    deltas[key] = zlib.compress(pickle.dumps(curr.layer_states[key]))
                else:
                    deltas[key] = pickle.dumps(curr.layer_states[key])
        
        return DeltaCheckpoint(
            from_step=prev.step,
            to_step=curr.step,
            deltas=deltas,
            metadata={
                'delta_norm': self._compute_delta_norms(prev, curr)
            }
        )
    
    def _compute_delta_norms(
        self, 
        prev: Checkpoint, 
        curr: Checkpoint
    ) -> Dict[str, float]:
        """Compute L2 norms of deltas for analysis"""
        norms = {}
        for key in curr.layer_states:
            if key in prev.layer_states:
                delta = curr.layer_states[key] - prev.layer_states[key]
                norms[key] = delta.norm().item()
        return norms
    
    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints to stay within memory budget"""
        while len(self.checkpoints) > self.config.max_checkpoints:
            self.checkpoints.pop(0)
            if len(self.delta_checkpoints) > 0:
                self.delta_checkpoints.pop(0)
    
    def retrieve_state(self, step: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve layer states at a specific step.
        Reconstructs from base checkpoint + accumulated deltas.
        """
        # Find the checkpoint at or before the requested step
        target_checkpoint = None
        for cp in reversed(self.checkpoints):
            if cp.step <= step:
                target_checkpoint = cp
                break
        
        if target_checkpoint is None:
            return None
        
        if target_checkpoint.step == step:
            return copy.deepcopy(target_checkpoint.layer_states)
        
        # Reconstruct by applying deltas
        state = copy.deepcopy(target_checkpoint.layer_states)
        
        for delta in self.delta_checkpoints:
            if delta.from_step >= target_checkpoint.step and delta.to_step <= step:
                for key, compressed_delta in delta.deltas.items():
                    if self.config.delta_compression:
                        delta_tensor = pickle.loads(zlib.decompress(compressed_delta))
                    else:
                        delta_tensor = pickle.loads(compressed_delta)
                    
                    if key in state:
                        state[key] = state[key] + delta_tensor
                    else:
                        state[key] = delta_tensor
        
        return state
    
    def get_layer_trajectory(
        self, 
        layer_name: str
    ) -> List[Tuple[int, Dict[str, float]]]:
        """
        Get the trajectory of changes for a specific layer.
        
        Returns:
            List of (step, delta_norms) tuples showing how layer changed
        """
        trajectory = []
        for delta in self.delta_checkpoints:
            layer_norms = {}
            for key, norm in delta.metadata.get('delta_norm', {}).items():
                if key.startswith(layer_name):
                    param_name = key.split('/')[-1]
                    layer_norms[param_name] = norm
            if layer_norms:
                trajectory.append((delta.to_step, layer_norms))
        return trajectory
    
    def get_checkpoint_summary(self) -> Dict:
        """Get summary statistics about stored checkpoints"""
        if not self.checkpoints:
            return {'count': 0}
        
        return {
            'count': len(self.checkpoints),
            'first_step': self.checkpoints[0].step,
            'last_step': self.checkpoints[-1].step,
            'delta_count': len(self.delta_checkpoints),
            'layers_tracked': list(self.checkpoints[-1].layer_states.keys())
        }


class HebbianLearningRule(nn.Module):
    """
    Implements Hebbian learning with stabilization for BDH layers.
    
    Update rule: Δw = η * (x * y^T) - λ * w
    Where:
        η: learning rate
        λ: decay factor for stability
        x: pre-synaptic activation
        y: post-synaptic activation
    """
    
    def __init__(self, config: HebbianConfig):
        super().__init__()
        self.config = config
        self.learning_rate = config.learning_rate
        self.decay = config.decay_factor
        
    def compute_update(
        self, 
        pre_activation: torch.Tensor, 
        post_activation: torch.Tensor,
        weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hebbian weight update.
        
        Args:
            pre_activation: [B, N_pre] pre-synaptic activations
            post_activation: [B, N_post] post-synaptic activations
            weight: [N_pre, N_post] current weight matrix
            
        Returns:
            delta_weight: [N_pre, N_post] weight update
        """
        # Hebbian term: correlation between pre and post
        # Average over batch
        B = pre_activation.size(0)
        hebbian_term = torch.einsum('bi,bj->ij', pre_activation, post_activation) / B
        
        # Decay term for stability
        decay_term = self.decay * weight
        
        # Combined update
        delta_weight = self.learning_rate * hebbian_term - decay_term
        
        return delta_weight
    
    def apply_update(
        self, 
        weight: nn.Parameter, 
        delta: torch.Tensor,
        soft_clamp: float = 1.0
    ) -> None:
        """Apply update with optional soft clamping for stability"""
        with torch.no_grad():
            weight.add_(delta)
            if soft_clamp > 0:
                # Soft clamping to prevent weight explosion
                weight.data = torch.tanh(weight.data / soft_clamp) * soft_clamp


# ============================================================================
# TOON Checkpoint Storage for Persistent Memory
# ============================================================================

import os
import json
import base64
from datetime import datetime


class CheckpointTOONStorage:
    """
    Stores checkpoint metadata and deltas in TOON format for:
    - Persistent storage across sessions
    - Easy change tracking and diffing
    - Human-readable checkpoint history
    - Reduced memory loss in long conversations
    
    Binary tensor data is stored separately, with TOON containing references.
    """
    
    def __init__(self, storage_dir: str = "checkpoints"):
        self.storage_dir = storage_dir
        self.toon_file = os.path.join(storage_dir, "checkpoint_history.toon")
        self.tensors_dir = os.path.join(storage_dir, "tensors")
        
        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.tensors_dir, exist_ok=True)
        
        # Load existing history
        self.history = self._load_history()
    
    def _load_history(self) -> dict:
        """Load existing checkpoint history from TOON"""
        if os.path.exists(self.toon_file):
            try:
                # Parse TOON manually (simple implementation)
                return self._parse_toon(self.toon_file)
            except Exception as e:
                print(f"Warning: Could not load checkpoint history: {e}")
        return {'metadata': {}, 'checkpoints': [], 'deltas': []}
    
    def _parse_toon(self, filepath: str) -> dict:
        """Simple TOON parser for checkpoint files"""
        result = {'metadata': {}, 'checkpoints': [], 'deltas': []}
        current_section = None
        current_item = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Section headers
                if line.startswith('[[') and line.endswith(']]'):
                    # Array of tables
                    if current_item and current_section:
                        result[current_section].append(current_item)
                    current_section = line[2:-2]
                    current_item = {}
                elif line.startswith('[') and line.endswith(']'):
                    # Regular table
                    if current_item and current_section:
                        if isinstance(result.get(current_section), list):
                            result[current_section].append(current_item)
                    current_section = line[1:-1]
                    if current_section == 'metadata':
                        current_item = result['metadata']
                    else:
                        current_item = {}
                elif '=' in line:
                    # Key-value pair
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith('[') and value.endswith(']'):
                        # Simple array parsing
                        inner = value[1:-1]
                        if inner:
                            value = [v.strip().strip('"') for v in inner.split(',')]
                        else:
                            value = []
                    elif value == 'true':
                        value = True
                    elif value == 'false':
                        value = False
                    else:
                        try:
                            value = float(value) if '.' in value else int(value)
                        except:
                            pass
                    
                    current_item[key] = value
        
        # Don't forget last item
        if current_item and current_section:
            if isinstance(result.get(current_section), list):
                result[current_section].append(current_item)
        
        return result
    
    def save_checkpoint_toon(
        self, 
        checkpoint_manager: 'HebbianCheckpointManager',
        session_id: str = None
    ) -> str:
        """
        Save checkpoint manager state to TOON format.
        
        Args:
            checkpoint_manager: The checkpoint manager to save
            session_id: Optional session identifier
            
        Returns:
            Path to saved TOON file
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build TOON content
        lines = []
        
        # Metadata section
        lines.append("# BDH Narrative Classifier - Checkpoint History")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append("")
        lines.append("[metadata]")
        lines.append(f'session_id = "{session_id}"')
        lines.append(f'created_at = "{datetime.now().isoformat()}"')
        lines.append(f"total_checkpoints = {len(checkpoint_manager.checkpoints)}")
        lines.append(f"total_deltas = {len(checkpoint_manager.delta_checkpoints)}")
        lines.append(f"current_step = {checkpoint_manager.current_step}")
        
        if checkpoint_manager.checkpoints:
            layers = list(checkpoint_manager.checkpoints[-1].layer_states.keys())
            lines.append(f"layers_tracked = {len(layers)}")
        lines.append("")
        
        # Checkpoint entries
        for i, cp in enumerate(checkpoint_manager.checkpoints):
            lines.append(f"[[checkpoints]]")
            lines.append(f"index = {i}")
            lines.append(f"step = {cp.step}")
            lines.append(f"timestamp = {cp.timestamp}")
            
            # Save tensor references (actual tensors saved separately)
            tensor_file = os.path.join(self.tensors_dir, f"cp_{session_id}_{cp.step}.pt")
            torch.save(cp.layer_states, tensor_file)
            lines.append(f'tensor_file = "{os.path.basename(tensor_file)}"')
            
            # Metadata
            if cp.metadata:
                for key, value in cp.metadata.items():
                    if isinstance(value, str):
                        lines.append(f'{key} = "{value}"')
                    elif isinstance(value, bool):
                        lines.append(f'{key} = {"true" if value else "false"}')
                    elif isinstance(value, (int, float)):
                        lines.append(f'{key} = {value}')
            lines.append("")
        
        # Delta entries with change tracking
        for i, delta in enumerate(checkpoint_manager.delta_checkpoints):
            lines.append(f"[[deltas]]")
            lines.append(f"index = {i}")
            lines.append(f"from_step = {delta.from_step}")
            lines.append(f"to_step = {delta.to_step}")
            
            # Delta norms for change tracking
            if 'delta_norm' in delta.metadata:
                norms = delta.metadata['delta_norm']
                # Aggregate by layer
                layer_changes = {}
                for key, norm in norms.items():
                    layer = key.split('/')[0]
                    if layer not in layer_changes:
                        layer_changes[layer] = 0.0
                    layer_changes[layer] += norm
                
                lines.append("# Layer changes (L2 norm of delta)")
                for layer, total_norm in layer_changes.items():
                    safe_key = layer.replace('_', '-')
                    lines.append(f'{safe_key} = {total_norm:.6f}')
            lines.append("")
        
        # Write TOON file
        toml_content = "\n".join(lines)
        output_path = os.path.join(self.storage_dir, f"checkpoint_{session_id}.toon")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
        
        # Also update main history file
        self._update_main_history(session_id, checkpoint_manager)
        
        print(f"Checkpoint saved to TOON: {output_path}")
        return output_path
    
    def _update_main_history(
        self, 
        session_id: str, 
        checkpoint_manager: 'HebbianCheckpointManager'
    ) -> None:
        """Update the main checkpoint history file"""
        lines = []
        lines.append("# BDH Checkpoint History - All Sessions")
        lines.append(f"# Last updated: {datetime.now().isoformat()}")
        lines.append("")
        lines.append("[metadata]")
        lines.append(f'last_session = "{session_id}"')
        lines.append(f'last_updated = "{datetime.now().isoformat()}"')
        lines.append("")
        
        # Add current session as entry
        lines.append("[[sessions]]")
        lines.append(f'session_id = "{session_id}"')
        lines.append(f"checkpoints = {len(checkpoint_manager.checkpoints)}")
        lines.append(f"current_step = {checkpoint_manager.current_step}")
        lines.append(f'toon_file = "checkpoint_{session_id}.toon"')
        lines.append("")
        
        with open(self.toon_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
    
    def load_checkpoint_toon(
        self, 
        toon_path: str,
        config: 'HebbianConfig'
    ) -> 'HebbianCheckpointManager':
        """
        Load checkpoint manager from TOON file.
        
        Args:
            toon_path: Path to TOON checkpoint file
            config: HebbianConfig for the manager
            
        Returns:
            Restored HebbianCheckpointManager
        """
        data = self._parse_toon(toon_path)
        
        manager = HebbianCheckpointManager(config)
        
        # Restore checkpoints
        for cp_data in data.get('checkpoints', []):
            tensor_file = cp_data.get('tensor_file', '')
            tensor_path = os.path.join(self.tensors_dir, tensor_file)
            
            if os.path.exists(tensor_path):
                layer_states = torch.load(tensor_path)
                
                checkpoint = Checkpoint(
                    step=cp_data.get('step', 0),
                    timestamp=cp_data.get('timestamp', 0.0),
                    layer_states=layer_states,
                    metadata={k: v for k, v in cp_data.items() 
                             if k not in ['index', 'step', 'timestamp', 'tensor_file']}
                )
                manager.checkpoints.append(checkpoint)
        
        if manager.checkpoints:
            manager.current_step = manager.checkpoints[-1].step
        
        print(f"Loaded {len(manager.checkpoints)} checkpoints from TOON")
        return manager
    
    def get_change_summary(self) -> str:
        """
        Get a TOON-formatted summary of recent changes.
        Useful for memory persistence in conversations.
        """
        lines = []
        lines.append("# Checkpoint Change Summary")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append("")
        
        if not self.history.get('checkpoints'):
            lines.append("# No checkpoints recorded yet")
            return "\n".join(lines)
        
        lines.append("[summary]")
        lines.append(f"total_sessions = {len(self.history.get('sessions', []))}")
        
        last_session = self.history.get('metadata', {}).get('last_session', 'unknown')
        lines.append(f'last_session = "{last_session}"')
        
        return "\n".join(lines)


def save_checkpoint_as_toon(
    checkpoint_manager: HebbianCheckpointManager,
    output_dir: str = "checkpoints",
    session_id: str = None
) -> str:
    """
    Convenience function to save checkpoint manager to TOON.
    
    Args:
        checkpoint_manager: The manager to save
        output_dir: Directory for checkpoint files
        session_id: Optional session identifier
        
    Returns:
        Path to saved TOON file
    """
    storage = CheckpointTOONStorage(output_dir)
    return storage.save_checkpoint_toon(checkpoint_manager, session_id)


def load_checkpoint_from_toon(
    toon_path: str,
    config: HebbianConfig,
    storage_dir: str = "checkpoints"
) -> HebbianCheckpointManager:
    """
    Convenience function to load checkpoint manager from TOON.
    
    Args:
        toml_path: Path to TOON file
        config: HebbianConfig for the manager
        storage_dir: Directory containing tensor files
        
    Returns:
        Restored HebbianCheckpointManager
    """
    storage = CheckpointTOONStorage(storage_dir)
    return storage.load_checkpoint_toon(toon_path, config)

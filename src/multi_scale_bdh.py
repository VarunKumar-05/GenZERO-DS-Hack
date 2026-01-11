# Multi-Scale BDH Architecture with Temporal Dynamics
# Copyright 2026 - Kharagpur Data Science Hackathon

import dataclasses
import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelConfig, TemporalScaleConfig


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    """Generate rotary position encoding frequencies"""
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class HomeostaticNorm(nn.Module):
    """
    "Better Suspension": Homeostatic Plasticity Normalization.
    
    Dynamically scales activations to maintain a target sparsity level,
    preventing "bumpy rides" (vanishing or exploding activity) in SNN-like
    architectures.
    """
    def __init__(self, dim: int, target_activity: float = 0.1, decay: float = 0.99):
        super().__init__()
        self.target_activity = target_activity
        self.decay = decay
        # Learnable scaling parameters (plasticity)
        self.scale = nn.Parameter(torch.ones(dim))
        # Running mean of activity (not trained, updated via EMA)
        self.register_buffer('running_activity', torch.ones(dim) * target_activity)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update running activity estimate
            # Average over all dims except the last (feature) dim
            dims_to_reduce = list(range(x.dim() - 1))
            batch_activity = x.mean(dim=dims_to_reduce).detach()
            
            self.running_activity.data.mul_(self.decay).add_(
                batch_activity * (1 - self.decay)
            )
            
        # Homeostatic scaling: if activity > target, downscale; if < target, upscale
        # Regulation factor = target / (current + epsilon)
        regulation = self.target_activity / (self.running_activity + 1e-6)
        
        # Apply smooth scaling (don't force hard jump)
        return x * regulation.sqrt() * self.scale


class RoPEAttention(nn.Module):
    """Rotary Position Embedding Attention from BDH"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = RoPEAttention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        assert self.freqs.dtype == torch.float32
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        QR = self.rope(r_phases, Q)
        KR = self.rope(r_phases, K)

        # Causal attention with lower triangular mask
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDHLayer(nn.Module):
    """Single BDH layer with sparse activations"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        
        self.attn = RoPEAttention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        # "Better Suspension": Homeostatic normalization
        self.pro_homeostasis = HomeostaticNorm(D)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            y: output tensor
            sparse_state: sparse activation state for Hebbian learning
        """
        C = self.config
        B, _, T, D = x.size()
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        
        # Encode to latent space
        x_latent = x @ self.encoder
        x_sparse = F.relu(x_latent)  # Sparse activations
        
        # Attention over sparse representations
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
        yKV = self.ln(yKV)
        
        # Value encoding and gating
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse  # Gated sparse state
        
        xy_sparse = self.drop(xy_sparse)
        
        # Decode back to embedding space
        yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
        y = self.ln(yMLP)
        
        # Apply homeostatic regulation to residual path
        y = self.pro_homeostasis(y)
        
        return self.ln(x + y), xy_sparse


class TemporalScaleBDH(nn.Module):
    """BDH encoder for a single temporal scale"""
    
    def __init__(self, config: ModelConfig, window_size: int, stride: int):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.stride = stride
        
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.ln_in = nn.LayerNorm(config.n_embd, elementwise_affine=False, bias=False)
        
        self.layers = nn.ModuleList([
            BDHLayer(config) for _ in range(config.n_layer)
        ])
        
        self.ln_out = nn.LayerNorm(config.n_embd)
        
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input_ids: [B, T] token indices
        Returns:
            output: [B, T, D] encoded representations
            layer_states: list of sparse states from each layer
        """
        B, T = input_ids.size()
        
        # Embed and normalize
        x = self.embed(input_ids).unsqueeze(1)  # [B, 1, T, D]
        x = self.ln_in(x)
        
        layer_states = []
        for layer in self.layers:
            x, sparse_state = layer(x)
            layer_states.append(sparse_state)
        
        output = self.ln_out(x.squeeze(1))  # [B, T, D]
        return output, layer_states


class MultiScaleBDH(nn.Module):
    """
    Multi-Scale BDH with temporal dynamics at sentence, paragraph, and chapter levels.
    Implements cross-scale attention fusion for comprehensive narrative understanding.
    """
    
    def __init__(self, model_config: ModelConfig, temporal_config: TemporalScaleConfig):
        super().__init__()
        self.model_config = model_config
        self.temporal_config = temporal_config
        
        # Three temporal scale encoders
        self.sentence_encoder = TemporalScaleBDH(
            model_config, 
            temporal_config.sentence_window, 
            temporal_config.sentence_stride
        )
        self.paragraph_encoder = TemporalScaleBDH(
            model_config,
            temporal_config.paragraph_window,
            temporal_config.paragraph_stride
        )
        self.chapter_encoder = TemporalScaleBDH(
            model_config,
            temporal_config.chapter_window,
            temporal_config.chapter_stride
        )
        
        # Cross-scale fusion
        D = model_config.n_embd
        self.scale_fusion = nn.Sequential(
            nn.Linear(D * 3, D * 2),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(D * 2, D),
            nn.LayerNorm(D)
        )
        
        # "Automatic Transmission": Dynamic Gating Network
        self.gating_net = nn.Sequential(
            nn.Linear(D * 3, D),
            nn.LeakyReLU(),
            nn.Linear(D, 3)  # Predict weights for 3 scales
        )
        # Fallback static weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        return_all_states: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            input_ids: [B, T] token indices
            return_all_states: whether to return all layer states for checkpointing
            
        Returns:
            fused_output: [B, T, D] fused multi-scale representation
            states_dict: dictionary containing states from all scales
        """
        B, T = input_ids.size()
        D = self.model_config.n_embd
        
        # Process at each temporal scale
        # Sentence level - process full sequence
        sent_out, sent_states = self.sentence_encoder(input_ids)
        
        # Paragraph level - use larger chunks, interpolate to match
        para_out, para_states = self.paragraph_encoder(input_ids)
        
        # Chapter level - use even larger chunks
        chap_out, chap_states = self.chapter_encoder(input_ids)
        
        # Ensure all outputs have same shape via interpolation if needed
        if para_out.size(1) != T:
            para_out = F.interpolate(
                para_out.transpose(1, 2), size=T, mode='linear', align_corners=False
            ).transpose(1, 2)
        if chap_out.size(1) != T:
            chap_out = F.interpolate(
                chap_out.transpose(1, 2), size=T, mode='linear', align_corners=False
            ).transpose(1, 2)
        
        # "Automatic Transmission": Compute dynamic gates
        # Uses content from all scales to determine optimal weighing
        raw_concat = torch.cat([sent_out, para_out, chap_out], dim=-1) # [B, T, D*3]
        gates = F.softmax(self.gating_net(raw_concat), dim=-1) # [B, T, 3]
        
        # Fuse with dynamic weights
        weighted_concat = torch.cat([
            gates[..., 0:1] * sent_out,
            gates[..., 1:2] * para_out,
            gates[..., 2:3] * chap_out
        ], dim=-1)  # [B, T, D*3]
        
        fused_output = self.scale_fusion(weighted_concat)  # [B, T, D]
        
        states_dict = {
            'sentence': sent_states,
            'paragraph': para_states,
            'chapter': chap_states,
            'scale_weights': gates.mean(dim=(0, 1)).detach()
        }
        
        if return_all_states:
            return fused_output, states_dict
        return fused_output, {'scale_weights': gates.mean(dim=(0, 1)).detach()}
    
    def get_layer_parameters(self) -> dict:
        """Get parameters organized by layer for Hebbian checkpointing"""
        params = {}
        for name, encoder in [
            ('sentence', self.sentence_encoder),
            ('paragraph', self.paragraph_encoder),
            ('chapter', self.chapter_encoder)
        ]:
            for i, layer in enumerate(encoder.layers):
                params[f'{name}_layer_{i}'] = {
                    'encoder': layer.encoder.data.clone(),
                    'encoder_v': layer.encoder_v.data.clone(),
                    'decoder': layer.decoder.data.clone()
                }
        return params

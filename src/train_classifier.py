# Training Pipeline for BDH Narrative Consistency Classifier
# Copyright 2026 - Kharagpur Data Science Hackathon

"""
Complete training script with:
- Multi-scale BDH encoding
- Hebbian learning checkpoints
- Trajectory attention pooling
- Classification training
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from multi_scale_bdh import MultiScaleBDH
from hebbian_checkpoint import HebbianCheckpointManager, HebbianLearningRule
from trajectory_attention import TrajectoryAttentionPooling, EvidenceExtractor
from data_pipeline import NarrativeDataPipeline, NarrativeExample
from text_processor import TextProcessor, BackstoryAligner

# Tokenizer (using simple byte-level for compatibility with BDH)
try:
    from transformers import AutoTokenizer
    USE_HF_TOKENIZER = True
except ImportError:
    USE_HF_TOKENIZER = False


class SimpleTokenizer:
    """Simple byte-level tokenizer when transformers not available"""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        tokens = list(text.encode('utf-8', errors='ignore'))[:max_length - 1]
        tokens.append(self.eos_token_id)
        return tokens
    
    def __call__(
        self, 
        text: str, 
        max_length: int = 512, 
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = None
    ) -> Dict:
        tokens = self.encode(text, max_length)
        
        # Pad to max length
        if padding == 'max_length':
            attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        else:
            attention_mask = [1] * len(tokens)
        
        result = {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
        
        if return_tensors == 'pt':
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result


class NarrativeDataset(Dataset):
    """Dataset for narrative consistency classification"""
    
    def __init__(
        self, 
        examples: List[NarrativeExample],
        tokenizer,
        max_length: int = 512,
        max_narrative_length: int = 2048
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_narrative_length = max_narrative_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize backstory
        backstory_enc = self.tokenizer(
            example.backstory,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize narrative (use relevant portions from processed text)
        narrative_text = ""
        if example.processed_narrative:
            # Take first N sentences as context
            narrative_text = " ".join(example.processed_narrative.sentences[:50])
        elif example.book_text:
            narrative_text = example.book_text[:5000]
        
        narrative_enc = self.tokenizer(
            narrative_text if narrative_text else "No narrative available.",
            max_length=self.max_narrative_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels
        label = example.label if example.label is not None else 0
        
        return {
            'backstory_ids': backstory_enc['input_ids'].squeeze(0),
            'backstory_mask': backstory_enc['attention_mask'].squeeze(0),
            'narrative_ids': narrative_enc['input_ids'].squeeze(0),
            'narrative_mask': narrative_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'example_id': example.id
        }


class NarrativeClassifier(nn.Module):
    """
    Complete classifier combining:
    - Multi-scale BDH for encoding
    - Trajectory attention for pooling
    - Binary classification head
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Multi-scale BDH encoder
        self.encoder = MultiScaleBDH(config.model, config.temporal)
        
        # Trajectory attention pooling
        self.pooling = TrajectoryAttentionPooling(config.model, num_classes=2)
        
        # Backstory encoder (shared architecture, separate instance)
        self.backstory_encoder = MultiScaleBDH(config.model, config.temporal)
        
        # Cross-attention between backstory and narrative
        D = config.model.n_embd
        self.cross_attention = nn.MultiheadAttention(D, config.model.n_head, batch_first=True)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(D, 2)
        )
    
    def forward(
        self, 
        backstory_ids: torch.Tensor,
        narrative_ids: torch.Tensor,
        backstory_mask: Optional[torch.Tensor] = None,
        narrative_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for classification.
        
        Returns:
            logits: [B, 2] classification logits
            extras: dict with attention weights for evidence extraction
        """
        # Encode narrative with multi-scale BDH
        narrative_repr, narrative_states = self.encoder(narrative_ids, return_all_states=True)
        
        # Encode backstory
        backstory_repr, _ = self.backstory_encoder(backstory_ids)
        
        # Cross attention: backstory attends to narrative
        cross_out, cross_attn = self.cross_attention(
            backstory_repr, 
            narrative_repr, 
            narrative_repr,
            key_padding_mask=(narrative_mask == 0) if narrative_mask is not None else None,
            need_weights=True
        )
        
        # Pool backstory representation (return pooled D-dim representation, not logits)
        backstory_pooled, _ = self.pooling(cross_out, backstory_mask, return_pooled=True)
        
        # Pool narrative representation
        narrative_pooled, narrative_attn = self.pooling(
            narrative_repr, narrative_mask, 
            return_attention=True, return_pooled=True
        )
        
        # Fuse and classify
        combined = torch.cat([backstory_pooled, narrative_pooled], dim=-1)
        logits = self.fusion(combined)
        
        extras = {
            'cross_attention': cross_attn,
            'narrative_attention': narrative_attn,
            'narrative_states': narrative_states
        }
        
        return logits, extras


def train_epoch(
    model: NarrativeClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_manager: HebbianCheckpointManager,
    epoch: int,
    config: Config
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        backstory_ids = batch['backstory_ids'].to(device)
        backstory_mask = batch['backstory_mask'].to(device)
        narrative_ids = batch['narrative_ids'].to(device)
        narrative_mask = batch['narrative_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, extras = model(backstory_ids, narrative_ids, backstory_mask, narrative_mask)
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Checkpoint at intervals
        global_step = epoch * len(dataloader) + batch_idx
        if global_step % config.hebbian.checkpoint_interval == 0:
            layer_states = model.encoder.get_layer_parameters()
            checkpoint_manager.save_checkpoint(
                layer_states, 
                global_step,
                metadata={'loss': loss.item(), 'accuracy': correct / total}
            )
        
        # Log progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }


def evaluate(
    model: NarrativeClassifier,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            backstory_ids = batch['backstory_ids'].to(device)
            backstory_mask = batch['backstory_mask'].to(device)
            narrative_ids = batch['narrative_ids'].to(device)
            narrative_mask = batch['narrative_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(backstory_ids, narrative_ids, backstory_mask, narrative_mask)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }


def main():
    parser = argparse.ArgumentParser(description='Train BDH Narrative Classifier')
    parser.add_argument('--train-csv', default='Dataset/train.csv', help='Path to training CSV')
    parser.add_argument('--books-dir', default='Dataset/Books', help='Path to books directory')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (1 epoch, small data)')
    args = parser.parse_args()
    
    # Configuration
    config = get_config()
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize tokenizer
    if USE_HF_TOKENIZER:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = SimpleTokenizer()
        config.model.vocab_size = 256
    
    # Load data
    print("Loading data...")
    pipeline = NarrativeDataPipeline(config.pathway, 'Dataset', args.books_dir)
    train_examples = pipeline.load_train_csv(args.train_csv)
    train_examples = pipeline.prepare_examples(train_examples, load_books=True)
    
    if args.test_mode:
        train_examples = train_examples[:5]
        config.training.epochs = 1
    
    print(f"Loaded {len(train_examples)} training examples")
    
    # Create dataset and dataloader
    dataset = NarrativeDataset(train_examples, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = NarrativeClassifier(config).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    
    # Hebbian checkpoint manager
    checkpoint_manager = HebbianCheckpointManager(config.hebbian)
    
    # Training loop
    print(f"Starting training for {config.training.epochs} epochs...")
    best_accuracy = 0.0
    
    for epoch in range(config.training.epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.training.epochs} ===")
        
        # Train
        train_metrics = train_epoch(
            model, dataloader, optimizer, device, 
            checkpoint_manager, epoch, config
        )
        
        scheduler.step()
        
        print(f"Epoch {epoch + 1} - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Save best model
        if train_metrics['accuracy'] > best_accuracy:
            best_accuracy = train_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'config': config
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  Saved best model (accuracy: {best_accuracy:.4f})")
    
    # Save final model
    torch.save({
        'epoch': config.training.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': train_metrics['accuracy'],
        'config': config,
        'checkpoint_summary': checkpoint_manager.get_checkpoint_summary()
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")
    print(f"Checkpoints saved: {checkpoint_manager.get_checkpoint_summary()}")


if __name__ == '__main__':
    main()

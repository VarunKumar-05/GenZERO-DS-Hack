# Test script for BDH Narrative Classifier
# Copyright 2026 - Kharagpur Data Science Hackathon

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, 'src')

import torch
from src.config import get_config
from src.data_pipeline import NarrativeDataPipeline
from src.train_classifier import NarrativeClassifier, NarrativeDataset, SimpleTokenizer
from src.hebbian_checkpoint import HebbianCheckpointManager

def test_pipeline():
    """Test the complete pipeline"""
    print("=" * 50)
    print("BDH Narrative Classifier - Pipeline Test")
    print("=" * 50)
    
    # Setup
    config = get_config()
    config.model.vocab_size = 256  # Byte-level
    device = torch.device('cpu')
    
    # 1. Test data loading
    print("\n[1] Testing data loading...")
    pipeline = NarrativeDataPipeline(config.pathway, 'Dataset', 'Dataset/Books')
    examples = pipeline.load_train_csv('Dataset/train.csv')
    examples = pipeline.prepare_examples(examples[:3])  # Only 3 for test
    print(f"    Loaded {len(examples)} examples")
    print(f"    First: {examples[0].book_name} - {examples[0].character}")
    
    # 2. Test dataset
    print("\n[2] Testing dataset creation...")
    tokenizer = SimpleTokenizer()
    dataset = NarrativeDataset(examples, tokenizer)
    sample = dataset[0]
    print(f"    Backstory shape: {sample['backstory_ids'].shape}")
    print(f"    Narrative shape: {sample['narrative_ids'].shape}")
    print(f"    Label: {sample['label'].item()}")
    
    # 3. Test model
    print("\n[3] Testing model initialization...")
    model = NarrativeClassifier(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {param_count:,}")
    
    # 4. Test forward pass
    print("\n[4] Testing forward pass...")
    with torch.no_grad():
        logits, extras = model(
            sample['backstory_ids'].unsqueeze(0),
            sample['narrative_ids'].unsqueeze(0)
        )
    pred = "consistent" if logits.argmax().item() == 1 else "inconsistent"
    prob = torch.softmax(logits, dim=-1)[0, logits.argmax().item()].item()
    print(f"    Logits shape: {logits.shape}")
    print(f"    Prediction: {pred} ({prob:.1%} confidence)")
    
    # 5. Test Hebbian checkpointing
    print("\n[5] Testing Hebbian checkpoint manager...")
    checkpoint_manager = HebbianCheckpointManager(config.hebbian)
    layer_states = model.encoder.get_layer_parameters()
    checkpoint_manager.save_checkpoint(layer_states, step=0)
    checkpoint_manager.save_checkpoint(layer_states, step=100)
    summary = checkpoint_manager.get_checkpoint_summary()
    print(f"    Checkpoints saved: {summary['count']}")
    print(f"    Layers tracked: {len(summary['layers_tracked'])}")
    
    print("\n" + "=" * 50)
    print("SUCCESS: All tests passed!")
    print("=" * 50)
    
    return True

if __name__ == '__main__':
    test_pipeline()

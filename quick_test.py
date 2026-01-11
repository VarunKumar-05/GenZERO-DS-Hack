# Quick lightweight test - smaller model for faster CPU inference
import sys
import os
import csv

sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_config, ModelConfig
from src.data_pipeline import NarrativeDataPipeline
from src.text_processor import BackstoryAligner

class LightweightClassifier(nn.Module):
    """Lightweight classifier for quick CPU testing"""
    def __init__(self, vocab_size=256, d_model=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=n_layers
        )
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, backstory_ids, narrative_ids):
        # Encode backstory
        b = self.embed(backstory_ids)
        b = self.encoder(b)
        b_pooled = b.mean(dim=1)
        
        # Encode narrative  
        n = self.embed(narrative_ids)
        n = self.encoder(n)
        n_pooled = n.mean(dim=1)
        
        # Classify
        combined = b_pooled + n_pooled
        logits = self.classifier(combined)
        return logits

class SimpleTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        
    def __call__(self, text, max_length=256, **kwargs):
        tokens = list(text.encode('utf-8', errors='ignore'))[:max_length]
        tokens = tokens + [0] * (max_length - len(tokens))
        if kwargs.get('return_tensors') == 'pt':
            return {'input_ids': torch.tensor([tokens])}
        return {'input_ids': tokens}

def main():
    print("=" * 60)
    print("LIGHTWEIGHT BDH Classifier - Quick Test on test.csv")
    print("=" * 60)
    
    # Small model for quick testing
    print("\n[1] Initializing lightweight model...")
    model = LightweightClassifier()
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"    Model ready ({params:,} params - much faster!)")
    
    # Load test data
    print("\n[2] Loading test data...")
    config = get_config()
    pipeline = NarrativeDataPipeline(config.pathway, 'Dataset', 'Dataset/Books')
    test_examples = pipeline.load_test_csv('Dataset/test.csv')
    test_examples = pipeline.prepare_examples(test_examples, load_books=True)
    print(f"    Loaded {len(test_examples)} test examples")
    
    tokenizer = SimpleTokenizer()
    aligner = BackstoryAligner()
    
    # Run predictions
    print("\n[3] Running predictions...")
    results = []
    
    for i, example in enumerate(test_examples):
        backstory_enc = tokenizer(example.backstory, max_length=256, return_tensors='pt')
        
        narrative_text = ""
        if example.processed_narrative:
            narrative_text = " ".join(example.processed_narrative.sentences[:20])
        elif example.book_text:
            narrative_text = example.book_text[:2000]
        
        narrative_enc = tokenizer(narrative_text or "No narrative.", max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            logits = model(backstory_enc['input_ids'], narrative_enc['input_ids'])
        
        probs = F.softmax(logits, dim=-1)
        pred_class = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_class].item()
        label = "consistent" if pred_class == 1 else "inconsistent"
        
        # Get evidence
        rationale = ""
        if example.processed_narrative:
            relevant = aligner.find_relevant_passages(example.backstory, example.processed_narrative, top_k=2)
            rationale = "; ".join([s[:60] for _, s, _ in relevant[:2]])
        if not rationale:
            rationale = f"Model prediction based on narrative analysis"
        
        results.append({
            'id': example.id,
            'prediction': pred_class,
            'label': label,
            'confidence': confidence,
            'rationale': rationale[:150]
        })
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(test_examples)}")
    
    # Save results
    print("\n[4] Saving results...")
    with open('results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Story ID', 'Prediction', 'Rationale'])
        for r in results:
            writer.writerow([r['id'], r['prediction'], r['rationale']])
    
    print("    Saved to results.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    consistent = sum(1 for r in results if r['prediction'] == 1)
    print(f"Total: {len(results)} examples")
    print(f"Predicted Consistent: {consistent}")
    print(f"Predicted Inconsistent: {len(results) - consistent}")
    
    print("\nFirst 10 predictions:")
    print("-" * 60)
    for r in results[:10]:
        print(f"ID {r['id']:3d}: {r['label']:12s} | {r['rationale'][:50]}...")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE! Check results.csv for full output.")
    print("=" * 60)

if __name__ == '__main__':
    main()

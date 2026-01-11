# Quick test script to run inference on test.csv
# Uses untrained model for demonstration

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, 'src')

import torch
from src.config import get_config
from src.data_pipeline import NarrativeDataPipeline
from src.train_classifier import NarrativeClassifier, SimpleTokenizer
from src.text_processor import BackstoryAligner

def run_test():
    print("=" * 60)
    print("BDH Narrative Consistency Classifier - Test Run")
    print("=" * 60)
    
    # Setup
    config = get_config()
    config.model.vocab_size = 256
    device = torch.device('cpu')
    
    # Initialize model
    print("\n[1] Initializing model...")
    model = NarrativeClassifier(config).to(device)
    model.eval()
    print(f"    Model ready ({sum(p.numel() for p in model.parameters()):,} params)")
    
    # Load test data
    print("\n[2] Loading test data...")
    pipeline = NarrativeDataPipeline(config.pathway, 'Dataset', 'Dataset/Books')
    test_examples = pipeline.load_test_csv('Dataset/test.csv')
    test_examples = pipeline.prepare_examples(test_examples, load_books=True)
    print(f"    Loaded {len(test_examples)} test examples")
    
    # Tokenizer
    tokenizer = SimpleTokenizer()
    aligner = BackstoryAligner()
    
    # Run predictions
    print("\n[3] Running predictions...")
    results = []
    
    for i, example in enumerate(test_examples):
        # Tokenize
        backstory_enc = tokenizer(
            example.backstory,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        narrative_text = ""
        if example.processed_narrative:
            narrative_text = " ".join(example.processed_narrative.sentences[:50])
        elif example.book_text:
            narrative_text = example.book_text[:5000]
        
        narrative_enc = tokenizer(
            narrative_text if narrative_text else "No narrative.",
            max_length=2048,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Forward pass
        with torch.no_grad():
            logits, extras = model(
                backstory_enc['input_ids'],
                narrative_enc['input_ids']
            )
        
        probs = torch.softmax(logits, dim=-1)
        pred_class = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_class].item()
        label = "consistent" if pred_class == 1 else "inconsistent"
        
        # Find relevant evidence
        evidence = []
        if example.processed_narrative:
            relevant = aligner.find_relevant_passages(
                example.backstory,
                example.processed_narrative,
                top_k=3
            )
            evidence = [sent[:80] + "..." for _, sent, _ in relevant[:2]]
        
        rationale = "; ".join(evidence) if evidence else f"Model prediction: {label}"
        
        results.append({
            'id': example.id,
            'prediction': 1 if pred_class == 1 else 0,
            'label': label,
            'confidence': confidence,
            'rationale': rationale[:150]
        })
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(test_examples)} examples")
    
    # Save results
    print("\n[4] Saving results...")
    output_path = 'results.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Story ID', 'Prediction', 'Rationale'])
        for r in results:
            writer.writerow([r['id'], r['prediction'], r['rationale']])
    
    print(f"    Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    consistent = sum(1 for r in results if r['prediction'] == 1)
    print(f"Total predictions: {len(results)}")
    print(f"Consistent: {consistent} ({consistent/len(results)*100:.1f}%)")
    print(f"Inconsistent: {len(results) - consistent} ({(len(results)-consistent)/len(results)*100:.1f}%)")
    
    # Show sample predictions
    print("\nSample predictions:")
    print("-" * 60)
    for r in results[:5]:
        print(f"ID {r['id']:3d}: {r['label']:12s} ({r['confidence']:.1%})")
        print(f"         {r['rationale'][:70]}...")
        print()
    
    return results

if __name__ == '__main__':
    run_test()

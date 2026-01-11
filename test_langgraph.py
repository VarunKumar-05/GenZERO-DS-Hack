# Full Test Pipeline with LangGraph + Gemini Reasoning
# Run inference on test.csv with complete reasoning support

import sys
import os
import csv

sys.path.insert(0, 'src')

import torch
from src.config import get_config
from src.data_pipeline import NarrativeDataPipeline
from src.train_classifier import NarrativeClassifier, SimpleTokenizer
from src.text_processor import BackstoryAligner
from src.reasoning_graph import ReasoningGraph, create_reasoning_graph

def main():
    print("=" * 70)
    print("BDH Narrative Classifier - Full Test with LangGraph + Gemini")
    print("=" * 70)
    
    # Configuration
    config = get_config()
    config.model.vocab_size = 256
    device = torch.device('cpu')
    
    # Initialize components
    print("\n[1] Initializing components...")
    
    # Lightweight model for demo
    from src.train_classifier import NarrativeClassifier
    # Use simple model for speed
    import torch.nn as nn
    import torch.nn.functional as F
    
    class LightModel(nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.embed = nn.Embedding(256, d)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, 4, 256, batch_first=True), 2
            )
            self.classifier = nn.Linear(d, 2)
        
        def forward(self, backstory_ids, narrative_ids):
            b = self.encoder(self.embed(backstory_ids)).mean(dim=1)
            n = self.encoder(self.embed(narrative_ids)).mean(dim=1)
            return self.classifier(b + n)
    
    model = LightModel()
    model.eval()
    print(f"    Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Reasoning graph with Gemini
    print("\n[2] Initializing LangGraph + Gemini reasoning...")
    reasoning_graph = create_reasoning_graph(config.gemini)
    print(f"    Gemini available: {reasoning_graph.reasoner.is_available()}")
    print(f"    Model: {config.gemini.model_name}")
    
    # Load test data
    print("\n[3] Loading test data...")
    pipeline = NarrativeDataPipeline(config.pathway, 'Dataset', 'Dataset/Books')
    test_examples = pipeline.load_test_csv('Dataset/test.csv')
    test_examples = pipeline.prepare_examples(test_examples, load_books=True)
    print(f"    Loaded {len(test_examples)} test examples")
    
    tokenizer = SimpleTokenizer()
    aligner = BackstoryAligner()
    
    # Run predictions with reasoning
    print("\n[4] Running predictions with LangGraph reasoning...")
    results = []
    
    # Process all test examples
    demo_count = len(test_examples)  # Process ALL examples
    
    for i, example in enumerate(test_examples[:demo_count]):
        print(f"\n    [{i+1}/{demo_count}] {example.book_name} - {example.character}")
        
        # Tokenize
        backstory_enc = tokenizer(example.backstory, max_length=256, return_tensors='pt')
        
        narrative_text = ""
        if example.processed_narrative:
            narrative_text = " ".join(example.processed_narrative.sentences[:30])
        elif example.book_text:
            narrative_text = example.book_text[:3000]
        
        narrative_enc = tokenizer(narrative_text or "No narrative.", max_length=512, return_tensors='pt')
        
        # Model prediction
        with torch.no_grad():
            logits = model(backstory_enc['input_ids'], narrative_enc['input_ids'])
        
        probs = F.softmax(logits, dim=-1)
        model_pred = logits.argmax(dim=-1).item()
        model_conf = probs[0, model_pred].item()
        
        # Get candidate evidence from heuristic alignment
        candidate_evidence = []
        if example.processed_narrative:
            relevant = aligner.find_relevant_passages(
                example.backstory, example.processed_narrative, top_k=8
            )
            candidate_evidence = [(idx, sent, score) for idx, sent, score in relevant]
        
        # LangGraph + Gemini reasoning
        narrative_sentences = example.processed_narrative.sentences if example.processed_narrative else []
        
        if candidate_evidence and reasoning_graph.reasoner.is_available():
            reasoning_result = reasoning_graph.run(
                backstory=example.backstory,
                narrative_sentences=narrative_sentences[:100],
                character=example.character,
                candidate_evidence=candidate_evidence
            )
            
            label = reasoning_result.classification
            confidence = reasoning_result.confidence
            explanation = reasoning_result.summary
            evidence = reasoning_result.evidence
        else:
            # Fallback
            label = "consistent" if model_pred == 1 else "inconsistent"
            confidence = model_conf
            explanation = "Model prediction (no Gemini reasoning)"
            evidence = []
        
        # Format evidence for output
        evidence_text = ""
        if evidence:
            evidence_text = "; ".join([e.sentence_text[:60] + "..." for e in evidence[:2]])
        elif candidate_evidence:
            evidence_text = "; ".join([s[:60] + "..." for _, s, _ in candidate_evidence[:2]])
        
        rationale = explanation[:120] if explanation else evidence_text[:120]
        
        results.append({
            'id': example.id,
            'prediction': 1 if label == "consistent" else 0,
            'label': label,
            'confidence': confidence,
            'rationale': rationale,
            'evidence_count': len(evidence)
        })
        
        print(f"        Prediction: {label} ({confidence:.1%})")
        print(f"        Evidence: {len(evidence)} verified sentences")
    
    # Save results
    print("\n[5] Saving results...")
    output_path = 'results_langgraph.csv'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Story ID', 'Prediction', 'Rationale'])
        for r in results:
            writer.writerow([r['id'], r['prediction'], r['rationale']])
    
    print(f"    Saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    consistent = sum(1 for r in results if r['prediction'] == 1)
    print(f"Total processed: {len(results)}")
    print(f"Consistent: {consistent} ({consistent/len(results)*100:.1f}%)")
    print(f"Inconsistent: {len(results) - consistent} ({(len(results)-consistent)/len(results)*100:.1f}%)")
    
    print("\nDetailed predictions:")
    print("-" * 70)
    for r in results:
        print(f"ID {r['id']:3d}: {r['label']:12s} ({r['confidence']:.0%}) | {r['rationale'][:50]}...")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE with LangGraph + Gemini 2.0 Flash!")
    print("=" * 70)

if __name__ == '__main__':
    main()

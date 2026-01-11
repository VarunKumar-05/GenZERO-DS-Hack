# Inference Pipeline for BDH Narrative Consistency Classifier
# Copyright 2026 - Kharagpur Data Science Hackathon

"""
Inference script that:
1. Loads trained model
2. Processes input backstory and narrative
3. Outputs classification, score, and supporting sentences
"""

import os
import sys
import argparse
import csv
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config, GeminiConfig
from train_classifier import NarrativeClassifier, SimpleTokenizer, USE_HF_TOKENIZER
from data_pipeline import NarrativeDataPipeline, NarrativeExample
from text_processor import TextProcessor, BackstoryAligner
from reasoning_graph import ReasoningGraph, create_reasoning_graph

if USE_HF_TOKENIZER:
    from transformers import AutoTokenizer


class InferencePipeline:
    """
    Complete inference pipeline for narrative consistency classification.
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = 'cuda',
        use_reasoning: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint.get('config', get_config())
        self.model = NarrativeClassifier(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Tokenizer
        if USE_HF_TOKENIZER:
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = SimpleTokenizer()
        
        # Text processor
        self.text_processor = TextProcessor()
        self.aligner = BackstoryAligner()
        
        # Reasoning graph (optional)
        self.use_reasoning = use_reasoning
        if use_reasoning:
            self.reasoning_graph = create_reasoning_graph(self.config.gemini)
        else:
            self.reasoning_graph = None
        
        print(f"Model loaded. Device: {self.device}")
    
    def predict(
        self,
        backstory: str,
        narrative_text: str,
        character: str = "",
        return_evidence: bool = True
    ) -> Dict:
        """
        Predict consistency of backstory with narrative.
        
        Args:
            backstory: The hypothetical backstory text
            narrative_text: The full narrative (or relevant portion)
            character: Character name (optional)
            return_evidence: Whether to extract supporting sentences
            
        Returns:
            Dict with:
                - label: "consistent" or "inconsistent"
                - score: confidence score (0-1)
                - supporting_sentences: list of evidence sentences
                - explanation: reasoning summary (if reasoning enabled)
        """
        # Process narrative
        processed = self.text_processor.process(narrative_text)
        
        # Tokenize inputs
        backstory_enc = self.tokenizer(
            backstory,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Use relevant portions of narrative
        narrative_context = " ".join(processed.sentences[:100])
        narrative_enc = self.tokenizer(
            narrative_context,
            max_length=2048,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        backstory_ids = backstory_enc['input_ids'].to(self.device)
        backstory_mask = backstory_enc['attention_mask'].to(self.device)
        narrative_ids = narrative_enc['input_ids'].to(self.device)
        narrative_mask = narrative_enc['attention_mask'].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, extras = self.model(
                backstory_ids, narrative_ids,
                backstory_mask, narrative_mask
            )
        
        # Get prediction
        probs = F.softmax(logits, dim=-1)
        predicted_class = logits.argmax(dim=-1).item()
        confidence = probs[0, predicted_class].item()
        
        label = "consistent" if predicted_class == 1 else "inconsistent"
        
        result = {
            'label': label,
            'score': confidence,
            'predicted_class': predicted_class
        }
        
        # Extract supporting sentences
        if return_evidence:
            # Use attention weights to find relevant sentences
            cross_attn = extras.get('cross_attention')
            narrative_attn = extras.get('narrative_attention')
            
            # Find relevant passages based on backstory content
            relevant_passages = self.aligner.find_relevant_passages(
                backstory, processed, top_k=10
            )
            
            # Use reasoning graph if available
            if self.use_reasoning and self.reasoning_graph is not None:
                reasoning_result = self.reasoning_graph.run(
                    backstory=backstory,
                    narrative_sentences=processed.sentences,
                    character=character,
                    candidate_evidence=relevant_passages
                )
                
                result['supporting_sentences'] = [
                    {
                        'sentence': e.sentence_text,
                        'relevance': e.relevance_score,
                        'explanation': e.explanation
                    }
                    for e in reasoning_result.evidence
                ]
                result['explanation'] = reasoning_result.summary
            else:
                # Simple evidence extraction without Gemini
                result['supporting_sentences'] = [
                    {
                        'sentence': sent,
                        'relevance': score,
                        'explanation': ''
                    }
                    for idx, sent, score in relevant_passages[:5]
                ]
                result['explanation'] = f"Classified as {label} with {confidence:.2%} confidence."
        
        return result
    
    def predict_csv(
        self,
        csv_path: str,
        books_dir: str,
        output_path: str,
        sample: Optional[int] = None
    ) -> List[Dict]:
        """
        Run predictions on a CSV file.
        
        Args:
            csv_path: Path to input CSV
            books_dir: Path to books directory
            output_path: Path to save results CSV
            sample: Optional number of samples to process
            
        Returns:
            List of prediction results
        """
        # Load data
        pipeline = NarrativeDataPipeline(self.config.pathway, 'Dataset', books_dir)
        
        # Check if test or train format
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            has_label = 'label' in reader.fieldnames
        
        if has_label:
            examples = pipeline.load_train_csv(csv_path)
        else:
            examples = pipeline.load_test_csv(csv_path)
        
        examples = pipeline.prepare_examples(examples, load_books=True)
        
        if sample:
            examples = examples[:sample]
        
        print(f"Processing {len(examples)} examples...")
        
        results = []
        for i, example in enumerate(examples):
            print(f"Processing {i+1}/{len(examples)}: {example.book_name} - {example.character}")
            
            # Get narrative text
            narrative_text = ""
            if example.processed_narrative:
                narrative_text = example.processed_narrative.original[:50000]
            elif example.book_text:
                narrative_text = example.book_text[:50000]
            
            # Predict
            prediction = self.predict(
                backstory=example.backstory,
                narrative_text=narrative_text,
                character=example.character,
                return_evidence=True
            )
            
            # Add metadata
            prediction['id'] = example.id
            prediction['book_name'] = example.book_name
            prediction['character'] = example.character
            if example.label is not None:
                prediction['true_label'] = 'consistent' if example.label == 1 else 'inconsistent'
            
            results.append(prediction)
        
        # Save results
        self._save_results(results, output_path)
        
        return results
    
    def _save_results(self, results: List[Dict], output_path: str):
        """Save results to CSV in competition format"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Story ID', 'Prediction', 'Rationale'])
            
            for r in results:
                prediction = 1 if r['label'] == 'consistent' else 0
                rationale = r.get('explanation', '')[:200]  # Limit length
                writer.writerow([r['id'], prediction, rationale])
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='BDH Narrative Classifier Inference')
    parser.add_argument('--model', default='outputs/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--input-csv', help='Input CSV file for batch prediction')
    parser.add_argument('--books-dir', default='Dataset/Books', help='Path to books directory')
    parser.add_argument('--output', default='results.csv', help='Output CSV path')
    parser.add_argument('--sample', type=int, help='Number of samples to process')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--no-reasoning', action='store_true', help='Disable Gemini reasoning')
    
    # Interactive mode arguments
    parser.add_argument('--backstory', help='Backstory text for single prediction')
    parser.add_argument('--narrative', help='Path to narrative text file')
    parser.add_argument('--character', default='', help='Character name')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        model_path=args.model,
        device=args.device,
        use_reasoning=not args.no_reasoning
    )
    
    if args.input_csv:
        # Batch mode
        results = pipeline.predict_csv(
            csv_path=args.input_csv,
            books_dir=args.books_dir,
            output_path=args.output,
            sample=args.sample
        )
        
        # Print summary
        if results and 'true_label' in results[0]:
            correct = sum(1 for r in results if r['label'] == r['true_label'])
            print(f"\nAccuracy: {correct}/{len(results)} = {correct/len(results):.2%}")
    
    elif args.backstory and args.narrative:
        # Single prediction mode
        with open(args.narrative, 'r', encoding='utf-8') as f:
            narrative_text = f.read()
        
        result = pipeline.predict(
            backstory=args.backstory,
            narrative_text=narrative_text,
            character=args.character,
            return_evidence=True
        )
        
        print("\n=== Prediction Result ===")
        print(f"Classification: {result['label']}")
        print(f"Confidence: {result['score']:.2%}")
        print(f"\nExplanation: {result['explanation']}")
        print("\nSupporting Sentences:")
        for i, sent in enumerate(result.get('supporting_sentences', []), 1):
            print(f"  {i}. [{sent['relevance']:.2f}] {sent['sentence'][:100]}...")
    
    else:
        print("Usage:")
        print("  Batch mode: python inference.py --input-csv Dataset/test.csv")
        print("  Single mode: python inference.py --backstory 'text' --narrative path/to/book.txt")


if __name__ == '__main__':
    main()

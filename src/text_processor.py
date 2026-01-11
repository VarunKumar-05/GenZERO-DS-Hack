# Text Preprocessing Utilities
# Copyright 2026 - Kharagpur Data Science Hackathon

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Container for processed text at multiple granularities"""
    original: str
    sentences: List[str]
    paragraphs: List[str]
    chapters: List[str]
    token_to_sentence_map: List[int]
    sentence_to_paragraph_map: List[int]
    paragraph_to_chapter_map: List[int]


class TextProcessor:
    """
    Processes narrative text into hierarchical chunks.
    Handles sentence, paragraph, and chapter segmentation.
    """
    
    def __init__(self):
        # Common sentence-ending patterns
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$'
        )
        # Chapter detection patterns
        self.chapter_patterns = [
            re.compile(r'^Chapter\s+\d+', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^CHAPTER\s+[IVXLCDM]+', re.MULTILINE),
            re.compile(r'^Part\s+\d+', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\d+\.\s+[A-Z]', re.MULTILINE),
        ]
        
    def process(self, text: str) -> ProcessedText:
        """
        Process text into hierarchical chunks.
        
        Args:
            text: Full narrative text
            
        Returns:
            ProcessedText with all granularities
        """
        # Clean text
        text = self._clean_text(text)
        
        # Extract chapters
        chapters = self._extract_chapters(text)
        
        # Extract paragraphs
        paragraphs = self._extract_paragraphs(text)
        
        # Extract sentences
        sentences = self._extract_sentences(text)
        
        # Build mappings
        token_to_sentence = self._build_token_sentence_map(sentences)
        sentence_to_para = self._build_sentence_paragraph_map(sentences, paragraphs)
        para_to_chapter = self._build_paragraph_chapter_map(paragraphs, chapters)
        
        return ProcessedText(
            original=text,
            sentences=sentences,
            paragraphs=paragraphs,
            chapters=chapters,
            token_to_sentence_map=token_to_sentence,
            sentence_to_paragraph_map=sentence_to_para,
            paragraph_to_chapter_map=para_to_chapter
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r' +', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()\-\n]', '', text)
        return text.strip()
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Split by sentence-ending punctuation
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) > 5:  # Filter very short fragments
                sentences.append(sent)
        
        return sentences
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        # Split by double newlines or indentation patterns
        raw_paragraphs = re.split(r'\n\s*\n', text)
        
        paragraphs = []
        for para in raw_paragraphs:
            para = para.strip()
            if len(para) > 20:  # Filter very short paragraphs
                paragraphs.append(para)
        
        return paragraphs
    
    def _extract_chapters(self, text: str) -> List[str]:
        """Extract chapters from text"""
        chapters = []
        
        # Find chapter boundaries
        boundaries = []
        for pattern in self.chapter_patterns:
            for match in pattern.finditer(text):
                boundaries.append(match.start())
        
        if not boundaries:
            # No explicit chapters, split by large gaps or treat whole text
            # Try to find natural breaks (3+ newlines)
            breaks = [m.start() for m in re.finditer(r'\n\s*\n\s*\n', text)]
            if breaks:
                boundaries = [0] + breaks
            else:
                return [text]
        
        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))
        
        for i in range(len(boundaries) - 1):
            chapter_text = text[boundaries[i]:boundaries[i+1]].strip()
            if len(chapter_text) > 100:
                chapters.append(chapter_text)
        
        return chapters if chapters else [text]
    
    def _build_token_sentence_map(self, sentences: List[str]) -> List[int]:
        """Map token positions to sentence indices (approximate)"""
        token_map = []
        for sent_idx, sent in enumerate(sentences):
            # Rough token count (words)
            tokens = sent.split()
            token_map.extend([sent_idx] * len(tokens))
        return token_map
    
    def _build_sentence_paragraph_map(
        self, 
        sentences: List[str], 
        paragraphs: List[str]
    ) -> List[int]:
        """Map sentence indices to paragraph indices"""
        mapping = []
        para_idx = 0
        current_para_text = paragraphs[0] if paragraphs else ""
        
        for sent in sentences:
            # Check if sentence is in current paragraph
            if sent in current_para_text:
                mapping.append(para_idx)
            else:
                # Move to next paragraph
                while para_idx < len(paragraphs) - 1:
                    para_idx += 1
                    current_para_text = paragraphs[para_idx]
                    if sent in current_para_text:
                        break
                mapping.append(para_idx)
        
        return mapping
    
    def _build_paragraph_chapter_map(
        self, 
        paragraphs: List[str], 
        chapters: List[str]
    ) -> List[int]:
        """Map paragraph indices to chapter indices"""
        mapping = []
        chap_idx = 0
        current_chap_text = chapters[0] if chapters else ""
        
        for para in paragraphs:
            if para in current_chap_text:
                mapping.append(chap_idx)
            else:
                while chap_idx < len(chapters) - 1:
                    chap_idx += 1
                    current_chap_text = chapters[chap_idx]
                    if para in current_chap_text:
                        break
                mapping.append(chap_idx)
        
        return mapping


class BackstoryAligner:
    """
    Aligns backstory content with narrative text to find relevant passages.
    """
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.threshold = similarity_threshold
        
    def find_relevant_passages(
        self,
        backstory: str,
        narrative: ProcessedText,
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        Find narrative passages most relevant to backstory.
        
        Args:
            backstory: The hypothetical backstory text
            narrative: Processed narrative text
            top_k: Number of passages to return
            
        Returns:
            List of (sentence_idx, sentence_text, relevance_score) tuples
        """
        # Extract key terms from backstory
        backstory_terms = self._extract_key_terms(backstory)
        
        # Score each sentence
        scored_sentences = []
        for idx, sent in enumerate(narrative.sentences):
            score = self._compute_relevance(backstory_terms, sent)
            scored_sentences.append((idx, sent, score))
        
        # Sort by score and return top-k
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        return scored_sentences[:top_k]
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms from text"""
        # Simple tokenization and filtering
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        # Filter common stopwords
        stopwords = {
            'the', 'and', 'was', 'for', 'that', 'with', 'his', 'her',
            'but', 'not', 'had', 'have', 'has', 'this', 'from', 'were',
            'been', 'would', 'could', 'should', 'their', 'them', 'they'
        }
        return set(w for w in words if w not in stopwords)
    
    def _compute_relevance(self, key_terms: set, sentence: str) -> float:
        """Compute relevance score between key terms and sentence"""
        sent_terms = self._extract_key_terms(sentence)
        if not key_terms or not sent_terms:
            return 0.0
        
        # Jaccard-like overlap
        intersection = len(key_terms & sent_terms)
        union = len(key_terms | sent_terms)
        
        return intersection / union if union > 0 else 0.0


def load_novel(filepath: str) -> str:
    """Load a novel text file"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_train_data(csv_path: str) -> List[Dict]:
    """Load training data from CSV"""
    import csv
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'id': int(row['id']),
                'book_name': row['book_name'],
                'character': row['char'],
                'caption': row.get('caption', ''),
                'content': row['content'],
                'label': 1 if row['label'] == 'consistent' else 0
            })
    
    return data

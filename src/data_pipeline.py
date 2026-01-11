# Pathway Data Pipeline for Narrative Streaming
# Copyright 2026 - Kharagpur Data Science Hackathon

"""
Data pipeline using Pathway for streaming narrative data.
Handles book loading, text chunking, and real-time processing.
"""

import os
from typing import List, Dict, Generator, Tuple, Optional
from dataclasses import dataclass
import csv

# Note: Pathway import is optional - we provide a fallback for systems without it
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False
    pw = None

from config import PathwayConfig
from text_processor import TextProcessor, ProcessedText, load_novel


@dataclass
class NarrativeExample:
    """Single training/inference example"""
    id: int
    book_name: str
    character: str
    caption: str
    backstory: str
    label: Optional[int]  # None for inference
    book_text: Optional[str] = None
    processed_narrative: Optional[ProcessedText] = None


class NarrativeDataPipeline:
    """
    Main data pipeline for narrative consistency classification.
    Supports both Pathway streaming and standard batch processing.
    """
    
    def __init__(self, config: PathwayConfig, data_dir: str, books_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.books_dir = books_dir
        self.text_processor = TextProcessor()
        
        # Book cache
        self._book_cache: Dict[str, ProcessedText] = {}
        
        # Book name to file mapping
        self._book_files = self._discover_books()
    
    def _discover_books(self) -> Dict[str, str]:
        """Discover available book files"""
        book_files = {}
        if os.path.exists(self.books_dir):
            for filename in os.listdir(self.books_dir):
                if filename.endswith('.txt'):
                    # Normalize book name for matching
                    book_name = filename.replace('.txt', '').lower()
                    book_files[book_name] = os.path.join(self.books_dir, filename)
        return book_files
    
    def _find_book_file(self, book_name: str) -> Optional[str]:
        """Find the file path for a book by name"""
        normalized = book_name.lower().replace(' ', '_')
        
        # Try exact match first
        if normalized in self._book_files:
            return self._book_files[normalized]
        
        # Try partial match
        for key, path in self._book_files.items():
            if normalized in key or key in normalized:
                return path
            # Try word matching
            if any(word in key for word in normalized.split('_')):
                return path
        
        return None
    
    def load_book(self, book_name: str) -> Optional[ProcessedText]:
        """Load and process a book by name"""
        if book_name in self._book_cache:
            return self._book_cache[book_name]
        
        book_path = self._find_book_file(book_name)
        if book_path is None:
            print(f"Warning: Could not find book file for '{book_name}'")
            return None
        
        try:
            raw_text = load_novel(book_path)
            processed = self.text_processor.process(raw_text)
            self._book_cache[book_name] = processed
            return processed
        except Exception as e:
            print(f"Error loading book {book_name}: {e}")
            return None
    
    def load_train_csv(self, csv_path: str) -> List[NarrativeExample]:
        """Load training examples from CSV"""
        examples = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = 1 if row['label'] == 'consistent' else 0
                example = NarrativeExample(
                    id=int(row['id']),
                    book_name=row['book_name'],
                    character=row['char'],
                    caption=row.get('caption', ''),
                    backstory=row['content'],
                    label=label
                )
                examples.append(example)
        
        return examples
    
    def load_test_csv(self, csv_path: str) -> List[NarrativeExample]:
        """Load test examples from CSV (no labels)"""
        examples = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                example = NarrativeExample(
                    id=int(row['id']),
                    book_name=row['book_name'],
                    character=row['char'],
                    caption=row.get('caption', ''),
                    backstory=row['content'],
                    label=None  # No label for test
                )
                examples.append(example)
        
        return examples
    
    def prepare_examples(
        self, 
        examples: List[NarrativeExample],
        load_books: bool = True
    ) -> List[NarrativeExample]:
        """Prepare examples by loading associated book data"""
        prepared = []
        
        for example in examples:
            if load_books:
                processed_narrative = self.load_book(example.book_name)
                example.processed_narrative = processed_narrative
                if processed_narrative:
                    example.book_text = processed_narrative.original
            prepared.append(example)
        
        return prepared
    
    def create_batches(
        self, 
        examples: List[NarrativeExample], 
        batch_size: int
    ) -> Generator[List[NarrativeExample], None, None]:
        """Create batches of examples"""
        for i in range(0, len(examples), batch_size):
            yield examples[i:i + batch_size]
    
    def stream_examples(
        self, 
        examples: List[NarrativeExample]
    ) -> Generator[NarrativeExample, None, None]:
        """Stream examples one at a time"""
        for example in examples:
            yield example


class PathwayStreamPipeline:
    """
    Pathway-based streaming pipeline for real-time processing.
    Only available when Pathway is installed.
    """
    
    def __init__(self, config: PathwayConfig, books_dir: str):
        if not PATHWAY_AVAILABLE:
            raise ImportError("Pathway is not installed. Install with: pip install pathway")
        
        self.config = config
        self.books_dir = books_dir
        self.text_processor = TextProcessor()
    
    def create_book_stream(self) -> 'pw.Table':
        """Create a Pathway stream from book files"""
        # Use Pathway file connector
        class BookSchema(pw.Schema):
            path: str
            content: str
        
        # Read all txt files from books directory
        books = pw.io.fs.read(
            self.books_dir,
            format='binary',
            mode='static',
            with_metadata=True
        )
        
        return books
    
    def process_stream(self, stream: 'pw.Table') -> 'pw.Table':
        """Process a stream of book content"""
        @pw.udf
        def extract_sentences(content: bytes) -> list:
            text = content.decode('utf-8', errors='ignore')
            processed = self.text_processor.process(text)
            return processed.sentences[:100]  # Limit for memory
        
        processed = stream.select(
            sentences=extract_sentences(stream.data)
        )
        
        return processed


def create_data_pipeline(
    config: PathwayConfig,
    data_dir: str = "Dataset",
    books_dir: str = "Dataset/Books"
) -> NarrativeDataPipeline:
    """Factory function to create the data pipeline"""
    return NarrativeDataPipeline(config, data_dir, books_dir)


# ============================================================================
# JSON to TOON Conversion Feature
# ============================================================================

import json
from datetime import datetime, date
from typing import Any, Union


class JSONToTOONConverter:
    """
    Converts JSON data to TOON (Token-Oriented Object Notation) format.
    
    TOON is a human-readable config format optimized for LLM memory retention,
    based on key-value pairs that are easy to parse and serialize.
    
    Features:
    - Handles nested objects (tables)
    - Handles arrays of tables
    - Proper type serialization (strings, numbers, booleans, dates)
    - Supports inline tables for simple objects
    """
    
    def __init__(self, indent: int = 0):
        self.indent = indent
        self._inline_threshold = 3  # Max keys for inline table
    
    def convert(self, data: Union[dict, str], pretty: bool = True) -> str:
        """
        Convert JSON data to TOON format.
        
        Args:
            data: JSON dict or JSON string
            pretty: If True, use multi-line formatting
            
        Returns:
            TOON formatted string
        """
        if isinstance(data, str):
            data = json.loads(data)
        
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON must be an object/dict")
        
        return self._serialize_dict(data, prefix="", pretty=pretty)
    
    def _serialize_dict(self, data: dict, prefix: str = "", pretty: bool = True) -> str:
        """Serialize a dictionary to TOON"""
        lines = []
        
        # Separate simple values from nested objects/arrays
        simple_values = {}
        nested_objects = {}
        array_of_tables = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                nested_objects[key] = value
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                array_of_tables[key] = value
            else:
                simple_values[key] = value
        
        # Write simple key-value pairs first
        for key, value in simple_values.items():
            lines.append(f"{self._escape_key(key)} = {self._serialize_value(value)}")
        
        # Add blank line before nested objects if there were simple values
        if simple_values and (nested_objects or array_of_tables):
            lines.append("")
        
        # Write nested objects as [table] sections
        for key, value in nested_objects.items():
            table_name = f"{prefix}.{key}" if prefix else key
            
            # Check if can use inline table
            if self._can_inline(value):
                lines.append(f"{self._escape_key(key)} = {self._serialize_inline_table(value)}")
            else:
                lines.append(f"[{table_name}]")
                lines.append(self._serialize_dict(value, prefix=table_name, pretty=pretty))
                lines.append("")
        
        # Write arrays of tables as [[table]] sections
        for key, values in array_of_tables.items():
            table_name = f"{prefix}.{key}" if prefix else key
            for item in values:
                lines.append(f"[[{table_name}]]")
                lines.append(self._serialize_dict(item, prefix=table_name, pretty=pretty))
                lines.append("")
        
        return "\n".join(lines).strip()
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize a single value to TOON format"""
        if value is None:
            return '""'  # TOON doesn't have null, use empty string
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            if value != value:  # NaN check
                return "nan"
            elif value == float('inf'):
                return "inf"
            elif value == float('-inf'):
                return "-inf"
            return str(value)
        elif isinstance(value, str):
            return self._serialize_string(value)
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        elif isinstance(value, list):
            return self._serialize_array(value)
        elif isinstance(value, dict):
            return self._serialize_inline_table(value)
        else:
            return self._serialize_string(str(value))
    
    def _serialize_string(self, s: str) -> str:
        """Serialize a string with proper escaping"""
        # Check if multi-line string is needed
        if '\n' in s:
            # Use multi-line literal string
            if "'''" not in s:
                return f"'''\n{s}'''"
            # Fall back to multi-line basic string
            return f'"""\n{self._escape_string(s)}"""'
        
        # Single line - escape special chars
        escaped = self._escape_string(s)
        return f'"{escaped}"'
    
    def _escape_string(self, s: str) -> str:
        """Escape special characters in a string"""
        escapes = {
            '\\': '\\\\',
            '"': '\\"',
            '\b': '\\b',
            '\t': '\\t',
            '\n': '\\n',
            '\f': '\\f',
            '\r': '\\r',
        }
        for char, escaped in escapes.items():
            s = s.replace(char, escaped)
        return s
    
    def _escape_key(self, key: str) -> str:
        """Escape a key if needed"""
        # Bare keys can contain A-Za-z0-9_-
        if key and all(c.isalnum() or c in '_-' for c in key):
            return key
        return f'"{self._escape_string(key)}"'
    
    def _serialize_array(self, arr: list) -> str:
        """Serialize an array to TOON format"""
        if not arr:
            return "[]"
        
        # Check if all elements are simple (not dicts)
        if all(not isinstance(x, dict) for x in arr):
            elements = [self._serialize_value(x) for x in arr]
            # Use multi-line for long arrays
            if len(", ".join(elements)) > 60:
                inner = ",\n  ".join(elements)
                return f"[\n  {inner}\n]"
            return f"[{', '.join(elements)}]"
        
        # Array of tables - handled at dict level
        return "[]"
    
    def _serialize_inline_table(self, obj: dict) -> str:
        """Serialize a dict as an inline table"""
        pairs = [f"{self._escape_key(k)} = {self._serialize_value(v)}" 
                 for k, v in obj.items()]
        return "{ " + ", ".join(pairs) + " }"
    
    def _can_inline(self, obj: dict) -> bool:
        """Check if a dict can be serialized as inline table"""
        if len(obj) > self._inline_threshold:
            return False
        # No nested objects or arrays
        return all(not isinstance(v, (dict, list)) for v in obj.values())


def json_to_toon(data: Union[dict, str, Any], output_path: Optional[str] = None) -> str:
    """
    Convert JSON data to TOON (Token-Oriented Object Notation) format.
    
    Args:
        data: JSON dict, JSON string, or path to JSON file
        output_path: Optional path to save TOON output
        
    Returns:
        TOON formatted string
        
    Example:
        >>> json_data = {"name": "BDH", "version": 1.0, "features": ["multi-scale", "hebbian"]}
        >>> print(json_to_toml(json_data))
        name = "BDH"
        version = 1.0
        features = ["multi-scale", "hebbian"]
    """
    converter = JSONToTOONConverter()
    
    # Handle file path input
    if isinstance(data, str) and os.path.isfile(data):
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    toon_str = converter.convert(data)
    
    # Optionally save to file
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(toon_str)
        print(f"TOON saved to: {output_path}")
    
    return toon_str


def example_to_toon(example: NarrativeExample) -> str:
    """Convert a NarrativeExample to TOON format"""
    data = {
        'id': example.id,
        'book_name': example.book_name,
        'character': example.character,
        'caption': example.caption,
        'backstory': example.backstory,
        'label': example.label
    }
    return json_to_toon(data)


def results_to_toon(results: List[Dict], output_path: Optional[str] = None) -> str:
    """
    Convert prediction results to TOON (Token-Oriented Object Notation) format.
    
    Args:
        results: List of prediction result dicts
        output_path: Optional path to save TOON output
        
    Returns:
        TOON formatted string
    """
    data = {
        'metadata': {
            'total_predictions': len(results),
            'generated_at': datetime.now().isoformat(),
            'model': 'BDH Narrative Consistency Classifier'
        },
        'predictions': results
    }
    return json_to_toon(data, output_path)

"""
Text Chunking Module for Multilingual RAG System

This module implements various text chunking strategies optimized for
Bengali and English text retrieval.
"""

import re
import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import math

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Bengali NLP
try:
    from bnlp import BasicTokenizer, SentenceTokenizer
    BNLP_AVAILABLE = True
except ImportError:
    BNLP_AVAILABLE = False
    logging.warning("BNLP not available. Bengali-specific chunking may be limited.")

from config import config

@dataclass
class TextChunk:
    """Data class for text chunks"""
    text: str
    start_index: int
    end_index: int
    chunk_id: str
    language: str
    word_count: int
    char_count: int
    sentence_count: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TextChunker:
    """
    Advanced text chunker with multilingual support for Bengali and English
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "sentence_aware"
    ):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            strategy: Chunking strategy ('fixed', 'sentence_aware', 'paragraph', 'semantic')
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Setup language-specific tools
        self.setup_language_tools()
    
    def setup_language_tools(self):
        """Setup language-specific tokenizers"""
        if BNLP_AVAILABLE:
            self.bn_tokenizer = BasicTokenizer()
            self.bn_sent_tokenizer = SentenceTokenizer()
        else:
            self.bn_tokenizer = None
            self.bn_sent_tokenizer = None
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text
        
        Args:
            text: Input text
            
        Returns:
            Language code ('bn' for Bengali, 'en' for English)
        """
        if not text.strip():
            return "unknown"
        
        # Check for Bengali characters
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return "unknown"
        
        bengali_ratio = bengali_chars / total_chars
        return "bn" if bengali_ratio > 0.3 else "en"
    
    def split_into_sentences(self, text: str, language: str) -> List[str]:
        """
        Split text into sentences based on language
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of sentences
        """
        if language == "bn" and self.bn_sent_tokenizer:
            try:
                sentences = self.bn_sent_tokenizer.sentence_tokenize(text)
                # Filter out very short sentences
                sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
                return sentences
            except Exception as e:
                self.logger.warning(f"Bengali sentence tokenization failed: {e}")
        
        # Fallback to NLTK or manual splitting
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if len(s.strip()) > 5]
        except Exception:
            # Manual sentence splitting for Bengali
            if language == "bn":
                # Split on Bengali sentence endings
                sentences = re.split(r'[।!?]', text)
            else:
                # Split on English sentence endings
                sentences = re.split(r'[.!?]', text)
            
            return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def chunk_fixed_size(self, text: str, language: str) -> List[TextChunk]:
        """
        Create fixed-size chunks with overlap
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at word boundary
            if end < text_length:
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Calculate statistics
                word_count = len(chunk_text.split())
                sentence_count = len(self.split_into_sentences(chunk_text, language))
                
                chunk = TextChunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    chunk_id=f"chunk_{chunk_id:04d}",
                    language=language,
                    word_count=word_count,
                    char_count=len(chunk_text),
                    sentence_count=sentence_count,
                    metadata={"strategy": "fixed_size"}
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = max(end - self.chunk_overlap, start + 1)
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def chunk_sentence_aware(self, text: str, language: str) -> List[TextChunk]:
        """
        Create chunks that respect sentence boundaries
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of TextChunk objects
        """
        sentences = self.split_into_sentences(text, language)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_length = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if (current_chunk_length + sentence_length > self.chunk_size 
                and current_chunk_sentences):
                
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                
                # Find start and end indices in original text
                start_index = text.find(current_chunk_sentences[0])
                end_index = start_index + len(chunk_text)
                
                chunk = TextChunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    chunk_id=f"chunk_{chunk_id:04d}",
                    language=language,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    sentence_count=len(current_chunk_sentences),
                    metadata={
                        "strategy": "sentence_aware",
                        "sentences": current_chunk_sentences.copy()
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Handle overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.chunk_overlap
                )
                current_chunk_sentences = overlap_sentences
                current_chunk_length = sum(len(s) for s in overlap_sentences)
            
            # Add current sentence
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_length
        
        # Handle remaining sentences
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            start_index = text.find(current_chunk_sentences[0])
            end_index = start_index + len(chunk_text)
            
            chunk = TextChunk(
                text=chunk_text,
                start_index=start_index,
                end_index=end_index,
                chunk_id=f"chunk_{chunk_id:04d}",
                language=language,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                sentence_count=len(current_chunk_sentences),
                metadata={
                    "strategy": "sentence_aware",
                    "sentences": current_chunk_sentences.copy()
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_paragraph_based(self, text: str, language: str) -> List[TextChunk]:
        """
        Create chunks based on paragraph boundaries
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of TextChunk objects
        """
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback to sentence-aware chunking
            return self.chunk_sentence_aware(text, language)
        
        chunks = []
        current_chunk_paras = []
        current_chunk_length = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            # If single paragraph is too large, split it further
            if para_length > self.chunk_size:
                # Process accumulated paragraphs first
                if current_chunk_paras:
                    chunk_text = '\n\n'.join(current_chunk_paras)
                    chunks.append(self._create_chunk(
                        chunk_text, text, chunk_id, language, "paragraph_based"
                    ))
                    chunk_id += 1
                    current_chunk_paras = []
                    current_chunk_length = 0
                
                # Split large paragraph using sentence-aware method
                para_chunks = self.chunk_sentence_aware(paragraph, language)
                for para_chunk in para_chunks:
                    para_chunk.chunk_id = f"chunk_{chunk_id:04d}"
                    para_chunk.metadata["strategy"] = "paragraph_based_sentence_split"
                    chunks.append(para_chunk)
                    chunk_id += 1
                
            elif current_chunk_length + para_length > self.chunk_size and current_chunk_paras:
                # Create chunk from current paragraphs
                chunk_text = '\n\n'.join(current_chunk_paras)
                chunks.append(self._create_chunk(
                    chunk_text, text, chunk_id, language, "paragraph_based"
                ))
                chunk_id += 1
                
                # Start new chunk with current paragraph
                current_chunk_paras = [paragraph]
                current_chunk_length = para_length
            else:
                # Add paragraph to current chunk
                current_chunk_paras.append(paragraph)
                current_chunk_length += para_length
        
        # Handle remaining paragraphs
        if current_chunk_paras:
            chunk_text = '\n\n'.join(current_chunk_paras)
            chunks.append(self._create_chunk(
                chunk_text, text, chunk_id, language, "paragraph_based"
            ))
        
        return chunks
    
    def chunk_semantic(self, text: str, language: str) -> List[TextChunk]:
        """
        Create semantically coherent chunks (simplified version)
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of TextChunk objects
        """
        # For now, use sentence-aware chunking with larger context
        # This could be enhanced with semantic similarity in the future
        
        original_chunk_size = self.chunk_size
        self.chunk_size = int(self.chunk_size * 1.2)  # Allow slightly larger chunks
        
        chunks = self.chunk_sentence_aware(text, language)
        
        # Restore original chunk size
        self.chunk_size = original_chunk_size
        
        # Update metadata
        for chunk in chunks:
            chunk.metadata["strategy"] = "semantic"
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_chars: int) -> List[str]:
        """
        Get sentences for overlap based on character count
        
        Args:
            sentences: List of sentences
            overlap_chars: Target overlap in characters
            
        Returns:
            List of overlap sentences
        """
        if not sentences or overlap_chars <= 0:
            return []
        
        overlap_sentences = []
        total_chars = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            if total_chars + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                total_chars += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(
        self, 
        chunk_text: str, 
        original_text: str, 
        chunk_id: int, 
        language: str, 
        strategy: str
    ) -> TextChunk:
        """
        Helper method to create a TextChunk object
        
        Args:
            chunk_text: The chunk text
            original_text: Original full text
            chunk_id: Chunk identifier
            language: Language code
            strategy: Chunking strategy used
            
        Returns:
            TextChunk object
        """
        start_index = original_text.find(chunk_text)
        end_index = start_index + len(chunk_text)
        
        # Calculate statistics
        sentences = self.split_into_sentences(chunk_text, language)
        
        return TextChunk(
            text=chunk_text,
            start_index=start_index,
            end_index=end_index,
            chunk_id=f"chunk_{chunk_id:04d}",
            language=language,
            word_count=len(chunk_text.split()),
            char_count=len(chunk_text),
            sentence_count=len(sentences),
            metadata={"strategy": strategy}
        )
    
    def chunk_text(self, text: str, language: str = None) -> List[TextChunk]:
        """
        Main chunking method that applies the specified strategy
        
        Args:
            text: Input text to chunk
            language: Language code (auto-detected if not provided)
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        self.logger.info(f"Chunking text with strategy: {self.strategy}")
        self.logger.info(f"Text length: {len(text)} characters")
        self.logger.info(f"Detected language: {language}")
        
        # Apply chunking strategy
        if self.strategy == "fixed":
            chunks = self.chunk_fixed_size(text, language)
        elif self.strategy == "sentence_aware":
            chunks = self.chunk_sentence_aware(text, language)
        elif self.strategy == "paragraph":
            chunks = self.chunk_paragraph_based(text, language)
        elif self.strategy == "semantic":
            chunks = self.chunk_semantic(text, language)
        else:
            self.logger.warning(f"Unknown strategy: {self.strategy}. Using sentence_aware.")
            chunks = self.chunk_sentence_aware(text, language)
        
        self.logger.info(f"Created {len(chunks)} chunks")
        
        # Log chunk statistics
        if chunks:
            avg_chunk_size = sum(chunk.char_count for chunk in chunks) / len(chunks)
            min_chunk_size = min(chunk.char_count for chunk in chunks)
            max_chunk_size = max(chunk.char_count for chunk in chunks)
            
            self.logger.info(f"Chunk size stats - Avg: {avg_chunk_size:.1f}, "
                           f"Min: {min_chunk_size}, Max: {max_chunk_size}")
        
        return chunks
    
    def optimize_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Post-process chunks to optimize them for retrieval
        
        Args:
            chunks: List of chunks to optimize
            
        Returns:
            Optimized list of chunks
        """
        if not chunks:
            return chunks
        
        optimized_chunks = []
        
        for chunk in chunks:
            # Remove very short chunks (likely to be noise)
            if chunk.char_count < 50:
                self.logger.debug(f"Removing short chunk: {chunk.chunk_id}")
                continue
            
            # Clean up whitespace
            cleaned_text = re.sub(r'\s+', ' ', chunk.text.strip())
            if cleaned_text != chunk.text:
                chunk.text = cleaned_text
                chunk.char_count = len(cleaned_text)
                chunk.word_count = len(cleaned_text.split())
            
            # Add context information
            chunk.metadata["optimization"] = "cleaned"
            
            optimized_chunks.append(chunk)
        
        return optimized_chunks

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample Bengali text
    bengali_text = """
    অনুপম তার জীবনে অনেক মানুষের সাথে পরিচিত হয়েছে। তার মধ্যে শম্ভুনাথ একজন বিশেষ ব্যক্তি।
    শম্ভুনাথকে অনুপম সুপুরুষ বলে মনে করে। তিনি অনুপমের খুব প্রিয় একজন মানুষ।
    
    অনুপমের মামা তার জীবনে একটি গুরুত্বপূর্ণ ভূমিকা পালন করেছেন। অনুপম তার মামাকে তার ভাগ্য দেবতা বলে মনে করে।
    মামার পরামর্শ এবং সাহায্য অনুপমের জীবনে অনেক পরিবর্তন এনেছে।
    
    কল্যাণীর বিয়ের সময় তার বয়স ছিল মাত্র ১৫ বছর। এই অল্প বয়সেই তাকে বিয়ে দিতে হয়েছিল।
    সেই যুগে এটি একটি সাধারণ ঘটনা ছিল।
    """
    
    # Test different chunking strategies
    strategies = ["fixed", "sentence_aware", "paragraph", "semantic"]
    
    for strategy in strategies:
        print(f"\n=== Testing {strategy} strategy ===")
        chunker = TextChunker(chunk_size=200, chunk_overlap=30, strategy=strategy)
        chunks = chunker.chunk_text(bengali_text)
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} ({chunk.chunk_id}):")
            print(f"  Language: {chunk.language}")
            print(f"  Length: {chunk.char_count} chars, {chunk.word_count} words")
            print(f"  Text: {chunk.text[:100]}...")
            print() 
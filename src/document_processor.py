"""
Document Processing Module for Multilingual RAG System

This module handles PDF text extraction, cleaning, and preprocessing
with special support for Bengali text.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import unicodedata

# PDF processing libraries
import fitz  # PyMuPDF - better for complex PDFs
import PyPDF2
from PIL import Image
import pytesseract

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Bengali NLP
try:
    from bnlp import BasicTokenizer, SentenceTokenizer
    BNLP_AVAILABLE = True
except ImportError:
    BNLP_AVAILABLE = False
    logging.warning("BNLP not available. Bengali-specific processing may be limited.")

# Language detection
try:
    import pycld2 as cld2
    CLD2_AVAILABLE = True
except ImportError:
    CLD2_AVAILABLE = False
    logging.warning("pycld2 not available. Language detection will be limited.")

from config import config

class DocumentProcessor:
    """
    Document processor with multilingual support for Bengali and English
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_nltk()
        self.setup_bengali_processor()
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def setup_bengali_processor(self):
        """Setup Bengali text processing tools"""
        if BNLP_AVAILABLE:
            self.bn_tokenizer = BasicTokenizer()
            self.bn_sent_tokenizer = SentenceTokenizer()
        else:
            self.bn_tokenizer = None
            self.bn_sent_tokenizer = None
    
    def extract_text_pymupdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text using PyMuPDF (fitz) - best for complex layouts and Bengali text
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            metadata = {
                "method": "PyMuPDF",
                "pages": len(doc),
                "chars_extracted": 0,
                "has_images": False
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with formatting
                text = page.get_text("text")
                
                # Check for images (might contain text)
                if page.get_images():
                    metadata["has_images"] = True
                
                if text.strip():
                    text_content.append(text)
            
            doc.close()
            
            full_text = "\n\n".join(text_content)
            metadata["chars_extracted"] = len(full_text)
            
            return full_text, metadata
            
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}")
            return "", {"method": "PyMuPDF", "error": str(e)}
    
    def extract_text_pypdf2(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text using PyPDF2 - fallback method
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            text_content = []
            metadata = {
                "method": "PyPDF2",
                "pages": 0,
                "chars_extracted": 0
            }
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
            
            full_text = "\n\n".join(text_content)
            metadata["chars_extracted"] = len(full_text)
            
            return full_text, metadata
            
        except Exception as e:
            self.logger.error(f"PyPDF2 extraction failed: {e}")
            return "", {"method": "PyPDF2", "error": str(e)}
    
    def extract_text_ocr(self, pdf_path: str, lang: str = "ben+eng") -> Tuple[str, Dict]:
        """
        Extract text using OCR (Tesseract) - for scanned documents
        
        Args:
            pdf_path: Path to PDF file
            lang: Language code for OCR (ben for Bengali, eng for English)
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            metadata = {
                "method": "OCR",
                "pages": len(doc),
                "chars_extracted": 0,
                "ocr_language": lang
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Create PIL Image
                import io
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                text = pytesseract.image_to_string(img, lang=lang)
                
                if text.strip():
                    text_content.append(text)
            
            doc.close()
            
            full_text = "\n\n".join(text_content)
            metadata["chars_extracted"] = len(full_text)
            
            return full_text, metadata
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return "", {"method": "OCR", "error": str(e)}
    
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
        
        if bengali_ratio > 0.3:
            return "bn"
        elif CLD2_AVAILABLE:
            try:
                is_reliable, text_bytes_found, details = cld2.detect(text)
                if is_reliable and details:
                    detected_lang = details[0][1]
                    if detected_lang == "BENGALI":
                        return "bn"
                    elif detected_lang == "ENGLISH":
                        return "en"
            except:
                pass
        
        return "en"  # Default to English
    
    def clean_text(self, text: str, language: str = None) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            language: Language code for language-specific cleaning
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # Language-specific cleaning
        if language == "bn":
            # Bengali-specific cleaning
            # Remove common OCR artifacts in Bengali
            text = re.sub(r'[।]+', '।', text)  # Multiple danda to single
            text = re.sub(r'[০-৯]+\s*[।]', '', text)  # Remove page numbers with danda
            
        elif language == "en":
            # English-specific cleaning
            # Remove common English artifacts
            text = re.sub(r'\b\d+\s*\.\s*\d+\b', '', text)  # Remove decimal numbers (likely page refs)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_text(self, pdf_path: Union[str, Path], method: str = "auto") -> Dict:
        """
        Main text extraction method with multiple fallbacks
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pymupdf', 'pypdf2', 'ocr', 'auto')
            
        Returns:
            Dictionary with extracted text and metadata
        """
        pdf_path = str(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        results = {
            "file_path": pdf_path,
            "file_size": os.path.getsize(pdf_path),
            "text": "",
            "metadata": {},
            "language": "unknown",
            "method_used": "",
            "success": False
        }
        
        # Try different extraction methods
        if method == "auto":
            methods = ["pymupdf", "pypdf2"]
        else:
            methods = [method]
        
        for extraction_method in methods:
            self.logger.info(f"Trying extraction method: {extraction_method}")
            
            if extraction_method == "pymupdf":
                text, metadata = self.extract_text_pymupdf(pdf_path)
            elif extraction_method == "pypdf2":
                text, metadata = self.extract_text_pypdf2(pdf_path)
            elif extraction_method == "ocr":
                text, metadata = self.extract_text_ocr(pdf_path)
            else:
                continue
            
            if text.strip() and len(text) > 100:  # Minimum text threshold
                # Clean the text
                cleaned_text = self.clean_text(text)
                
                if cleaned_text:
                    results["text"] = cleaned_text
                    results["metadata"] = metadata
                    results["language"] = self.detect_language(cleaned_text)
                    results["method_used"] = extraction_method
                    results["success"] = True
                    
                    self.logger.info(f"Successfully extracted text using {extraction_method}")
                    self.logger.info(f"Text length: {len(cleaned_text)} characters")
                    self.logger.info(f"Detected language: {results['language']}")
                    break
        
        # If no method worked, try OCR as last resort
        if not results["success"] and "ocr" not in methods:
            self.logger.info("Trying OCR as last resort")
            text, metadata = self.extract_text_ocr(pdf_path)
            if text.strip():
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    results["text"] = cleaned_text
                    results["metadata"] = metadata
                    results["language"] = self.detect_language(cleaned_text)
                    results["method_used"] = "ocr"
                    results["success"] = True
        
        if not results["success"]:
            self.logger.error(f"Failed to extract text from {pdf_path}")
        
        return results
    
    def process_document(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Process a document completely - extract, clean, and prepare for chunking
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with processed document information
        """
        extraction_result = self.extract_text(pdf_path)
        
        if not extraction_result["success"]:
            return extraction_result
        
        text = extraction_result["text"]
        language = extraction_result["language"]
        
        # Additional processing
        processed_result = extraction_result.copy()
        
        # Sentence tokenization
        if language == "bn" and self.bn_sent_tokenizer:
            sentences = self.bn_sent_tokenizer.sentence_tokenize(text)
        else:
            sentences = sent_tokenize(text)
        
        processed_result["sentences"] = sentences
        processed_result["sentence_count"] = len(sentences)
        processed_result["word_count"] = len(text.split())
        processed_result["char_count"] = len(text)
        
        return processed_result

# Example usage and testing
if __name__ == "__main__":
    import io
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    processor = DocumentProcessor()
    
    # Test with sample PDF
    pdf_path = "sample.pdf"
    if os.path.exists(pdf_path):
        result = processor.process_document(pdf_path)
        
        if result["success"]:
            print(f"Successfully processed {pdf_path}")
            print(f"Method: {result['method_used']}")
            print(f"Language: {result['language']}")
            print(f"Characters: {result['char_count']}")
            print(f"Sentences: {result['sentence_count']}")
            print(f"Text preview: {result['text'][:200]}...")
        else:
            print(f"Failed to process {pdf_path}")
            print(f"Error: {result.get('metadata', {}).get('error', 'Unknown error')}")
    else:
        print(f"PDF file {pdf_path} not found") 
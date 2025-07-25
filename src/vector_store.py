"""
Vector Store Module for Multilingual RAG System

This module handles vector database operations for storing and retrieving
document embeddings with support for Bengali and English text.
"""

import os
import json
import logging
import uuid
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np

# Vector DB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Embeddings
from sentence_transformers import SentenceTransformer
import torch

# Local imports
from config import config
from src.text_chunker import TextChunk

class MultilingualEmbedding:
    """
    Multilingual embedding model wrapper with caching and optimization
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.logger.info(f"Loading embedding model: {self.model_name}")
        self.logger.info(f"Using device: {self.device}")
        
        # Load the model
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Encode with progress bar for large batches
            if len(texts) > 10:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array of embedding
        """
        embedding = self.encode([text])
        return embedding[0]

class VectorStore:
    """
    Vector database wrapper with multilingual support
    """
    
    def __init__(
        self,
        collection_name: str = "multilingual_docs",
        persist_directory: str = None,
        embedding_model: str = None
    ):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_model: Embedding model name
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        self.logger = logging.getLogger(__name__)
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = MultilingualEmbedding(embedding_model)
        
        # Initialize ChromaDB
        self.setup_chromadb()
    
    def setup_chromadb(self):
        """Setup ChromaDB client and collection"""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            self.logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            self.logger.info(f"Collection count: {self.collection.count()}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup ChromaDB: {e}")
            raise
    
    def add_chunks(self, chunks: List[TextChunk], batch_size: int = 100) -> bool:
        """
        Add text chunks to the vector store
        
        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for processing
            
        Returns:
            Success status
        """
        if not chunks:
            self.logger.warning("No chunks to add")
            return False
        
        try:
            self.logger.info(f"Adding {len(chunks)} chunks to vector store")
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                self._add_chunk_batch(batch_chunks)
                
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            self.logger.info(f"Successfully added {len(chunks)} chunks")
            self.logger.info(f"Total collection count: {self.collection.count()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add chunks: {e}")
            return False
    
    def _add_chunk_batch(self, chunks: List[TextChunk]):
        """
        Add a batch of chunks to the collection
        
        Args:
            chunks: List of TextChunk objects
        """
        # Prepare data for batch insertion
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Prepare metadata
        metadatas = []
        for chunk in chunks:
            metadata = {
                "language": chunk.language,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "sentence_count": chunk.sentence_count,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                **chunk.metadata
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
    
    def search(
        self,
        query: str,
        n_results: int = None,
        language_filter: str = None,
        similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            language_filter: Filter by language ('bn' or 'en')
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results with metadata
        """
        n_results = n_results or config.TOP_K_RETRIEVAL
        similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode_single(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if language_filter:
                where_clause["language"] = language_filter
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results * 2, self.collection.count()),  # Get more to filter
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = 1 - distance
                
                # Apply similarity threshold
                if similarity >= similarity_threshold:
                    result = {
                        "text": doc,
                        "metadata": metadata,
                        "similarity": similarity,
                        "distance": distance,
                        "rank": i + 1
                    }
                    search_results.append(result)
            
            # Limit to requested number of results
            search_results = search_results[:n_results]
            
            self.logger.info(f"Search query: {query[:50]}...")
            self.logger.info(f"Found {len(search_results)} results above threshold {similarity_threshold}")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve a specific chunk by ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if results["documents"]:
                return {
                    "text": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            total_count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_size = min(100, total_count)
            if sample_size > 0:
                sample_results = self.collection.get(
                    limit=sample_size,
                    include=["metadatas"]
                )
                
                # Analyze languages
                languages = {}
                strategies = {}
                total_chars = 0
                total_words = 0
                
                for metadata in sample_results["metadatas"]:
                    lang = metadata.get("language", "unknown")
                    languages[lang] = languages.get(lang, 0) + 1
                    
                    strategy = metadata.get("strategy", "unknown")
                    strategies[strategy] = strategies.get(strategy, 0) + 1
                    
                    total_chars += metadata.get("char_count", 0)
                    total_words += metadata.get("word_count", 0)
                
                avg_chars = total_chars / sample_size if sample_size > 0 else 0
                avg_words = total_words / sample_size if sample_size > 0 else 0
                
                return {
                    "total_chunks": total_count,
                    "languages": languages,
                    "chunking_strategies": strategies,
                    "avg_chars_per_chunk": avg_chars,
                    "avg_words_per_chunk": avg_words,
                    "embedding_model": self.embedding_model.model_name,
                    "embedding_dimension": self.embedding_model.embedding_dim
                }
            else:
                return {
                    "total_chunks": 0,
                    "languages": {},
                    "chunking_strategies": {},
                    "avg_chars_per_chunk": 0,
                    "avg_words_per_chunk": 0,
                    "embedding_model": self.embedding_model.model_name,
                    "embedding_dimension": self.embedding_model.embedding_dim
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)"""
        try:
            self.delete_collection()
            self.setup_chromadb()
            self.logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")

class VectorStoreManager:
    """
    High-level manager for vector store operations
    """
    
    def __init__(self, collection_name: str = "multilingual_docs"):
        """
        Initialize the vector store manager
        
        Args:
            collection_name: Name of the collection
        """
        self.vector_store = VectorStore(collection_name)
        self.logger = logging.getLogger(__name__)
    
    def index_document(self, text: str, doc_id: str = None) -> bool:
        """
        Index a complete document
        
        Args:
            text: Document text
            doc_id: Document identifier
            
        Returns:
            Success status
        """
        from src.text_chunker import TextChunker
        
        try:
            # Create chunker
            chunker = TextChunker(strategy="sentence_aware")
            
            # Chunk the text
            chunks = chunker.chunk_text(text)
            
            if not chunks:
                self.logger.warning("No chunks created from text")
                return False
            
            # Add document ID to chunks if provided
            if doc_id:
                for chunk in chunks:
                    chunk.metadata["document_id"] = doc_id
            
            # Add chunks to vector store
            success = self.vector_store.add_chunks(chunks)
            
            if success:
                self.logger.info(f"Successfully indexed document with {len(chunks)} chunks")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to index document: {e}")
            return False
    
    def search_documents(
        self,
        query: str,
        n_results: int = 5,
        language: str = None,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            n_results: Number of results
            language: Language filter
            threshold: Similarity threshold
            
        Returns:
            List of relevant chunks
        """
        return self.vector_store.search(
            query=query,
            n_results=n_results,
            language_filter=language,
            similarity_threshold=threshold
        )
    
    def get_collection_info(self) -> Dict:
        """Get collection information"""
        return self.vector_store.get_stats()

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample Bengali text
    sample_text = """
    অনুপম তার জীবনে অনেক মানুষের সাথে পরিচিত হয়েছে। তার মধ্যে শম্ভুনাথ একজন বিশেষ ব্যক্তি।
    শম্ভুনাথকে অনুপম সুপুরুষ বলে মনে করে। তিনি অনুপমের খুব প্রিয় একজন মানুষ।
    
    অনুপমের মামা তার জীবনে একটি গুরুত্বপূর্ণ ভূমিকা পালন করেছেন। অনুপম তার মামাকে তার ভাগ্য দেবতা বলে মনে করে।
    মামার পরামর্শ এবং সাহায্য অনুপমের জীবনে অনেক পরিবর্তন এনেছে।
    
    কল্যাণীর বিয়ের সময় তার বয়স ছিল মাত্র ১৫ বছর। এই অল্প বয়সেই তাকে বিয়ে দিতে হয়েছিল।
    সেই যুগে এটি একটি সাধারণ ঘটনা ছিল।
    """
    
    # Create vector store manager
    manager = VectorStoreManager("test_collection")
    
    # Index the document
    success = manager.index_document(sample_text, "sample_doc")
    
    if success:
        print("Document indexed successfully!")
        
        # Get collection info
        info = manager.get_collection_info()
        print(f"Collection info: {info}")
        
        # Test search queries
        test_queries = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "কল্যাণীর বিয়ের সময় তার বয়স কত ছিল?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = manager.search_documents(query, n_results=3)
            
            for i, result in enumerate(results):
                print(f"Result {i+1} (similarity: {result['similarity']:.3f}):")
                print(f"  {result['text'][:100]}...")
                print()
    else:
        print("Failed to index document") 
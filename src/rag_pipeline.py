"""
RAG Pipeline Module for Multilingual RAG System

This module implements the core Retrieval-Augmented Generation pipeline
with support for Bengali and English queries, including memory management.
"""

import logging
import json
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import re

# LLM
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.callbacks import get_openai_callback

# Local imports
from config import config
from src.vector_store import VectorStoreManager
from src.document_processor import DocumentProcessor

@dataclass
class ChatTurn:
    """Data class for chat conversation turns"""
    timestamp: str
    user_query: str
    assistant_response: str
    retrieved_chunks: List[Dict]
    language: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MemoryManager:
    """
    Memory management for RAG system with short-term and long-term memory
    """
    
    def __init__(self, max_history: int = None):
        """
        Initialize memory manager
        
        Args:
            max_history: Maximum number of chat turns to keep in short-term memory
        """
        self.max_history = max_history or config.MAX_CHAT_HISTORY
        self.chat_history: List[ChatTurn] = []
        self.logger = logging.getLogger(__name__)
    
    def add_turn(self, chat_turn: ChatTurn):
        """
        Add a chat turn to memory
        
        Args:
            chat_turn: ChatTurn object
        """
        self.chat_history.append(chat_turn)
        
        # Maintain max history limit
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
        
        self.logger.debug(f"Added chat turn. History length: {len(self.chat_history)}")
    
    def get_recent_context(self, num_turns: int = None) -> str:
        """
        Get recent conversation context
        
        Args:
            num_turns: Number of recent turns to include
            
        Returns:
            Formatted conversation context
        """
        if not self.chat_history:
            return ""
        
        num_turns = num_turns or min(3, len(self.chat_history))
        recent_turns = self.chat_history[-num_turns:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_query}")
            context_parts.append(f"Assistant: {turn.assistant_response}")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        self.logger.info("Chat history cleared")
    
    def get_history_summary(self) -> Dict:
        """Get summary of chat history"""
        if not self.chat_history:
            return {"total_turns": 0, "languages": {}, "topics": []}
        
        languages = {}
        for turn in self.chat_history:
            lang = turn.language
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            "total_turns": len(self.chat_history),
            "languages": languages,
            "first_turn": self.chat_history[0].timestamp,
            "last_turn": self.chat_history[-1].timestamp
        }

class MultilingualRAGPipeline:
    """
    Core RAG pipeline with multilingual support and memory management
    """
    
    def __init__(
        self,
        collection_name: str = "multilingual_docs",
        llm_model: str = None,
        temperature: float = None
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            collection_name: Vector store collection name
            llm_model: LLM model name
            temperature: Temperature for generation
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vector_store = VectorStoreManager(collection_name)
        self.memory = MemoryManager()
        self.document_processor = DocumentProcessor()
        
        # LLM configuration
        self.llm_model = llm_model or config.LLM_MODEL
        self.temperature = temperature or config.TEMPERATURE
        
        # Initialize LLM
        self.setup_llm()
        
        # Setup prompts
        self.setup_prompts()
    
    def setup_llm(self):
        """Setup the language model"""
        try:
            if self.llm_model.startswith("gpt"):
                self.llm = ChatOpenAI(
                    model_name=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=config.MAX_TOKENS,
                    openai_api_key=config.OPENAI_API_KEY
                )
            else:
                # Fallback to standard OpenAI
                self.llm = OpenAI(
                    model_name=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=config.MAX_TOKENS,
                    openai_api_key=config.OPENAI_API_KEY
                )
            
            self.logger.info(f"LLM initialized: {self.llm_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def setup_prompts(self):
        """Setup prompt templates for different languages"""
        
        # Bengali system prompt
        self.system_prompt_bn = """আপনি একটি বাংলা ভাষার সহায়ক AI অ্যাসিস্ট্যান্ট। আপনাকে প্রদান করা তথ্যের ভিত্তিতে প্রশ্নের উত্তর দিতে হবে।

নির্দেশনা:
1. শুধুমাত্র প্রদান করা তথ্যের ভিত্তিতে উত্তর দিন
2. যদি তথ্যে উত্তর না থাকে, তাহলে "প্রদান করা তথ্যে এই প্রশ্নের উত্তর নেই" বলুন
3. উত্তর সংক্ষিপ্ত এবং সঠিক হতে হবে
4. প্রয়োজনে প্রাসঙ্গিক তথ্য উল্লেখ করুন

প্রদান করা তথ্য:
{context}

পূর্ববর্তী কথোপকথন:
{chat_history}

প্রশ্ন: {question}
উত্তর:"""

        # English system prompt
        self.system_prompt_en = """You are a helpful AI assistant. Answer the question based on the provided information.

Instructions:
1. Answer only based on the provided information
2. If the answer is not in the information, say "The answer is not available in the provided information"
3. Keep answers concise and accurate
4. Include relevant context when necessary

Provided information:
{context}

Previous conversation:
{chat_history}

Question: {question}
Answer:"""
        
        # Create prompt templates
        self.prompt_template_bn = PromptTemplate(
            template=self.system_prompt_bn,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.prompt_template_en = PromptTemplate(
            template=self.system_prompt_en,
            input_variables=["context", "chat_history", "question"]
        )
    
    def detect_language(self, text: str) -> str:
        """
        Detect query language
        
        Args:
            text: Input text
            
        Returns:
            Language code ('bn' or 'en')
        """
        # Simple language detection based on character script
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return "en"  # Default to English
        
        bengali_ratio = bengali_chars / total_chars
        return "bn" if bengali_ratio > 0.3 else "en"
    
    def retrieve_relevant_chunks(
        self,
        query: str,
        language: str = None,
        num_chunks: int = None,
        similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks for the query
        
        Args:
            query: User query
            language: Query language
            num_chunks: Number of chunks to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of relevant chunks
        """
        num_chunks = num_chunks or config.TOP_K_RETRIEVAL
        similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        # Retrieve chunks
        chunks = self.vector_store.search_documents(
            query=query,
            n_results=num_chunks,
            language=language,
            threshold=similarity_threshold
        )
        
        self.logger.info(f"Retrieved {len(chunks)} relevant chunks for query")
        
        return chunks
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "কোন প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Include similarity score in context for transparency
            similarity = chunk.get('similarity', 0)
            text = chunk.get('text', '')
            
            context_parts.append(f"তথ্য {i} (প্রাসঙ্গিকতা: {similarity:.3f}):\n{text}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        context: str,
        language: str,
        chat_history: str = ""
    ) -> Tuple[str, Dict]:
        """
        Generate response using LLM
        
        Args:
            query: User query
            context: Retrieved context
            language: Query language
            chat_history: Previous conversation
            
        Returns:
            Tuple of (response, generation_metadata)
        """
        try:
            # Select appropriate prompt template
            if language == "bn":
                prompt_template = self.prompt_template_bn
            else:
                prompt_template = self.prompt_template_en
            
            # Format prompt
            prompt = prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=query
            )
            
            # Generate response with callback to track usage
            with get_openai_callback() as cb:
                if hasattr(self.llm, 'invoke'):
                    # For newer LangChain versions
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                else:
                    # For older versions
                    response_text = self.llm(prompt)
                
                # Generation metadata
                metadata = {
                    "model": self.llm_model,
                    "temperature": self.temperature,
                    "prompt_tokens": cb.prompt_tokens if cb else 0,
                    "completion_tokens": cb.completion_tokens if cb else 0,
                    "total_tokens": cb.total_tokens if cb else 0,
                    "total_cost": cb.total_cost if cb else 0.0
                }
            
            self.logger.info(f"Generated response. Tokens: {metadata.get('total_tokens', 0)}")
            
            return response_text.strip(), metadata
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            error_msg = "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।" if language == "bn" else "Sorry, there was an error generating the response."
            return error_msg, {"error": str(e)}
    
    def process_query(
        self,
        query: str,
        use_memory: bool = True,
        language: str = None,
        **kwargs
    ) -> Dict:
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            query: User query
            use_memory: Whether to use conversation memory
            language: Query language (auto-detected if None)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = datetime.now()
        
        # Detect language if not provided
        if language is None:
            language = self.detect_language(query)
        
        self.logger.info(f"Processing query in {language}: {query[:50]}...")
        
        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(
            query=query,
            language=language,
            num_chunks=kwargs.get('num_chunks'),
            similarity_threshold=kwargs.get('similarity_threshold')
        )
        
        # Format context
        context = self.format_context(chunks)
        
        # Get conversation history if using memory
        chat_history = ""
        if use_memory:
            chat_history = self.memory.get_recent_context(
                num_turns=kwargs.get('context_turns', 2)
            )
        
        # Generate response
        response, generation_metadata = self.generate_response(
            query=query,
            context=context,
            language=language,
            chat_history=chat_history
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = {
            "query": query,
            "response": response,
            "language": language,
            "retrieved_chunks": chunks,
            "num_chunks_retrieved": len(chunks),
            "processing_time_seconds": processing_time,
            "timestamp": start_time.isoformat(),
            "generation_metadata": generation_metadata,
            "context_used": len(context),
            "memory_used": bool(chat_history)
        }
        
        # Add to memory if using memory
        if use_memory:
            chat_turn = ChatTurn(
                timestamp=start_time.isoformat(),
                user_query=query,
                assistant_response=response,
                retrieved_chunks=chunks,
                language=language,
                metadata={"processing_time": processing_time}
            )
            self.memory.add_turn(chat_turn)
        
        self.logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        return result
    
    def add_document(self, file_path: str, doc_id: str = None) -> Dict:
        """
        Add a document to the knowledge base
        
        Args:
            file_path: Path to the document file
            doc_id: Document identifier
            
        Returns:
            Processing result
        """
        try:
            self.logger.info(f"Adding document: {file_path}")
            
            # Process document
            result = self.document_processor.process_document(file_path)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to process document: {result.get('metadata', {}).get('error', 'Unknown error')}"
                }
            
            # Index document
            success = self.vector_store.index_document(result["text"], doc_id)
            
            if success:
                return {
                    "success": True,
                    "document_id": doc_id,
                    "text_length": len(result["text"]),
                    "language": result["language"],
                    "method_used": result["method_used"],
                    "sentences": result["sentence_count"],
                    "collection_stats": self.vector_store.get_collection_info()
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to index document in vector store"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        try:
            vector_stats = self.vector_store.get_collection_info()
            memory_stats = self.memory.get_history_summary()
            
            return {
                "vector_store": vector_stats,
                "memory": memory_stats,
                "llm_model": self.llm_model,
                "temperature": self.temperature,
                "supported_languages": config.SUPPORTED_LANGUAGES
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear_history()
    
    def reset_system(self):
        """Reset the entire system (clear memory and vector store)"""
        self.logger.warning("Resetting entire system...")
        self.memory.clear_history()
        # Note: Vector store reset would be destructive - implement with caution

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create RAG pipeline
    rag = MultilingualRAGPipeline("test_collection")
    
    # Sample Bengali text for testing
    sample_text = """
    অনুপম তার জীবনে অনেক মানুষের সাথে পরিচিত হয়েছে। তার মধ্যে শম্ভুনাথ একজন বিশেষ ব্যক্তি।
    শম্ভুনাথকে অনুপম সুপুরুষ বলে মনে করে। তিনি অনুপমের খুব প্রিয় একজন মানুষ।
    
    অনুপমের মামা তার জীবনে একটি গুরুত্বপূর্ণ ভূমিকা পালন করেছেন। অনুপম তার মামাকে তার ভাগ্য দেবতা বলে মনে করে।
    মামার পরামর্শ এবং সাহায্য অনুপমের জীবনে অনেক পরিবর্তন এনেছে।
    
    কল্যাণীর বিয়ের সময় তার বয়স ছিল মাত্র ১৫ বছর। এই অল্প বয়সেই তাকে বিয়ে দিতে হয়েছিল।
    সেই যুগে এটি একটি সাধারণ ঘটনা ছিল।
    """
    
    # Index sample document
    success = rag.vector_store.index_document(sample_text, "sample_doc")
    
    if success:
        print("Sample document indexed successfully!")
        
        # Test queries
        test_queries = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "কল্যাণীর বিয়ের সময় তার বয়স কত ছিল?",
            "Who is considered a good man according to Anupam?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            
            result = rag.process_query(query)
            
            print(f"Language: {result['language']}")
            print(f"Response: {result['response']}")
            print(f"Chunks retrieved: {result['num_chunks_retrieved']}")
            print(f"Processing time: {result['processing_time_seconds']:.2f}s")
    else:
        print("Failed to index sample document") 
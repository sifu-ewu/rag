"""
Main script for Multilingual RAG System

This script demonstrates the complete RAG pipeline by:
1. Processing the Bengali PDF document
2. Building the vector database
3. Testing with sample queries
4. Evaluating system performance
"""

import logging
import os
import sys
from pathlib import Path
import time
import json
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Local imports
from config import config
from src.rag_pipeline import MultilingualRAGPipeline
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.text_chunker import TextChunker

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RAGSystemDemo:
    """
    Demonstration class for the multilingual RAG system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rag_pipeline = None
        self.pdf_path = "sample.pdf"
        
        # Sample test cases from the assessment
        self.test_cases = [
            {
                "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected": "শম্ভুনাথ",
                "language": "bn"
            },
            {
                "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "expected": "মামাকে",
                "language": "bn"
            },
            {
                "query": "বি বি য়ে য়ে র সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected": "১৫ বছর",
                "language": "bn"
            }
        ]
        
        # Additional test queries for comprehensive testing
        self.additional_queries = [
            "অনুপম কে?",
            "শম্ভুনাথ সম্পর্কে বলুন",
            "মামার ভূমিকা কী?",
            "কল্যাণী কে?",
            "Who is Anupam?",
            "Tell me about the characters in the story",
            "What is the role of uncle in Anupam's life?"
        ]
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        self.logger.info("Checking system prerequisites...")
        
        # Check if PDF file exists
        if not os.path.exists(self.pdf_path):
            self.logger.error(f"PDF file not found: {self.pdf_path}")
            return False
        
        # Check API key
        if not config.OPENAI_API_KEY:
            self.logger.error("OpenAI API key not found in environment variables")
            print("Please set your OpenAI API key:")
            print("export OPENAI_API_KEY='your_api_key_here'")
            return False
        
        # Validate configuration
        if not config.validate_config():
            self.logger.error("Configuration validation failed")
            return False
        
        self.logger.info("All prerequisites met!")
        return True
    
    def initialize_system(self) -> bool:
        """Initialize the RAG system"""
        try:
            self.logger.info("Initializing RAG system...")
            
            # Create RAG pipeline
            self.rag_pipeline = MultilingualRAGPipeline(
                collection_name="bengali_book_rag",
                llm_model=config.LLM_MODEL,
                temperature=config.TEMPERATURE
            )
            
            self.logger.info("RAG system initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def process_document(self) -> bool:
        """Process the Bengali PDF document"""
        try:
            self.logger.info(f"Processing document: {self.pdf_path}")
            
            # Add document to the knowledge base
            result = self.rag_pipeline.add_document(self.pdf_path, "bengali_book")
            
            if result["success"]:
                self.logger.info("Document processed successfully!")
                self.logger.info(f"Document language: {result['language']}")
                self.logger.info(f"Text length: {result['text_length']} characters")
                self.logger.info(f"Extraction method: {result['method_used']}")
                self.logger.info(f"Sentences: {result['sentences']}")
                
                # Print collection statistics
                stats = result["collection_stats"]
                self.logger.info(f"Total chunks in collection: {stats['total_chunks']}")
                self.logger.info(f"Languages: {stats['languages']}")
                self.logger.info(f"Embedding model: {stats['embedding_model']}")
                
                return True
            else:
                self.logger.error(f"Failed to process document: {result['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return False
    
    def test_sample_queries(self) -> Dict:
        """Test the system with sample queries"""
        self.logger.info("Testing with sample queries...")
        
        results = {
            "total_tests": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            self.logger.info(f"\n--- Test Case {i} ---")
            self.logger.info(f"Query: {test_case['query']}")
            self.logger.info(f"Expected: {test_case['expected']}")
            
            try:
                # Process query
                start_time = time.time()
                result = self.rag_pipeline.process_query(
                    query=test_case['query'],
                    language=test_case['language']
                )
                processing_time = time.time() - start_time
                
                response = result['response']
                chunks_retrieved = result['num_chunks_retrieved']
                
                self.logger.info(f"Response: {response}")
                self.logger.info(f"Chunks retrieved: {chunks_retrieved}")
                self.logger.info(f"Processing time: {processing_time:.2f}s")
                
                # Simple evaluation: check if expected answer is in response
                expected_in_response = test_case['expected'].lower() in response.lower()
                
                test_result = {
                    "test_case": i,
                    "query": test_case['query'],
                    "expected": test_case['expected'],
                    "response": response,
                    "chunks_retrieved": chunks_retrieved,
                    "processing_time": processing_time,
                    "expected_found": expected_in_response,
                    "retrieval_success": chunks_retrieved > 0,
                    "language": result['language']
                }
                
                if expected_in_response:
                    results["passed"] += 1
                    self.logger.info("✅ PASSED - Expected answer found in response")
                else:
                    results["failed"] += 1
                    self.logger.warning("❌ FAILED - Expected answer not found in response")
                
                results["test_results"].append(test_result)
                
            except Exception as e:
                self.logger.error(f"Error processing test case {i}: {e}")
                results["failed"] += 1
                results["test_results"].append({
                    "test_case": i,
                    "error": str(e),
                    "expected_found": False
                })
        
        # Calculate success rate
        success_rate = (results["passed"] / results["total_tests"]) * 100
        self.logger.info(f"\n--- Test Results Summary ---")
        self.logger.info(f"Total tests: {results['total_tests']}")
        self.logger.info(f"Passed: {results['passed']}")
        self.logger.info(f"Failed: {results['failed']}")
        self.logger.info(f"Success rate: {success_rate:.1f}%")
        
        return results
    
    def test_additional_queries(self) -> List[Dict]:
        """Test additional queries for system exploration"""
        self.logger.info("\nTesting additional queries...")
        
        additional_results = []
        
        for i, query in enumerate(self.additional_queries, 1):
            self.logger.info(f"\n--- Additional Query {i} ---")
            self.logger.info(f"Query: {query}")
            
            try:
                result = self.rag_pipeline.process_query(query)
                
                self.logger.info(f"Language detected: {result['language']}")
                self.logger.info(f"Response: {result['response']}")
                self.logger.info(f"Chunks retrieved: {result['num_chunks_retrieved']}")
                
                additional_results.append({
                    "query": query,
                    "response": result['response'],
                    "language": result['language'],
                    "chunks_retrieved": result['num_chunks_retrieved'],
                    "processing_time": result['processing_time_seconds']
                })
                
            except Exception as e:
                self.logger.error(f"Error processing additional query {i}: {e}")
                additional_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        return additional_results
    
    def demonstrate_memory(self):
        """Demonstrate conversation memory functionality"""
        self.logger.info("\n--- Demonstrating Conversation Memory ---")
        
        # Have a conversation
        queries = [
            "অনুপম কে?",
            "তার সম্পর্কে আরো বলুন",
            "শম্ভুনাথ সম্পর্কে কী জানেন?",
            "আগের কথোপকথনে আমরা কার কথা বলছিলাম?"
        ]
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"\nConversation turn {i}: {query}")
            result = self.rag_pipeline.process_query(query, use_memory=True)
            self.logger.info(f"Response: {result['response']}")
            self.logger.info(f"Memory used: {result['memory_used']}")
    
    def save_results(self, sample_results: Dict, additional_results: List[Dict]):
        """Save test results to file"""
        results_file = config.DATA_DIR / "test_results.json"
        
        combined_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_config": {
                "llm_model": config.LLM_MODEL,
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k_retrieval": config.TOP_K_RETRIEVAL
            },
            "sample_test_results": sample_results,
            "additional_query_results": additional_results,
            "system_stats": self.rag_pipeline.get_system_stats()
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        self.logger.info("🚀 Starting Multilingual RAG System Demo")
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("Prerequisites not met. Exiting.")
            return False
        
        # Initialize system
        if not self.initialize_system():
            self.logger.error("System initialization failed. Exiting.")
            return False
        
        # Process document
        if not self.process_document():
            self.logger.error("Document processing failed. Exiting.")
            return False
        
        # Test sample queries
        sample_results = self.test_sample_queries()
        
        # Test additional queries
        additional_results = self.test_additional_queries()
        
        # Demonstrate memory
        self.demonstrate_memory()
        
        # Save results
        self.save_results(sample_results, additional_results)
        
        # Final summary
        self.logger.info("\n🎉 Demo completed successfully!")
        self.logger.info("Check the logs and test_results.json for detailed information.")
        
        return True

def main():
    """Main function"""
    print("=" * 60)
    print("    Multilingual RAG System for Bengali and English")
    print("=" * 60)
    print()
    
    # Create and run demo
    demo = RAGSystemDemo()
    success = demo.run_complete_demo()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("Check the logs directory for detailed logs.")
        print("Check data/test_results.json for test results.")
    else:
        print("\n❌ Demo failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    main() 
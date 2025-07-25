"""
Configuration settings for the Multilingual RAG System
"""
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCS_DIR = DATA_DIR / "documents"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    MODELS_DIR = DATA_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, DOCS_DIR, VECTOR_DB_DIR, MODELS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Vector Database Configuration
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")
    CHROMA_PERSIST_DIRECTORY: str = str(VECTOR_DB_DIR / "chroma")
    
    # Document Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Language Configuration
    SUPPORTED_LANGUAGES = ["en", "bn"]  # English and Bengali
    DEFAULT_LANGUAGE = "bn"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Memory Configuration
    MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "4000"))
    
    # Evaluation Configuration
    EVAL_METRICS = ["rouge", "bleu", "bert_score", "semantic_similarity"]
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate essential configuration"""
        if not cls.OPENAI_API_KEY and cls.LLM_MODEL.startswith("gpt"):
            print("Warning: OpenAI API key not found for GPT model")
            return False
        return True

# Global config instance
config = Config() 
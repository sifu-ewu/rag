"""
Enhanced Configuration settings for the Professional Multilingual RAG System
"""
import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import logging

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
    
    # Security Configuration
    JWT_SECRET_KEY: Optional[str] = os.getenv("JWT_SECRET_KEY")
    ADMIN_API_KEY: Optional[str] = os.getenv("ADMIN_API_KEY")
    API_KEYS: List[Dict] = []
    
    # Cache Configuration
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes
    ENABLE_GPU: bool = os.getenv("ENABLE_GPU", "true").lower() == "true"
    
    # Database Configuration
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    
    # Environment Configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    @classmethod
    def load_api_keys_from_file(cls, file_path: str):
        """Load API keys from JSON file"""
        try:
            with open(file_path, 'r') as f:
                cls.API_KEYS = json.load(f)
        except FileNotFoundError:
            logging.warning(f"API keys file not found: {file_path}")
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in API keys file: {file_path}")
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production"""
        return cls.ENVIRONMENT.lower() == "production"
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development"""
        return cls.ENVIRONMENT.lower() == "development"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Enhanced configuration validation"""
        errors = []
        
        # Validate LLM configuration
        if not cls.OPENAI_API_KEY and cls.LLM_MODEL.startswith("gpt"):
            errors.append("OpenAI API key not found for GPT model")
        
        # Validate production requirements
        if cls.is_production():
            if not cls.JWT_SECRET_KEY:
                errors.append("JWT_SECRET_KEY required in production")
            if not cls.ADMIN_API_KEY:
                errors.append("ADMIN_API_KEY required in production")
            if cls.DEBUG:
                errors.append("DEBUG should be False in production")
        
        # Validate database configuration
        if cls.DATABASE_URL and not cls.DATABASE_URL.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            errors.append("Invalid DATABASE_URL format")
        
        # Log errors and return result
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
        
        logging.info("Configuration validation passed")
        return True
    
    @classmethod
    def get_settings_dict(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }

# Global config instance
config = Config() 
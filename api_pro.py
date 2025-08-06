"""
Professional REST API for Multilingual RAG System

Enhanced with:
- Authentication & Authorization
- Rate Limiting
- Caching
- Monitoring & Metrics
- Error Handling
- Performance Tracking
- Health Checks
- API Versioning
"""

import logging
import os
import sys
import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import time
from datetime import datetime, timedelta

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Local imports
from config import config
from src.rag_pipeline import MultilingualRAGPipeline
from src.middleware import (
    SecurityManager, RateLimitMiddleware, SecurityMiddleware,
    get_current_user, get_admin_user, require_permission
)
from src.monitoring import metrics_collector, PerformanceTracker, QueryMetrics
from src.cache import cache_manager

# Setup enhanced logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced Pydantic models
class QueryRequest(BaseModel):
    """Enhanced request model for queries"""
    query: str = Field(..., description="User query in Bengali or English", min_length=1, max_length=5000)
    language: Optional[str] = Field(None, description="Query language (auto-detected if not provided)")
    use_memory: bool = Field(True, description="Whether to use conversation memory")
    use_cache: bool = Field(True, description="Whether to use cached results")
    num_chunks: Optional[int] = Field(None, description="Number of chunks to retrieve", ge=1, le=20)
    similarity_threshold: Optional[float] = Field(None, description="Minimum similarity threshold", ge=0.0, le=1.0)
    stream_response: bool = Field(False, description="Stream the response")
    
    @validator('language')
    def validate_language(cls, v):
        if v and v not in config.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language must be one of {config.SUPPORTED_LANGUAGES}")
        return v

class QueryResponse(BaseModel):
    """Enhanced response model for queries"""
    query: str
    response: str
    language: str
    num_chunks_retrieved: int
    processing_time_seconds: float
    timestamp: str
    memory_used: bool
    cache_used: bool = False
    session_id: Optional[str] = None
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    confidence_score: Optional[float] = None
    
class HealthResponse(BaseModel):
    """Enhanced health check response"""
    status: str
    timestamp: str
    version: str = "2.0.0"
    environment: str = config.ENVIRONMENT
    uptime_seconds: float
    system_metrics: Dict[str, Any]
    api_stats: Dict[str, Any]

class MetricsResponse(BaseModel):
    """Metrics response model"""
    performance_summary: Dict[str, Any]
    cache_stats: Dict[str, Any]
    system_health: Dict[str, Any]

# Global instances
rag_pipeline: Optional[MultilingualRAGPipeline] = None
security_manager = SecurityManager()
sessions: Dict[str, MultilingualRAGPipeline] = {}

# Request ID middleware
class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())
            scope["request_id"] = request_id
        
        return await self.app(scope, receive, send)

def get_rag_pipeline() -> MultilingualRAGPipeline:
    """Enhanced dependency to get or create RAG pipeline"""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = MultilingualRAGPipeline(
            collection_name="pro_multilingual_docs",
            llm_model=config.LLM_MODEL,
            temperature=config.TEMPERATURE
        )
    return rag_pipeline

def get_session_pipeline(session_id: Optional[str] = None) -> MultilingualRAGPipeline:
    """Get or create a session-specific pipeline"""
    if session_id and session_id in sessions:
        return sessions[session_id]
    else:
        return get_rag_pipeline()

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Professional Multilingual RAG System API",
    description="Production-ready REST API for Bengali and English Retrieval-Augmented Generation",
    version="2.0.0",
    docs_url="/v1/docs" if not config.is_production() else None,
    redoc_url="/v1/redoc" if not config.is_production() else None,
    openapi_url="/v1/openapi.json" if not config.is_production() else None
)

# Add middleware stack
app.add_middleware(RequestIDMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware, security_manager=security_manager)

# CORS configuration
if config.is_development():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # More restrictive CORS for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],  # Configure for your domain
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

# Trusted hosts for production
if config.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )

@app.on_event("startup")
async def startup_event():
    """Enhanced startup initialization"""
    logger.info("Starting Professional Multilingual RAG API...")
    
    # Validate configuration
    if not config.validate_config():
        raise Exception("Configuration validation failed")
    
    # Initialize global pipeline
    try:
        get_rag_pipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    
    # Start background metrics collection
    if config.ENABLE_METRICS:
        asyncio.create_task(background_metrics_collection())
    
    logger.info("API startup completed successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced shutdown cleanup"""
    logger.info("Shutting down Professional Multilingual RAG API...")
    
    # Clear sessions
    sessions.clear()
    
    # Clear cache if needed
    if config.ENABLE_CACHING:
        await cache_manager.clear_all()
    
    logger.info("API shutdown completed")

async def background_metrics_collection():
    """Background task for collecting system metrics"""
    while True:
        try:
            metrics_collector.record_system_metrics()
            await asyncio.sleep(60)  # Collect every minute
        except Exception as e:
            logger.error(f"Error in background metrics collection: {e}")
            await asyncio.sleep(60)

# Health and monitoring endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def enhanced_health_check():
    """Enhanced health check with system metrics"""
    health_status = metrics_collector.get_health_status()
    
    return HealthResponse(
        status=health_status["status"],
        timestamp=datetime.now().isoformat(),
        uptime_seconds=health_status["uptime_hours"] * 3600,
        system_metrics=health_status["current_metrics"] or {},
        api_stats=health_status["recent_performance"]
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_enhanced_metrics(
    hours: int = Query(24, description="Hours of history to include", ge=1, le=168),
    user: Dict = Depends(require_permission("read"))
):
    """Get comprehensive system metrics"""
    return MetricsResponse(
        performance_summary=metrics_collector.get_performance_summary(hours),
        cache_stats=cache_manager.get_stats(),
        system_health=metrics_collector.get_health_status()
    )

@app.get("/metrics/export", tags=["System"])
async def export_metrics(
    format_type: str = Query("json", description="Export format: json or prometheus"),
    user: Dict = Depends(get_admin_user)
):
    """Export metrics in various formats"""
    content = metrics_collector.export_metrics(format_type)
    
    if format_type == "prometheus":
        return PlainTextResponse(content, media_type="text/plain")
    else:
        return JSONResponse(content)

# Authentication endpoints
@app.post("/auth/api-key", tags=["Authentication"])
async def generate_api_key(
    name: str = Query(..., description="Name for the API key"),
    permissions: List[str] = Query(["read"], description="Permissions for the key"),
    user: Dict = Depends(get_admin_user)
):
    """Generate a new API key"""
    api_key = security_manager.generate_api_key(name, permissions)
    return {
        "api_key": api_key,
        "name": name,
        "permissions": permissions,
        "created_at": datetime.now().isoformat()
    }

# Enhanced query endpoint
@app.post("/v1/query", response_model=QueryResponse, tags=["Query Processing"])
async def process_enhanced_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = Query(None, description="Session ID for conversation memory"),
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline),
    user: Dict = Depends(require_permission("read"))
):
    """Process a user query with enhanced features"""
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Check cache first
        cached_result = None
        if request.use_cache:
            cached_result = await cache_manager.get_query_result(request.query, request.language)
        
        if cached_result:
            # Return cached result
            response = QueryResponse(
                query=request.query,
                response=cached_result['result']['response'],
                language=cached_result['result']['language'],
                num_chunks_retrieved=cached_result['result']['num_chunks_retrieved'],
                processing_time_seconds=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                memory_used=False,
                cache_used=True,
                session_id=session_id,
                query_id=query_id
            )
        else:
            # Use session-specific pipeline if session_id provided
            if session_id:
                if session_id not in sessions:
                    sessions[session_id] = MultilingualRAGPipeline(
                        collection_name="pro_multilingual_docs",
                        llm_model=config.LLM_MODEL,
                        temperature=config.TEMPERATURE
                    )
                pipeline = sessions[session_id]
            
            # Process query with performance tracking
            with PerformanceTracker("query_processing", {"query_id": query_id}):
                result = pipeline.process_query(
                    query=request.query,
                    use_memory=request.use_memory,
                    language=request.language,
                    num_chunks=request.num_chunks,
                    similarity_threshold=request.similarity_threshold
                )
            
            # Cache result if requested
            if request.use_cache:
                background_tasks.add_task(
                    cache_manager.set_query_result,
                    request.query,
                    result,
                    request.language,
                    config.CACHE_TTL
                )
            
            response = QueryResponse(
                query=result["query"],
                response=result["response"],
                language=result["language"],
                num_chunks_retrieved=result["num_chunks_retrieved"],
                processing_time_seconds=result["processing_time_seconds"],
                timestamp=result["timestamp"],
                memory_used=result["memory_used"],
                cache_used=False,
                session_id=session_id,
                query_id=query_id
            )
        
        # Record metrics
        query_metrics = QueryMetrics(
            query_id=query_id,
            processing_time=time.time() - start_time,
            retrieval_time=0.0,  # Would need to extract from pipeline
            llm_time=0.0,  # Would need to extract from pipeline
            chunks_retrieved=response.num_chunks_retrieved,
            query_length=len(request.query),
            response_length=len(response.response),
            language=response.language,
            success=True
        )
        
        background_tasks.add_task(metrics_collector.record_query_metrics, query_metrics)
        
        return response
        
    except Exception as e:
        # Record error metrics
        error_metrics = QueryMetrics(
            query_id=query_id,
            processing_time=time.time() - start_time,
            query_length=len(request.query),
            language=request.language or "unknown",
            success=False,
            error_type=type(e).__name__
        )
        
        background_tasks.add_task(metrics_collector.record_query_metrics, error_metrics)
        
        logger.error(f"Error processing query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.post("/v1/documents/upload", tags=["Document Management"])
async def upload_document_enhanced(
    file: UploadFile = File(...),
    document_id: Optional[str] = Query(None, description="Document ID"),
    background_tasks: BackgroundTasks,
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline),
    user: Dict = Depends(require_permission("write"))
):
    """Upload and process a document with enhanced features"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate document ID if not provided
    if not document_id:
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    try:
        # Save and process file
        temp_dir = config.DATA_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{document_id}_{file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        with PerformanceTracker("document_processing", {"document_id": document_id}):
            result = pipeline.add_document(str(temp_file_path), document_id)
        
        # Clean up temp file
        background_tasks.add_task(temp_file_path.unlink)
        
        # Invalidate related cache entries
        if config.ENABLE_CACHING:
            background_tasks.add_task(cache_manager.invalidate_document, document_id)
        
        if result["success"]:
            return {
                "success": True,
                "document_id": document_id,
                "filename": file.filename,
                "text_length": result["text_length"],
                "language": result["language"],
                "chunks_created": result["collection_stats"]["total_chunks"],
                "method_used": result["method_used"],
                "processing_time": result.get("processing_time", 0),
                "message": f"Document processed successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management endpoints
@app.get("/v1/cache/stats", tags=["Cache Management"])
async def get_cache_stats(user: Dict = Depends(require_permission("read"))):
    """Get cache statistics"""
    return cache_manager.get_stats()

@app.delete("/v1/cache/clear", tags=["Cache Management"])
async def clear_cache(user: Dict = Depends(get_admin_user)):
    """Clear all cache entries"""
    success = await cache_manager.clear_all()
    return {"success": success, "message": "Cache cleared" if success else "Failed to clear cache"}

# Session management
@app.delete("/v1/sessions/{session_id}", tags=["Session Management"])
async def clear_session_enhanced(
    session_id: str,
    user: Dict = Depends(require_permission("write"))
):
    """Clear a conversation session"""
    try:
        if session_id in sessions:
            sessions[session_id].clear_memory()
            del sessions[session_id]
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced search endpoint
@app.get("/v1/search", tags=["Search"])
async def search_documents_enhanced(
    query: str = Query(..., description="Search query"),
    language: Optional[str] = Query(None, description="Language filter"),
    num_results: int = Query(5, description="Number of results", ge=1, le=50),
    threshold: float = Query(0.7, description="Similarity threshold", ge=0.0, le=1.0),
    use_cache: bool = Query(True, description="Use cached results"),
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline),
    user: Dict = Depends(require_permission("read"))
):
    """Search for relevant document chunks with caching"""
    cache_key = f"search:{query}:{language}:{num_results}:{threshold}"
    
    # Check cache
    if use_cache:
        cached_result = await cache_manager.backend.get(cache_key)
        if cached_result:
            return cached_result
    
    try:
        results = pipeline.vector_store.search_documents(
            query=query,
            n_results=num_results,
            language=language,
            threshold=threshold
        )
        
        response = {
            "query": query,
            "num_results": len(results),
            "results": results,
            "search_time": time.time(),
            "cached": False
        }
        
        # Cache results
        if use_cache:
            await cache_manager.backend.set(cache_key, response, 3600)  # 1 hour TTL
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoints (only in development)
if config.is_development():
    @app.get("/v1/test/sample-queries", tags=["Testing"])
    async def test_sample_queries_enhanced(
        pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)
    ):
        """Test the system with sample queries"""
        test_cases = [
            {
                "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected": "শম্ভুনাথ"
            },
            {
                "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "expected": "মামাকে"
            },
            {
                "query": "বি বি য়ে য়ে র সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected": "১৫ বছর"
            }
        ]
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            try:
                start_time = time.time()
                result = pipeline.process_query(test_case["query"], use_memory=False)
                processing_time = time.time() - start_time
                
                expected_found = test_case["expected"].lower() in result["response"].lower()
                
                results.append({
                    "test_case": i,
                    "query": test_case["query"],
                    "expected": test_case["expected"],
                    "response": result["response"],
                    "expected_found": expected_found,
                    "chunks_retrieved": result["num_chunks_retrieved"],
                    "processing_time": processing_time,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "test_case": i,
                    "query": test_case["query"],
                    "error": str(e),
                    "success": False
                })
        
        # Calculate success metrics
        passed = sum(1 for r in results if r.get("expected_found", False))
        total = len(results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": success_rate,
            "results": results,
            "test_timestamp": datetime.now().isoformat()
        }

# Run the enhanced server
if __name__ == "__main__":
    import uvicorn
    
    # Enhanced server configuration
    server_config = {
        "app": "api_pro:app",
        "host": config.API_HOST,
        "port": config.API_PORT,
        "log_level": config.LOG_LEVEL.lower(),
        "access_log": True,
        "server_header": False,  # Hide server header for security
        "date_header": False,    # Hide date header for security
    }
    
    # Production optimizations
    if config.is_production():
        server_config.update({
            "workers": 4,  # Multiple workers for production
            "reload": False,
            "debug": False
        })
    else:
        server_config.update({
            "reload": True,
            "debug": True
        })
    
    uvicorn.run(**server_config)

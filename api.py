"""
REST API for Multilingual RAG System

This module provides a FastAPI-based REST API for interacting with
the multilingual RAG system.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import time
import uuid
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Local imports
from config import config
from src.rag_pipeline import MultilingualRAGPipeline

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str = Field(..., description="User query in Bengali or English")
    language: Optional[str] = Field(None, description="Query language (auto-detected if not provided)")
    use_memory: bool = Field(True, description="Whether to use conversation memory")
    num_chunks: Optional[int] = Field(None, description="Number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(None, description="Minimum similarity threshold")

class QueryResponse(BaseModel):
    """Response model for queries"""
    query: str
    response: str
    language: str
    num_chunks_retrieved: int
    processing_time_seconds: float
    timestamp: str
    memory_used: bool
    session_id: Optional[str] = None

class DocumentAddRequest(BaseModel):
    """Request model for adding documents"""
    document_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document text content")

class DocumentAddResponse(BaseModel):
    """Response model for document addition"""
    success: bool
    document_id: str
    text_length: int
    language: str
    chunks_created: int
    message: str

class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    vector_store: Dict
    memory: Dict
    llm_model: str
    temperature: float
    supported_languages: List[str]

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    version: str = "1.0.0"

# Global RAG pipeline instance
rag_pipeline: Optional[MultilingualRAGPipeline] = None

# Session management (simple in-memory store)
sessions: Dict[str, MultilingualRAGPipeline] = {}

def get_rag_pipeline() -> MultilingualRAGPipeline:
    """Dependency to get or create RAG pipeline"""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = MultilingualRAGPipeline(
            collection_name="api_multilingual_docs",
            llm_model=config.LLM_MODEL,
            temperature=config.TEMPERATURE
        )
    return rag_pipeline

def get_session_pipeline(session_id: Optional[str] = None) -> MultilingualRAGPipeline:
    """Get or create a session-specific pipeline"""
    if session_id and session_id in sessions:
        return sessions[session_id]
    else:
        # Use global pipeline for non-session requests
        return get_rag_pipeline()

# Create FastAPI app
app = FastAPI(
    title="Multilingual RAG System API",
    description="REST API for Bengali and English Retrieval-Augmented Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    logger.info("Starting Multilingual RAG API...")
    
    # Check configuration
    if not config.validate_config():
        logger.error("Configuration validation failed")
        raise Exception("Invalid configuration")
    
    # Initialize global pipeline
    try:
        get_rag_pipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Multilingual RAG API...")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)):
    """Get system statistics"""
    try:
        stats = pipeline.get_system_stats()
        return SystemStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    session_id: Optional[str] = Query(None, description="Session ID for conversation memory"),
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)
):
    """Process a user query"""
    try:
        # Use session-specific pipeline if session_id provided
        if session_id:
            if session_id not in sessions:
                sessions[session_id] = MultilingualRAGPipeline(
                    collection_name="api_multilingual_docs",
                    llm_model=config.LLM_MODEL,
                    temperature=config.TEMPERATURE
                )
            pipeline = sessions[session_id]
        
        # Process query
        result = pipeline.process_query(
            query=request.query,
            use_memory=request.use_memory,
            language=request.language,
            num_chunks=request.num_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        # Create response
        response = QueryResponse(
            query=result["query"],
            response=result["response"],
            language=result["language"],
            num_chunks_retrieved=result["num_chunks_retrieved"],
            processing_time_seconds=result["processing_time_seconds"],
            timestamp=result["timestamp"],
            memory_used=result["memory_used"],
            session_id=session_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/add", response_model=DocumentAddResponse)
async def add_document(
    request: DocumentAddRequest,
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)
):
    """Add a document to the knowledge base"""
    try:
        # Index the document
        success = pipeline.vector_store.index_document(request.text, request.document_id)
        
        if success:
            # Get collection stats
            stats = pipeline.vector_store.get_collection_info()
            
            # Detect language
            from src.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            language = processor.detect_language(request.text)
            
            return DocumentAddResponse(
                success=True,
                document_id=request.document_id,
                text_length=len(request.text),
                language=language,
                chunks_created=stats.get("total_chunks", 0),
                message="Document added successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to index document")
            
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Query(None, description="Document ID (auto-generated if not provided)"),
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)
):
    """Upload and process a PDF document"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate document ID if not provided
        if not document_id:
            document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Save uploaded file temporarily
        temp_dir = config.DATA_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{document_id}_{file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        result = pipeline.add_document(str(temp_file_path), document_id)
        
        # Clean up temp file
        temp_file_path.unlink()
        
        if result["success"]:
            return DocumentAddResponse(
                success=True,
                document_id=document_id,
                text_length=result["text_length"],
                language=result["language"],
                chunks_created=result["collection_stats"]["total_chunks"],
                message=f"Document '{file.filename}' processed successfully using {result['method_used']}"
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session"""
    try:
        if session_id in sessions:
            sessions[session_id].clear_memory()
            del sessions[session_id]
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory")
async def clear_memory(
    session_id: Optional[str] = Query(None, description="Session ID (clears global memory if not provided)"),
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)
):
    """Clear conversation memory"""
    try:
        if session_id and session_id in sessions:
            sessions[session_id].clear_memory()
            return {"message": f"Memory cleared for session {session_id}"}
        else:
            pipeline.clear_memory()
            return {"message": "Global memory cleared"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    language: Optional[str] = Query(None, description="Language filter"),
    num_results: int = Query(5, description="Number of results"),
    threshold: float = Query(0.7, description="Similarity threshold"),
    pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)
):
    """Search for relevant document chunks"""
    try:
        results = pipeline.vector_store.search_documents(
            query=query,
            n_results=num_results,
            language=language,
            threshold=threshold
        )
        
        return {
            "query": query,
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoints for the sample queries
@app.get("/test/sample-queries")
async def test_sample_queries(pipeline: MultilingualRAGPipeline = Depends(get_rag_pipeline)):
    """Test the system with the sample queries from the assessment"""
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
            result = pipeline.process_query(test_case["query"], use_memory=False)
            
            # Check if expected answer is in response
            expected_found = test_case["expected"].lower() in result["response"].lower()
            
            results.append({
                "test_case": i,
                "query": test_case["query"],
                "expected": test_case["expected"],
                "response": result["response"],
                "expected_found": expected_found,
                "chunks_retrieved": result["num_chunks_retrieved"],
                "processing_time": result["processing_time_seconds"]
            })
            
        except Exception as e:
            results.append({
                "test_case": i,
                "query": test_case["query"],
                "error": str(e)
            })
    
    # Calculate success rate
    passed = sum(1 for r in results if r.get("expected_found", False))
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    return {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": success_rate,
        "results": results
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    ) 
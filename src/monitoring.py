"""
Professional Monitoring and Metrics for RAG System
"""

import time
import psutil
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import json

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Sentry for error tracking (optional)
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

from config import config

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    active_connections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'active_connections': self.active_connections
        }

@dataclass
class QueryMetrics:
    """RAG query performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    query_id: str = ""
    processing_time: float = 0.0
    retrieval_time: float = 0.0
    llm_time: float = 0.0
    chunks_retrieved: int = 0
    query_length: int = 0
    response_length: int = 0
    language: str = "unknown"
    success: bool = True
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'query_id': self.query_id,
            'processing_time': self.processing_time,
            'retrieval_time': self.retrieval_time,
            'llm_time': self.llm_time,
            'chunks_retrieved': self.chunks_retrieved,
            'query_length': self.query_length,
            'response_length': self.response_length,
            'language': self.language,
            'success': self.success,
            'error_type': self.error_type
        }

class MetricsCollector:
    """Professional metrics collection and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_metrics_history = deque(maxlen=1000)
        self.query_metrics_history = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.start_time = datetime.now()
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE and config.ENABLE_METRICS:
            self._init_prometheus_metrics()
        
        # Initialize Sentry if available
        if SENTRY_AVAILABLE and config.SENTRY_DSN:
            self._init_sentry()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.request_counter = Counter(
            'rag_requests_total',
            'Total number of RAG requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'rag_request_duration_seconds',
            'RAG request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.query_processing_time = Histogram(
            'rag_query_processing_seconds',
            'RAG query processing time',
            ['language', 'status']
        )
        
        self.retrieval_time = Histogram(
            'rag_retrieval_seconds',
            'Vector retrieval time',
            ['language']
        )
        
        self.llm_time = Histogram(
            'rag_llm_seconds',
            'LLM processing time',
            ['model', 'language']
        )
        
        self.active_connections = Gauge(
            'rag_active_connections',
            'Number of active connections'
        )
        
        self.system_cpu = Gauge(
            'rag_system_cpu_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory = Gauge(
            'rag_system_memory_percent',
            'System memory usage percentage'
        )
        
        self.chunks_retrieved = Histogram(
            'rag_chunks_retrieved',
            'Number of chunks retrieved per query',
            ['language']
        )
        
        # Start metrics server
        if config.METRICS_PORT:
            try:
                start_http_server(config.METRICS_PORT)
                self.logger.info(f"Prometheus metrics server started on port {config.METRICS_PORT}")
            except Exception as e:
                self.logger.error(f"Failed to start metrics server: {e}")
    
    def _init_sentry(self):
        """Initialize Sentry error tracking"""
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            integrations=[
                FastApiIntegration(),
                LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
            ],
            traces_sample_rate=0.1,
            environment=config.ENVIRONMENT
        )
        self.logger.info("Sentry error tracking initialized")
    
    def record_system_metrics(self):
        """Record current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent
            )
            
            self.system_metrics_history.append(metrics)
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and hasattr(self, 'system_cpu'):
                self.system_cpu.set(cpu_percent)
                self.system_memory.set(memory.percent)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error recording system metrics: {e}")
            return None
    
    def record_query_metrics(self, metrics: QueryMetrics):
        """Record query performance metrics"""
        self.query_metrics_history.append(metrics)
        
        # Update error counts
        if not metrics.success and metrics.error_type:
            self.error_counts[metrics.error_type] += 1
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and hasattr(self, 'query_processing_time'):
            status = 'success' if metrics.success else 'error'
            self.query_processing_time.labels(
                language=metrics.language,
                status=status
            ).observe(metrics.processing_time)
            
            if metrics.success:
                self.retrieval_time.labels(
                    language=metrics.language
                ).observe(metrics.retrieval_time)
                
                self.chunks_retrieved.labels(
                    language=metrics.language
                ).observe(metrics.chunks_retrieved)
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent queries
        recent_queries = [
            q for q in self.query_metrics_history
            if q.timestamp >= cutoff_time
        ]
        
        if not recent_queries:
            return {"message": "No recent query data available"}
        
        # Calculate statistics
        total_queries = len(recent_queries)
        successful_queries = sum(1 for q in recent_queries if q.success)
        avg_processing_time = sum(q.processing_time for q in recent_queries) / total_queries
        avg_retrieval_time = sum(q.retrieval_time for q in recent_queries) / total_queries
        avg_chunks = sum(q.chunks_retrieved for q in recent_queries) / total_queries
        
        # Language breakdown
        language_stats = defaultdict(int)
        for q in recent_queries:
            language_stats[q.language] += 1
        
        # Error breakdown
        error_stats = defaultdict(int)
        for q in recent_queries:
            if not q.success and q.error_type:
                error_stats[q.error_type] += 1
        
        # System metrics
        recent_system = [
            s for s in self.system_metrics_history
            if s.timestamp >= cutoff_time
        ]
        
        avg_cpu = sum(s.cpu_percent for s in recent_system) / len(recent_system) if recent_system else 0
        avg_memory = sum(s.memory_percent for s in recent_system) / len(recent_system) if recent_system else 0
        
        return {
            "period_hours": hours,
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "query_stats": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": (successful_queries / total_queries) * 100,
                "avg_processing_time_seconds": round(avg_processing_time, 3),
                "avg_retrieval_time_seconds": round(avg_retrieval_time, 3),
                "avg_chunks_retrieved": round(avg_chunks, 1)
            },
            "language_breakdown": dict(language_stats),
            "error_breakdown": dict(error_stats),
            "system_stats": {
                "avg_cpu_percent": round(avg_cpu, 1),
                "avg_memory_percent": round(avg_memory, 1),
                "current_time": datetime.now().isoformat()
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        current_metrics = self.record_system_metrics()
        
        # Get recent performance
        recent_summary = self.get_performance_summary(hours=1)
        
        # Determine health status
        status = "healthy"
        issues = []
        
        if current_metrics:
            if current_metrics.cpu_percent > 90:
                status = "warning"
                issues.append("High CPU usage")
            
            if current_metrics.memory_percent > 90:
                status = "warning"
                issues.append("High memory usage")
            
            if current_metrics.disk_usage_percent > 95:
                status = "critical"
                issues.append("Low disk space")
        
        # Check error rate
        if recent_summary["query_stats"]["total_queries"] > 0:
            error_rate = 100 - recent_summary["query_stats"]["success_rate"]
            if error_rate > 50:
                status = "critical"
                issues.append("High error rate")
            elif error_rate > 10:
                status = "warning"
                issues.append("Elevated error rate")
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "issues": issues,
            "current_metrics": current_metrics.to_dict() if current_metrics else None,
            "recent_performance": recent_summary
        }
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in various formats"""
        if format_type == "prometheus" and PROMETHEUS_AVAILABLE:
            return generate_latest().decode('utf-8')
        
        # Default JSON export
        data = {
            "system_metrics": [m.to_dict() for m in self.system_metrics_history],
            "query_metrics": [m.to_dict() for m in self.query_metrics_history],
            "error_counts": dict(self.error_counts),
            "export_timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(data, indent=2)

# Global metrics collector instance
metrics_collector = MetricsCollector()

class PerformanceTracker:
    """Context manager for tracking operation performance"""
    
    def __init__(self, operation_name: str, metadata: Dict = None):
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Log performance
        logging.info(f"{self.operation_name} completed in {duration:.3f}s")
        
        # Record metrics
        if hasattr(metrics_collector, 'record_operation_time'):
            metrics_collector.record_operation_time(
                self.operation_name,
                duration,
                success=exc_type is None,
                metadata=self.metadata
            )

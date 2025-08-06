# Multi-stage Dockerfile for Professional RAG System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-ben \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash rag_user

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements_pro.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements_pro.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/documents data/vector_db data/models logs temp && \
    chown -R rag_user:rag_user /app

# Switch to non-root user
USER rag_user

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "api_pro.py"]

# Development stage
FROM base as development
USER root
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    mypy
USER rag_user
CMD ["python", "api_pro.py"]

# Production stage
FROM base as production
ENV ENVIRONMENT=production
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "api_pro:app"]

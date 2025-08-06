#!/bin/bash

# Professional RAG System Deployment Script
set -e

echo "🚀 Starting Professional RAG System Deployment..."

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="rag-system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed!"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Creating from template..."
        if [ -f "env.example" ]; then
            cp env.example .env
            log_warning "Please edit .env file with your configuration before continuing!"
            exit 1
        else
            log_error "env.example not found!"
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed!"
}

# Validate environment variables
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Source environment variables
    source .env
    
    # Check required variables
    required_vars=("OPENAI_API_KEY" "JWT_SECRET_KEY" "ADMIN_API_KEY")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable $var is not set!"
            exit 1
        fi
    done
    
    # Validate API key format
    if [[ ! "$OPENAI_API_KEY" =~ ^sk-[a-zA-Z0-9]{32,}$ ]]; then
        log_warning "OPENAI_API_KEY format looks incorrect (should start with 'sk-')"
    fi
    
    log_info "Environment validation passed!"
}

# Build and deploy
deploy() {
    log_info "Building and deploying containers..."
    
    # Set environment
    export ENVIRONMENT=$ENVIRONMENT
    
    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose -f $COMPOSE_FILE pull
    
    # Build custom images
    log_info "Building RAG API image..."
    docker-compose -f $COMPOSE_FILE build rag-api
    
    # Start core services
    log_info "Starting core services (Redis, PostgreSQL)..."
    docker-compose -f $COMPOSE_FILE up -d redis postgres
    
    # Wait for databases to be ready
    log_info "Waiting for databases to be ready..."
    sleep 30
    
    # Start main application
    log_info "Starting RAG API..."
    docker-compose -f $COMPOSE_FILE up -d rag-api
    
    # Start monitoring (optional)
    if [[ "$1" == "--with-monitoring" ]]; then
        log_info "Starting monitoring stack..."
        docker-compose -f $COMPOSE_FILE --profile monitoring up -d
    fi
    
    # Start reverse proxy (optional)
    if [[ "$1" == "--with-proxy" || "$2" == "--with-proxy" ]]; then
        log_info "Starting NGINX reverse proxy..."
        docker-compose -f $COMPOSE_FILE --profile proxy up -d
    fi
    
    log_info "Deployment completed!"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for API to be ready
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_info "Health check passed!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: Waiting for API to be ready..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts!"
    return 1
}

# Show status
show_status() {
    log_info "System Status:"
    docker-compose -f $COMPOSE_FILE ps
    
    echo ""
    log_info "Service URLs:"
    echo "  - API: http://localhost:8000"
    echo "  - Health Check: http://localhost:8000/health"
    echo "  - API Documentation: http://localhost:8000/v1/docs"
    echo "  - Metrics: http://localhost:8000/metrics"
    
    if docker-compose -f $COMPOSE_FILE ps | grep -q prometheus; then
        echo "  - Prometheus: http://localhost:9090"
    fi
    
    if docker-compose -f $COMPOSE_FILE ps | grep -q grafana; then
        echo "  - Grafana: http://localhost:3000 (admin/admin)"
    fi
}

# Backup data
backup_data() {
    log_info "Creating data backup..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backup_$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Backup PostgreSQL
    docker-compose exec -T postgres pg_dump -U rag_user rag_db > "$backup_dir/postgres_backup.sql"
    
    # Backup Redis
    docker-compose exec -T redis redis-cli --rdb - > "$backup_dir/redis_backup.rdb"
    
    # Backup application data
    cp -r data "$backup_dir/" 2>/dev/null || true
    
    log_info "Backup created: $backup_dir"
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    docker-compose -f $COMPOSE_FILE down
}

# Update system
update_system() {
    log_info "Updating system..."
    
    # Backup first
    backup_data
    
    # Pull latest changes
    git pull origin main || log_warning "Git pull failed or not a git repository"
    
    # Rebuild and restart
    docker-compose -f $COMPOSE_FILE build rag-api
    docker-compose -f $COMPOSE_FILE up -d
    
    # Health check
    health_check
    
    log_info "Update completed!"
}

# Show logs
show_logs() {
    service=${1:-rag-api}
    docker-compose -f $COMPOSE_FILE logs -f "$service"
}

# Main script logic
case "$1" in
    "deploy")
        check_prerequisites
        validate_environment
        deploy "$2" "$3"
        health_check
        show_status
        ;;
    "status")
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        deploy "$2" "$3"
        health_check
        ;;
    "backup")
        backup_data
        ;;
    "update")
        update_system
        ;;
    "logs")
        show_logs "$2"
        ;;
    "health")
        health_check
        ;;
    *)
        echo "Professional RAG System Deployment Script"
        echo ""
        echo "Usage: $0 {deploy|status|stop|restart|backup|update|logs|health}"
        echo ""
        echo "Commands:"
        echo "  deploy [--with-monitoring] [--with-proxy]  Deploy the system"
        echo "  status                                     Show system status"
        echo "  stop                                       Stop all services"
        echo "  restart [--with-monitoring] [--with-proxy] Restart the system"
        echo "  backup                                     Create data backup"
        echo "  update                                     Update system"
        echo "  logs [service]                             Show logs (default: rag-api)"
        echo "  health                                     Check system health"
        echo ""
        echo "Examples:"
        echo "  $0 deploy                    # Basic deployment"
        echo "  $0 deploy --with-monitoring  # Deploy with Prometheus/Grafana"
        echo "  $0 logs redis               # Show Redis logs"
        echo ""
        exit 1
        ;;
esac

log_info "Script completed successfully!"

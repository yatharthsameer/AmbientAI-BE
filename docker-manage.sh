#!/bin/bash
# Docker management scripts for Nurse Conversation Processing API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Function to build all images
build() {
    print_status "Building Docker images..."
    docker-compose build --no-cache
    print_success "Docker images built successfully!"
}

# Function to start all services
start() {
    print_status "Starting all services..."
    check_docker
    docker-compose up -d
    print_success "All services started!"
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    sleep 10
    
    # Check health
    health
}

# Function to start development environment
dev() {
    print_status "Starting development environment..."
    check_docker
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    print_success "Development environment started!"
    
    print_status "Services available at:"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Flower (Celery Monitor): http://localhost:5555"
    echo "  • Adminer (DB Admin): http://localhost:8080"
    echo "  • Redis Commander: http://localhost:8081"
    echo "  • Jupyter Lab: http://localhost:8888"
}

# Function to stop all services
stop() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped!"
}

# Function to restart all services
restart() {
    print_status "Restarting all services..."
    stop
    start
}

# Function to view logs
logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        print_status "Showing logs for service: $service"
        docker-compose logs -f "$service"
    else
        print_status "Showing logs for all services..."
        docker-compose logs -f
    fi
}

# Function to check service health
health() {
    print_status "Checking service health..."
    
    # Check API health
    if curl -f -s http://localhost:8000/health > /dev/null; then
        print_success "API is healthy"
    else
        print_warning "API is not responding"
    fi
    
    # Check PostgreSQL
    if docker-compose exec ambientai_be_postgres pg_isready -U postgres > /dev/null; then
        print_success "PostgreSQL is healthy"
    else
        print_warning "PostgreSQL is not ready"
    fi
    
    # Check Redis
    if docker-compose exec ambientai_be_redis redis-cli ping | grep -q PONG; then
        print_success "Redis is healthy"
    else
        print_warning "Redis is not responding"
    fi
}

# Function to run database migrations
migrate() {
    print_status "Running database migrations..."
    docker-compose run --rm ambientai_be_migrate
    print_success "Database migrations completed!"
}

# Function to create a new migration
makemigrations() {
    local message=${1:-"Auto migration"}
    print_status "Creating new migration: $message"
    docker-compose exec ambientai_be_api alembic revision --autogenerate -m "$message"
    print_success "Migration created!"
}

# Function to reset database
reset_db() {
    print_warning "This will DELETE ALL DATA in the database!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_status "Resetting database..."
        docker-compose down
        docker volume rm nurse_conversation_processor_ambientai_be_postgres_data 2>/dev/null || true
        docker-compose up -d ambientai_be_postgres
        sleep 5
        migrate
        print_success "Database reset complete!"
    else
        print_status "Database reset cancelled."
    fi
}

# Function to run tests
test() {
    print_status "Running tests..."
    docker-compose exec ambientai_be_api pytest -v
}

# Function to run shell in container
shell() {
    local service=${1:-ambientai_be_api}
    print_status "Opening shell in $service container..."
    docker-compose exec "$service" /bin/bash
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful!)
    read -p "Remove unused volumes? This may delete data! (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        docker volume prune -f
    fi
    
    print_success "Cleanup complete!"
}

# Function to backup database
backup() {
    local backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
    print_status "Creating database backup: $backup_file"
    
    docker-compose exec ambientai_be_postgres pg_dump -U postgres nurse_conversations > "$backup_file"
    print_success "Database backup saved to: $backup_file"
}

# Function to restore database
restore() {
    local backup_file=$1
    if [ -z "$backup_file" ]; then
        print_error "Please provide backup file path"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    print_warning "This will restore database from: $backup_file"
    read -p "Continue? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_status "Restoring database..."
        docker-compose exec -T ambientai_be_postgres psql -U postgres -d nurse_conversations < "$backup_file"
        print_success "Database restored successfully!"
    fi
}

# Function to show help
help() {
    echo "Nurse Conversation Processing API - Docker Management Script"
    echo
    echo "Usage: $0 <command>"
    echo
    echo "Commands:"
    echo "  build           Build all Docker images"
    echo "  start           Start all services"
    echo "  dev             Start development environment with hot reload"
    echo "  stop            Stop all services"
    echo "  restart         Restart all services"
    echo "  logs [service]  Show logs (optionally for specific service)"
    echo "  health          Check service health"
    echo "  migrate         Run database migrations"
    echo "  makemigrations  Create new database migration"
    echo "  reset_db        Reset database (WARNING: deletes all data)"
    echo "  test            Run tests"
    echo "  shell [service] Open shell in container (default: ambientai_be_api)"
    echo "  cleanup         Clean up Docker resources"
    echo "  backup          Create database backup"
    echo "  restore <file>  Restore database from backup"
    echo "  help            Show this help message"
    echo
    echo "Examples:"
    echo "  $0 dev                    # Start development environment"
    echo "  $0 logs ambientai_be_api               # Show API logs"
    echo "  $0 shell ambientai_be_worker           # Open shell in worker container"
    echo "  $0 backup                 # Create database backup"
    echo "  $0 restore backup.sql     # Restore from backup"
}

# Main command dispatcher
case "${1:-}" in
    build)
        build
        ;;
    start)
        start
        ;;
    dev)
        dev
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs "$2"
        ;;
    health)
        health
        ;;
    migrate)
        migrate
        ;;
    makemigrations)
        makemigrations "$2"
        ;;
    reset_db)
        reset_db
        ;;
    test)
        test
        ;;
    shell)
        shell "$2"
        ;;
    cleanup)
        cleanup
        ;;
    backup)
        backup
        ;;
    restore)
        restore "$2"
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: ${1:-}"
        echo
        help
        exit 1
        ;;
esac
#!/bin/bash

# Quantum-Enhanced Medical Imaging AI System Deployment Script
# This script sets up the complete production environment

set -e

echo "🚀 Starting Quantum-Enhanced Medical Imaging AI System Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="quantum-medical-llava"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

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

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if NVIDIA Docker is available (for GPU support)
check_nvidia_docker() {
    print_status "Checking NVIDIA Docker support..."
    if command -v nvidia-docker &> /dev/null; then
        print_success "NVIDIA Docker is available"
        export NVIDIA_DOCKER_AVAILABLE=true
    else
        print_warning "NVIDIA Docker not found. GPU acceleration may not be available."
        export NVIDIA_DOCKER_AVAILABLE=false
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "uploads"
        "temp/dicom"
        "temp/nifti"
        "reports"
        "logs"
        "models"
        "data"
        "ssl"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "All directories created"
}

# Set up environment configuration
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example "$ENV_FILE"
            print_warning "Copied .env.example to .env. Please update with your actual values."
        else
            print_error ".env.example file not found"
            exit 1
        fi
    else
        print_status ".env file already exists"
    fi
    
    # Generate SSL certificates for development
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        print_status "Generating self-signed SSL certificates for development..."
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        print_success "SSL certificates generated"
    fi
}

# Pull Docker images
pull_images() {
    print_status "Pulling Docker images..."
    docker-compose pull
    print_success "Docker images pulled"
}

# Build custom images
build_images() {
    print_status "Building application images..."
    docker-compose build --no-cache
    print_success "Application images built"
}

# Start services
start_services() {
    print_status "Starting services..."
    docker-compose up -d
    print_success "Services started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    # Wait for database
    print_status "Waiting for database..."
    timeout=60
    counter=0
    while ! docker-compose exec postgres pg_isready -U medical_user -d medical_analysis > /dev/null 2>&1; do
        if [ $counter -eq $timeout ]; then
            print_error "Database failed to start within $timeout seconds"
            exit 1
        fi
        sleep 1
        counter=$((counter + 1))
    done
    print_success "Database is ready"
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    timeout=30
    counter=0
    while ! docker-compose exec redis redis-cli ping > /dev/null 2>&1; do
        if [ $counter -eq $timeout ]; then
            print_error "Redis failed to start within $timeout seconds"
            exit 1
        fi
        sleep 1
        counter=$((counter + 1))
    done
    print_success "Redis is ready"
    
    # Wait for main application
    print_status "Waiting for main application..."
    timeout=120
    counter=0
    while ! curl -f http://localhost:5000/health > /dev/null 2>&1; do
        if [ $counter -eq $timeout ]; then
            print_error "Application failed to start within $timeout seconds"
            exit 1
        fi
        sleep 1
        counter=$((counter + 1))
    done
    print_success "Application is ready"
}

# Run database migrations (if needed)
run_migrations() {
    print_status "Running database migrations..."
    # Add migration commands here if needed
    print_success "Database migrations completed"
}

# Display service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_success "Deployment completed successfully!"
    echo ""
    print_status "Access the application at:"
    echo "  • HTTP:  http://localhost"
    echo "  • HTTPS: https://localhost"
    echo "  • API:   http://localhost/api"
    echo ""
    print_status "Monitoring dashboards:"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana:    http://localhost:3000 (admin/admin123)"
    echo ""
    print_warning "Please update the .env file with your actual configuration values!"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down
    print_success "Cleanup completed"
}

# Health check
health_check() {
    print_status "Running health checks..."
    
    services=("redis" "postgres" "quantum-medical-app")
    
    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up"; then
            print_success "$service is running"
        else
            print_error "$service is not running"
            return 1
        fi
    done
    
    # Test API endpoint
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        return 1
    fi
    
    print_success "All health checks passed"
}

# Main deployment function
main() {
    case "${1:-deploy}" in
        "deploy")
            check_docker
            check_nvidia_docker
            create_directories
            setup_environment
            pull_images
            build_images
            start_services
            wait_for_services
            run_migrations
            health_check
            show_status
            ;;
        "start")
            start_services
            wait_for_services
            show_status
            ;;
        "stop")
            print_status "Stopping services..."
            docker-compose down
            print_success "Services stopped"
            ;;
        "restart")
            print_status "Restarting services..."
            docker-compose restart
            wait_for_services
            show_status
            ;;
        "status")
            docker-compose ps
            health_check
            ;;
        "logs")
            docker-compose logs -f
            ;;
        "clean")
            print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                cleanup
                docker-compose down -v --rmi all
                docker system prune -f
                print_success "Complete cleanup performed"
            else
                print_status "Cleanup cancelled"
            fi
            ;;
        "help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  start    - Start services"
            echo "  stop     - Stop services"
            echo "  restart  - Restart services"
            echo "  status   - Show service status"
            echo "  logs     - Show service logs"
            echo "  clean    - Clean up all containers and images"
            echo "  help     - Show this help message"
            ;;
        *)
            print_error "Unknown command: $1"
            print_status "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Trap for cleanup on exit
trap cleanup EXIT

# Run main function with arguments
main "$@"
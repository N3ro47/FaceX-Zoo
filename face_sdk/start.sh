#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_error() {
    echo -e "${RED}[!]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[*]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. This is not recommended for security reasons."
    read -p "Do you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if camera device exists
if [ ! -e "/dev/video0" ]; then
    print_warning "Camera device /dev/video0 not found. The application might not work properly."
    read -p "Do you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set up X11 forwarding
print_status "Setting up X11 forwarding..."
xhost +local:docker

# Create necessary directories if they don't exist
print_status "Creating necessary directories..."
mkdir -p config tmp logs

# Function to run the application
run_application() {
    case $1 in
        "configurator")
            print_status "Starting Face Recognition Configurator..."
            docker-compose up --build
            ;;
        "main")
            print_status "Starting Face Recognition Main Application..."
            docker-compose run --rm face-recognition python api_usage/main.py -m image_taker --load /app/config/config.json
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
}

# Main menu
while true; do
    echo -e "\n${GREEN}Face Recognition Docker Setup${NC}"
    echo "1. Run Configurator (GUI)"
    echo "2. Run Main Application (Camera Mode)"
    echo "3. Clean up Docker resources"
    echo "4. Exit"
    read -p "Select an option (1-4): " choice

    case $choice in
        1)
            run_application "configurator"
            ;;
        2)
            run_application "main"
            ;;
        3)
            print_status "Cleaning up Docker resources..."
            docker-compose down
            docker system prune -f
            print_status "Cleanup complete"
            ;;
        4)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
done

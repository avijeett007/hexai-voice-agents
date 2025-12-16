#!/bin/bash

# Hexai LiveKit Agent - Production Startup Script
# This script helps you start the agent with proper checks

set -e

echo "🚀 Hexai LiveKit Agent - Production Startup"
echo "==========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please create .env file from template:"
    echo "  cp .env.production.example .env"
    echo "  nano .env  # Fill in your credentials"
    echo ""
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed!"
    echo "Please install docker-compose and try again."
    exit 1
fi

echo "✅ Pre-flight checks passed"
echo ""

# Ask user what to do
echo "What would you like to do?"
echo "1) Start agent (build if needed)"
echo "2) Rebuild and start agent"
echo "3) Stop agent"
echo "4) View logs"
echo "5) Check status"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🔨 Building and starting agent..."
        docker-compose up -d
        echo ""
        echo "✅ Agent started successfully!"
        echo ""
        echo "View logs with: docker-compose logs -f"
        ;;
    2)
        echo ""
        echo "🔨 Rebuilding agent from scratch..."
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        echo ""
        echo "✅ Agent rebuilt and started successfully!"
        echo ""
        echo "View logs with: docker-compose logs -f"
        ;;
    3)
        echo ""
        echo "🛑 Stopping agent..."
        docker-compose down
        echo ""
        echo "✅ Agent stopped successfully!"
        ;;
    4)
        echo ""
        echo "📋 Showing logs (Ctrl+C to exit)..."
        echo ""
        docker-compose logs -f
        ;;
    5)
        echo ""
        echo "📊 Agent Status:"
        echo "==============="
        docker-compose ps
        echo ""
        echo "Health Status:"
        docker inspect hexai-livekit-agent --format='{{.State.Health.Status}}' 2>/dev/null || echo "Container not running"
        echo ""
        echo "Recent Logs:"
        docker-compose logs --tail=20
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done! 🎉"


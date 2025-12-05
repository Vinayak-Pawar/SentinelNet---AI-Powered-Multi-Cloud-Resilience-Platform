#!/bin/bash

# SentinelNet Setup and Run Script
# Version: 1.0
# Author: Vinayak Pawar
# Description: One-click setup and execution for SentinelNet project
# Compatible with: M1 Pro MacBook Pro (Apple Silicon)

set -e  # Exit on any error

echo "🚀 SentinelNet Project Setup and Execution Script"
echo "================================================="

# Check if running on macOS with Apple Silicon
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS. Please run on macOS."
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: Not running on Apple Silicon. Some optimizations may not work."
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command_exists uv; then
    echo "❌ UV is not installed. Please install it first: https://github.com/astral-sh/uv"
    echo "   Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker Desktop for Mac."
    echo "   Download from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3."
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Create Python environment
echo "🐍 Creating Python UV environment..."
uv venv sentinelnet_env

# Activate environment
echo "🔄 Activating environment..."
source sentinelnet_env/bin/activate

# Install requirements
echo "📦 Installing requirements..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
    echo "✅ Requirements installed successfully!"
else
    echo "⚠️  requirements.txt not found. Installing basic dependencies..."
    uv pip install langchain langchain-community langgraph openai google-cloud-bigquery azure-storage-blob azure-devops python-dotenv fastapi uvicorn streamlit
fi

# Create .env file if it doesn't exist
echo "🔐 Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# SentinelNet Configuration
# IMPORTANT: Edit this file with your actual API keys before running the project

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id

# Azure DevOps Configuration
AZURE_DEVOPS_ORGANIZATION=your-organization
AZURE_DEVOPS_PAT=your-personal-access-token

# OpenAI/ChatGPT Configuration
OPENAI_API_KEY=your-openai-api-key

# Other API Keys (Optional)
NVIDIA_NIM_API_KEY=your-nvidia-api-key

# Application Configuration
DEBUG=true
LOG_LEVEL=INFO
AGENT_DISCOVERY_PORT=8080
DASHBOARD_PORT=8501
API_PORT=8000

# Safety Settings
ALLOW_AUTOMATED_ACTIONS=false
REQUIRE_HUMAN_APPROVAL=true

# Database Configuration (Local)
DATABASE_URL=sqlite:///./sentinelnet.db
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-influx-token
INFLUXDB_ORG=sentinelnet
INFLUXDB_BUCKET=outages

# Communication Settings
USE_P2P_COMMUNICATION=true
WEBRTC_SIGNALING_SERVER=ws://localhost:8080
FALLBACK_EMAIL_ENABLED=false
FALLBACK_SMS_ENABLED=false

EOF
    echo "✅ Created .env file with placeholder values."
    echo "⚠️  IMPORTANT: Please edit the .env file with your actual API keys before running the project!"
else
    echo "ℹ️  .env file already exists. Skipping creation."
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs data models agents dashboard docs

# Setup complete message
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. ✏️  Edit the .env file with your actual API keys and configuration"
echo "2. 🏃 Run the project with: python main.py"
echo "3. 🌐 Open your browser to http://localhost:8501 for the dashboard"
echo "4. 📊 Open http://localhost:8000/docs for the API documentation"
echo ""
echo "📖 For detailed setup instructions, see README.md"
echo "🆘 If you encounter issues, check the troubleshooting section in docs/"
echo ""
echo "💡 Quick start commands:"
echo "   • Start dashboard only: streamlit run dashboard/app.py"
echo "   • Start API only: uvicorn api.main:app --reload"
echo "   • Run all agents: python -m agents.orchestrator"
echo ""

# Optional: Ask if user wants to run the project immediately
read -p "❓ Would you like to start the SentinelNet dashboard now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting SentinelNet Dashboard..."
    if [ -f "dashboard/app.py" ]; then
        streamlit run dashboard/app.py
    elif [ -f "main.py" ]; then
        python main.py
    else
        echo "⚠️  No main application file found. Please run manually after setup."
    fi
fi

echo ""
echo "🎯 SentinelNet is ready! Check the .env file and start building amazing things! 🚀"

#!/bin/bash

# Startup script for Quantum-Enhanced Medical Imaging AI System Backend

echo "🧠 Quantum-Enhanced Medical Imaging AI System - Backend"
echo "======================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python -c "import flask, torch, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed. Please run: pip install -r requirements_simple.txt"
    exit 1
fi

# Start Flask backend
echo "🚀 Starting Flask backend server on port 5000..."
echo "📊 Backend API: http://localhost:5000"
echo "🔍 Health Check: http://localhost:5000/api/health"
echo "📈 System Status: http://localhost:5000/api/status"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python flask_api/app_minimal.py

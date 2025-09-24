#!/bin/bash

# Startup script for Quantum-Enhanced Medical Imaging AI System Frontend

echo "🌐 Quantum-Enhanced Medical Imaging AI System - Frontend"
echo "======================================================="

# Check if demo.html exists
if [ ! -f "demo.html" ]; then
    echo "❌ demo.html not found in current directory."
    exit 1
fi

# Start HTTP server for frontend
echo "🚀 Starting frontend server on port 3000..."
echo "🌐 Frontend UI: http://localhost:3000/demo.html"
echo "📁 Static files: http://localhost:3000/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m http.server 3000

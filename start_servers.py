#!/usr/bin/env python3
"""
Startup script for Quantum-Enhanced Medical Imaging AI System
Runs both Flask backend and HTTP frontend servers
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def run_flask_server():
    """Run Flask backend server"""
    print("🚀 Starting Flask backend server on port 5000...")
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Activate virtual environment and run Flask
        if os.name == 'nt':  # Windows
            cmd = ["venv\\Scripts\\activate", "&&", "python", "flask_api/app.py"]
        else:  # Unix/Linux/macOS
            cmd = ["source", "venv/bin/activate", "&&", "python", "flask_api/app.py"]
        
        # Use shell=True for cross-platform compatibility
        process = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ Flask backend server started successfully!")
        return process
        
    except Exception as e:
        print(f"❌ Error starting Flask server: {e}")
        return None

def run_frontend_server():
    """Run HTTP frontend server"""
    print("🌐 Starting frontend server on port 3000...")
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Start HTTP server
        process = subprocess.Popen(
            [sys.executable, "-m", "http.server", "3000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ Frontend server started successfully!")
        return process
        
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")
        return None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down servers...")
    sys.exit(0)

def main():
    """Main function to start both servers"""
    print("🧠 Quantum-Enhanced Medical Imaging AI System")
    print("=" * 50)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Flask backend
    flask_process = run_flask_server()
    if not flask_process:
        print("❌ Failed to start Flask server. Exiting.")
        sys.exit(1)
    
    # Wait a moment for Flask to start
    time.sleep(2)
    
    # Start frontend server
    frontend_process = run_frontend_server()
    if not frontend_process:
        print("❌ Failed to start frontend server. Exiting.")
        flask_process.terminate()
        sys.exit(1)
    
    print("\n🎉 Both servers are running!")
    print("📊 Backend API: http://localhost:5000")
    print("🌐 Frontend UI: http://localhost:3000/demo.html")
    print("🔍 Health Check: http://localhost:5000/api/health")
    print("📈 System Status: http://localhost:5000/api/status")
    print("\nPress Ctrl+C to stop both servers")
    
    try:
        # Keep both processes running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if flask_process.poll() is not None:
                print("❌ Flask server stopped unexpectedly")
                break
            
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    
    finally:
        # Clean up processes
        if flask_process:
            flask_process.terminate()
            flask_process.wait()
        
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
        
        print("✅ Servers stopped successfully")

if __name__ == "__main__":
    main()

# 🧠 Quantum-Enhanced Medical Imaging AI System - Run Instructions

## 🎯 Quick Start Guide

This system provides a complete Quantum-Enhanced Medical Imaging AI solution with both frontend and backend components.

## 📋 Prerequisites

- Python 3.9 or higher
- macOS, Linux, or Windows
- At least 4GB RAM
- 2GB free disk space

## 🚀 Setup Instructions

### 1. Environment Setup

```bash
# Navigate to the project directory
cd /Users/nikhil/Documents/vortex

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements_simple.txt
```

### 2. Run the System

#### Option A: Using Startup Scripts (Recommended)

**Terminal 1 - Backend Server:**
```bash
cd /Users/nikhil/Documents/vortex
./start_backend.sh
```

**Terminal 2 - Frontend Server:**
```bash
cd /Users/nikhil/Documents/vortex
./start_frontend.sh
```

#### Option B: Manual Startup

**Terminal 1 - Backend:**
```bash
cd /Users/nikhil/Documents/vortex
source venv/bin/activate
python flask_api/app.py
```

**Terminal 2 - Frontend:**
```bash
cd /Users/nikhil/Documents/vortex
python -m http.server 3000
```

## 🌐 Access Points

Once both servers are running, you can access:

- **🌐 Frontend UI**: http://localhost:3000/demo.html
- **📊 Backend API**: http://localhost:5000
- **🔍 Health Check**: http://localhost:5000/api/health
- **📈 System Status**: http://localhost:5000/api/status

## 🧪 Testing the System

### 1. Health Check
```bash
curl http://localhost:5000/api/health
```

### 2. System Status
```bash
curl http://localhost:5000/api/status
```

### 3. Authentication
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'
```

### 4. Image Analysis (with authentication)
```bash
# Get token first
TOKEN=$(curl -s -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}' \
  | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)

# Analyze image
curl -X POST http://localhost:5000/api/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@test_medical_image.png" \
  -F "query=Analyze this medical image"
```

## 🎭 Demo Features

The frontend demo includes:

1. **🔍 Image Analysis Tab**
   - Drag & drop medical images
   - Support for DICOM, NIfTI, JPG, PNG formats
   - Quantum-enhanced analysis with confidence metrics
   - Mock analysis results for demonstration

2. **💬 Chat Tab**
   - Interactive medical AI consultation
   - Context-aware responses
   - Medical terminology processing

3. **📊 Dashboard Tab**
   - System metrics and analytics
   - Analysis history
   - Performance monitoring

4. **ℹ️ About Tab**
   - System information
   - Technical specifications
   - Feature overview

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key

# OpenAI API (optional, for enhanced chat)
OPENAI_API_KEY=your-openai-api-key

# Model Configuration
LLAVA_MODEL_PATH=microsoft/llava-med-v1.5-mistral-7b
QUANTUM_BACKEND=qasm_simulator
MAX_IMAGE_SIZE=1024
CONFIDENCE_THRESHOLD=0.7
```

### Model Configuration

The system uses mock models by default since the full LLaVA-Med model requires significant resources. The mock models provide:

- Simulated medical image analysis
- Quantum-enhanced feature extraction (simulated)
- Uncertainty quantification
- Confidence scoring

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing processes
   pkill -f "python flask_api/app.py"
   pkill -f "http.server"
   ```

2. **Dependencies Not Installed**
   ```bash
   pip install -r requirements_simple.txt
   ```

3. **Virtual Environment Issues**
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_simple.txt
   ```

4. **Permission Issues (macOS/Linux)**
   ```bash
   chmod +x start_backend.sh start_frontend.sh
   ```

### Log Files

Check the console output for detailed logs. The system provides comprehensive logging for:
- Model initialization
- API requests
- Analysis processing
- Error handling

## 📁 Project Structure

```
vortex/
├── flask_api/              # Flask backend API
│   └── app.py             # Main Flask application
├── llava_medical/         # LLaVA-Med integration
│   └── __init__.py        # Medical model wrapper
├── explainable_ai/        # XAI implementations
│   └── __init__.py        # Grad-CAM, LIME, SHAP
├── quantum_circuits.py    # Quantum computing module
├── medical_analyzer.py    # Core analysis engine
├── medical_chatbot.py     # AI consultation system
├── config.py             # Configuration management
├── demo.html             # Frontend interface
├── requirements_simple.txt # Dependencies
├── start_backend.sh      # Backend startup script
├── start_frontend.sh     # Frontend startup script
└── test_medical_image.png # Sample test image
```

## 🔬 Technical Details

### Backend Architecture

- **Flask API**: RESTful endpoints with JWT authentication
- **Quantum Processing**: Simulated quantum circuits for feature enhancement
- **Medical Analysis**: Mock LLaVA-Med integration with fallback models
- **Explainable AI**: Grad-CAM, LIME, SHAP implementations (simulated)

### Frontend Features

- **Modern UI**: Responsive design with tabbed interface
- **Real-time Analysis**: Live progress indicators and results
- **Interactive Chat**: Medical consultation interface
- **Dashboard Analytics**: System metrics and history

### Security Features

- JWT-based authentication
- CORS protection
- Input validation
- Secure file handling

## 🚀 Production Deployment

For production deployment, consider:

1. **Environment Configuration**
   - Set `FLASK_ENV=production`
   - Use strong secret keys
   - Configure proper CORS origins

2. **Database Integration**
   - Replace in-memory storage with PostgreSQL/Redis
   - Implement proper session management

3. **Model Deployment**
   - Deploy actual LLaVA-Med models
   - Configure GPU resources
   - Implement model versioning

4. **Security Hardening**
   - HTTPS configuration
   - Rate limiting
   - Input sanitization
   - Audit logging

## 📞 Support

For issues or questions:

1. Check the console logs for error details
2. Verify all dependencies are installed
3. Ensure ports 5000 and 3000 are available
4. Test individual components using the test scripts

## 🎉 Success Indicators

When everything is working correctly, you should see:

- ✅ Flask server running on port 5000
- ✅ HTTP server running on port 3000
- ✅ Health check returns `{"status":"healthy"}`
- ✅ Frontend accessible at http://localhost:3000/demo.html
- ✅ Authentication working with test credentials
- ✅ Mock analysis results displayed in the UI

---

**🎯 Ready to explore Quantum-Enhanced Medical Imaging AI!**

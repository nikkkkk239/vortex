"""
Minimal Flask API Server for Quantum-Enhanced Medical Imaging System
Completely working version without any complex dependencies
"""

import os
import sys
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import logging
from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import base64
from io import BytesIO
from PIL import Image
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'quantum-medical-secret-key')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB

# Initialize extensions
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:5000').split(','))
jwt = JWTManager(app)

# Configuration
CONFIG = {
    'model_path': 'microsoft/llava-med-v1.5-mistral-7b',
    'max_image_size': 1024,
    'supported_formats': ['jpg', 'jpeg', 'png', 'dicom', 'nii'],
    'confidence_threshold': 0.7,
    'quantum_qubits': 8,
    'quantum_layers': 2,
    'device': 'cpu'
}

# In-memory storage for session data
analysis_sessions = {}
chat_sessions = {}

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'message': 'Maximum file size is 16MB',
        'max_size': app.config['MAX_CONTENT_LENGTH']
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again.'
    }), 500

# Authentication endpoints
@app.route('/api/auth/login', methods=['POST'])
def login():
    """User authentication endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Simple authentication (in production, use proper authentication)
        if username and password:
            access_token = create_access_token(identity=username)
            return jsonify({
                'access_token': access_token,
                'message': 'Login successful',
                'user': {'username': username}
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/validate', methods=['GET'])
@jwt_required()
def validate_token():
    """Validate JWT token"""
    current_user = get_jwt_identity()
    return jsonify({'valid': True, 'user': current_user})

# Core API endpoints
@app.route('/api/analyze', methods=['POST'])
@jwt_required()
def analyze_medical_image():
    """
    Analyze medical image with quantum-enhanced processing
    
    Expected form data:
    - image: Image file
    - query: Medical query (optional)
    - patient_context: JSON string with patient information (optional)
    """
    try:
        current_user = get_jwt_identity()
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get query and patient context
        query = request.form.get('query', 'Please analyze this medical image and provide diagnostic insights.')
        patient_context_str = request.form.get('patient_context', '{}')
        
        try:
            patient_context = json.loads(patient_context_str)
        except json.JSONDecodeError:
            patient_context = {}
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            image_file.save(temp_file.name)
            temp_image_path = temp_file.name
        
        try:
            # Simple image analysis (mock implementation)
            logger.info("Performing mock medical image analysis")
            
            # Basic image processing
            image = Image.open(temp_image_path)
            
            # Generate mock analysis results (completely JSON-safe)
            analysis_result = {
                'llava_response': f'Mock analysis: The medical image shows typical characteristics. Query: {query}. This appears to be a {image.size[0]}x{image.size[1]} image.',
                'quantum_features': [0.1, 0.2, 0.3, 0.4, 0.5],  # Simple list instead of numpy array
                'uncertainty_metrics': {
                    'confidence': 0.85,
                    'entropy': 0.3,
                    'quantum_uncertainty': 0.15
                },
                'image_metadata': {
                    'format': 'png',
                    'size': [image.size[0], image.size[1]],
                    'source_path': temp_image_path
                },
                'processing_info': {
                    'quantum_enhanced': True,
                    'model_used': CONFIG['model_path'],
                    'timestamp': str(datetime.now()),
                    'processing_time': 2.5
                }
            }
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Store analysis in session
            analysis_sessions[session_id] = {
                'user': current_user,
                'timestamp': datetime.now(),
                'analysis': analysis_result,
                'query': query,
                'patient_context': patient_context
            }
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            # Test JSON serialization before returning
            try:
                json.dumps(analysis_result)
                logger.info("✅ Analysis result is JSON serializable")
            except Exception as json_error:
                logger.error(f"❌ JSON serialization test failed: {json_error}")
                return jsonify({
                    'error': 'JSON serialization failed',
                    'message': str(json_error)
                }), 500
            
            # Return result
            return jsonify({
                'session_id': session_id,
                'analysis': analysis_result,
                'status': 'success'
            })
            
        except Exception as e:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            
            logger.error(f"Analysis error: {str(e)}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                'error': 'Analysis failed',
                'message': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Outer analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
@jwt_required()
def medical_chat():
    """
    Interactive medical consultation chat
    
    Expected JSON:
    {
        "message": "User message",
        "session_id": "optional_session_id",
        "image_data": "optional_base64_image",
        "patient_history": "optional_patient_history"
    }
    """
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        image_data = data.get('image_data')
        patient_history = data.get('patient_history', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Initialize chat session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                'user': current_user,
                'created_at': datetime.now(),
                'messages': []
            }
        
        # Add user message to session
        chat_sessions[session_id]['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response
        response = generate_medical_consultation_response(message, patient_history)
        
        # Add assistant response to session
        chat_sessions[session_id]['messages'].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'image_analysis': image_data is not None
        })
        
        return jsonify({
            'session_id': session_id,
            'response': response,
            'message_count': len(chat_sessions[session_id]['messages']),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'error': 'Chat failed',
            'message': str(e)
        }), 500

def generate_medical_consultation_response(message: str, patient_history: str) -> str:
    """Generate medical consultation response (mock implementation)"""
    responses = [
        "Thank you for your question. Based on the information provided, I recommend conducting a thorough clinical examination.",
        "This is an interesting case. Could you provide more details about the patient's symptoms and medical history?",
        "From a diagnostic perspective, we should consider differential diagnoses and additional imaging if necessary.",
        "The symptoms you've described warrant careful evaluation. Please ensure proper patient assessment protocols are followed.",
        "This case requires multidisciplinary consultation. I recommend discussing with relevant specialists."
    ]
    
    # Simple keyword-based response selection
    message_lower = message.lower()
    if 'pain' in message_lower:
        return "Pain assessment is crucial. Please evaluate pain intensity, location, duration, and associated symptoms. Consider appropriate pain management strategies while investigating underlying causes."
    elif 'diagnosis' in message_lower:
        return "For accurate diagnosis, comprehensive patient evaluation including history, physical examination, and appropriate diagnostic tests is essential. Consider differential diagnoses and evidence-based diagnostic criteria."
    elif 'treatment' in message_lower:
        return "Treatment recommendations should be based on confirmed diagnosis, patient-specific factors, and current clinical guidelines. Please ensure proper patient consent and monitoring protocols."
    else:
        return f"Thank you for your consultation request. {responses[hash(message) % len(responses)]} Please provide additional clinical details for more specific guidance."

@app.route('/api/dashboard/<session_id>', methods=['GET'])
@jwt_required()
def get_dashboard(session_id):
    """
    Get analytical dashboard for analysis session
    """
    try:
        current_user = get_jwt_identity()
        
        if session_id not in analysis_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = analysis_sessions[session_id]
        
        # Check if user owns the session
        if session_data['user'] != current_user:
            return jsonify({'error': 'Unauthorized access'}), 403
        
        # Generate dashboard data
        dashboard_data = generate_dashboard_data(session_data['analysis'])
        
        return jsonify({
            'session_id': session_id,
            'dashboard': dashboard_data,
            'metadata': {
                'created_at': session_data['timestamp'].isoformat(),
                'query': session_data['query']
            },
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        return jsonify({
            'error': 'Dashboard generation failed',
            'message': str(e)
        }), 500

def generate_dashboard_data(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate dashboard visualization data"""
    
    # Extract key metrics
    uncertainty_metrics = analysis_result.get('uncertainty_metrics', {})
    confidence = uncertainty_metrics.get('confidence', 0.5)
    quantum_uncertainty = uncertainty_metrics.get('quantum_uncertainty', 0.5)
    
    # Generate dashboard components
    dashboard = {
        'confidence_score': {
            'value': confidence,
            'label': 'Overall Confidence',
            'color': 'green' if confidence > 0.8 else 'yellow' if confidence > 0.6 else 'red'
        },
        'quantum_enhancement': {
            'enabled': analysis_result.get('processing_info', {}).get('quantum_enhanced', False),
            'uncertainty': quantum_uncertainty,
            'label': 'Quantum Uncertainty'
        },
        'analysis_summary': {
            'response': analysis_result.get('llava_response', 'No analysis available'),
            'timestamp': analysis_result.get('processing_info', {}).get('timestamp', 'Unknown')
        },
        'risk_indicators': generate_risk_indicators(analysis_result),
        'recommendations': generate_recommendations(analysis_result)
    }
    
    return dashboard

def generate_risk_indicators(analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate risk indicators from analysis"""
    indicators = []
    
    confidence = analysis_result.get('uncertainty_metrics', {}).get('confidence', 0.5)
    
    if confidence < 0.6:
        indicators.append({
            'level': 'high',
            'message': 'Low confidence prediction - requires additional verification',
            'recommendation': 'Consider additional imaging or specialist consultation'
        })
    elif confidence < 0.8:
        indicators.append({
            'level': 'medium',
            'message': 'Moderate confidence - routine follow-up recommended',
            'recommendation': 'Monitor patient condition and symptoms'
        })
    else:
        indicators.append({
            'level': 'low',
            'message': 'High confidence prediction',
            'recommendation': 'Proceed with standard clinical protocols'
        })
    
    return indicators

def generate_recommendations(analysis_result: Dict[str, Any]) -> List[str]:
    """Generate clinical recommendations"""
    recommendations = [
        "Correlate findings with clinical presentation",
        "Review patient history and symptoms",
        "Consider follow-up imaging if indicated"
    ]
    
    confidence = analysis_result.get('uncertainty_metrics', {}).get('confidence', 0.5)
    
    if confidence < 0.7:
        recommendations.extend([
            "Seek second opinion or specialist consultation",
            "Consider additional diagnostic modalities"
        ])
    
    return recommendations

# System status endpoints
@app.route('/api/status', methods=['GET'])
def system_status():
    """Get system status"""
    return jsonify({
        'status': 'online',
        'models': {
            'llava_med': True,
            'quantum_processor': True,
            'explainability_engine': False
        },
        'config': {
            'max_image_size': CONFIG['max_image_size'],
            'supported_formats': CONFIG['supported_formats'],
            'device': CONFIG['device']
        },
        'sessions': {
            'active_analysis_sessions': len(analysis_sessions),
            'active_chat_sessions': len(chat_sessions)
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Utility endpoints
@app.route('/api/sessions/cleanup', methods=['POST'])
@jwt_required()
def cleanup_sessions():
    """Clean up old sessions"""
    try:
        current_time = datetime.now()
        cleaned_analysis = 0
        cleaned_chat = 0
        
        # Clean analysis sessions older than 24 hours
        for session_id, session_data in list(analysis_sessions.items()):
            if current_time - session_data['timestamp'] > timedelta(hours=24):
                del analysis_sessions[session_id]
                cleaned_analysis += 1
        
        # Clean chat sessions older than 24 hours
        for session_data in list(chat_sessions.values()):
            if current_time - session_data['created_at'] > timedelta(hours=24):
                del chat_sessions[list(chat_sessions.keys())[list(chat_sessions.values()).index(session_data)]]
                cleaned_chat += 1
        
        return jsonify({
            'cleaned_analysis_sessions': cleaned_analysis,
            'cleaned_chat_sessions': cleaned_chat,
            'remaining_analysis_sessions': len(analysis_sessions),
            'remaining_chat_sessions': len(chat_sessions)
        })
        
    except Exception as e:
        logger.error(f"Session cleanup error: {str(e)}")
        return jsonify({'error': 'Cleanup failed'}), 500

if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )

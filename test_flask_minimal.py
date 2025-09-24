#!/usr/bin/env python3
"""
Minimal Flask test to isolate the JSON serialization issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
import tempfile
import json
import numpy as np
from quantum_circuits import QuantumMedicalImageProcessor
from llava_medical import LLaVAMedicalModel

# Create minimal Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-secret'
app.config['JWT_SECRET_KEY'] = 'jwt-secret'

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

# Initialize models
llava_model = LLaVAMedicalModel(CONFIG)

@app.route('/test-login', methods=['POST'])
def test_login():
    """Simple login endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if username and password:
        access_token = create_access_token(identity=username)
        return jsonify({
            'access_token': access_token,
            'message': 'Login successful'
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/test-analyze', methods=['POST'])
@jwt_required()
def test_analyze():
    """Test analysis endpoint"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get query
        query = request.form.get('query', 'Please analyze this medical image.')
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            image_file.save(temp_file.name)
            temp_image_path = temp_file.name
        
        try:
            # Analyze image
            print(f"Analyzing image: {temp_image_path}")
            print(f"Query: {query}")
            
            analysis_result = llava_model.analyze_medical_image(
                temp_image_path, query, {}
            )
            
            print(f"Analysis result keys: {analysis_result.keys()}")
            print(f"Analysis result type: {type(analysis_result)}")
            
            # Test JSON serialization
            try:
                json.dumps(analysis_result)
                print("✅ Analysis result is JSON serializable")
            except Exception as json_error:
                print(f"❌ JSON serialization error: {json_error}")
                return jsonify({'error': f'JSON serialization failed: {json_error}'}), 500
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            # Return result
            return jsonify({
                'analysis': analysis_result,
                'status': 'success'
            })
            
        except Exception as e:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise e
            
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting minimal Flask test server...")
    app.run(host='0.0.0.0', port=5001, debug=True)

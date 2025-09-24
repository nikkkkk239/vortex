#!/usr/bin/env python3
"""
Test Flask analysis endpoint directly
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from medical_analyzer import MedicalImageAnalyzer, PatientContext
import json
import numpy as np

def test_flask_analysis():
    """Test the Flask analysis logic directly"""
    print("🧪 Testing Flask Analysis Logic")
    print("=" * 40)
    
    # Create test configuration
    config = {
        'device': 'cpu',
        'max_image_size': 1024,
        'supported_formats': ['jpg', 'jpeg', 'png', 'dicom', 'nii'],
        'quantum_qubits': 4,
        'quantum_layers': 2,
        'model_path': 'microsoft/llava-med-v1.5-mistral-7b',
        'confidence_threshold': 0.7
    }
    
    # Initialize analyzer
    print("🔧 Initializing Medical Image Analyzer...")
    analyzer = MedicalImageAnalyzer(config)
    
    # Test image path
    image_path = "test_medical_image.png"
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return
    
    # Create patient context
    patient_context = PatientContext(
        patient_id="TEST001",
        age=45,
        gender="M",
        medical_history=["diabetes"],
        current_symptoms=["chest pain"]
    )
    
    # Simulate Flask analysis logic
    print("🔍 Simulating Flask analysis...")
    try:
        # This is the same logic as in Flask app
        query = "Analyze this medical image"
        
        # Analyze image
        analysis_result = analyzer.analyze_image(image_path, query, patient_context)
        
        # Convert to dict format like Flask does
        flask_result = {
            'session_id': analysis_result.session_id,
            'analysis': {
                'llava_response': analysis_result.processing_metadata.get('llava_response', ''),
                'quantum_features': analysis_result.quantum_metrics.get('quantum_features', []),
                'uncertainty_metrics': analysis_result.confidence_scores,
                'image_metadata': analysis_result.processing_metadata.get('image_metadata', {}),
                'processing_info': {
                    'quantum_enhanced': analysis_result.quantum_metrics.get('quantum_enhanced', False),
                    'model_used': config['model_path'],
                    'timestamp': str(analysis_result.timestamp)
                }
            },
            'status': 'success'
        }
        
        print("✅ Flask analysis completed successfully!")
        
        # Test JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            else:
                return obj
        
        serializable_result = convert_numpy(flask_result)
        
        try:
            json_str = json.dumps(serializable_result)
            print("✅ JSON serialization test passed")
            print(f"📊 Result keys: {list(serializable_result.keys())}")
            print(f"📈 Analysis keys: {list(serializable_result['analysis'].keys())}")
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
            print(f"🔍 Error details: {type(e).__name__}: {e}")
        
    except Exception as e:
        print(f"❌ Flask analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flask_analysis()

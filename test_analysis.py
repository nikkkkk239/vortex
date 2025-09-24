#!/usr/bin/env python3
"""
Test script to debug the analysis endpoint issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from quantum_circuits import QuantumMedicalImageProcessor
from llava_medical import LLaVAMedicalModel, MedicalImagePreprocessor
import numpy as np
import json

def test_analysis():
    """Test the analysis pipeline"""
    print("Testing analysis pipeline...")
    
    # Test configuration
    config = {
        'model_path': 'microsoft/llava-med-v1.5-mistral-7b',
        'max_image_size': 1024,
        'supported_formats': ['jpg', 'jpeg', 'png', 'dicom', 'nii'],
        'confidence_threshold': 0.7,
        'quantum_qubits': 8,
        'quantum_layers': 2,
        'device': 'cpu'
    }
    
    try:
        # Test quantum processor
        print("1. Testing quantum processor...")
        quantum_processor = QuantumMedicalImageProcessor(config)
        
        # Create test features
        test_features = np.random.rand(256)
        print(f"   Test features shape: {test_features.shape}")
        
        # Process with quantum processor
        quantum_result = quantum_processor.process_medical_image(test_features)
        print(f"   Quantum result keys: {quantum_result.keys()}")
        
        # Test JSON serialization
        print("2. Testing JSON serialization...")
        try:
            json_str = json.dumps(quantum_result)
            print("   ✅ Quantum result is JSON serializable")
        except Exception as e:
            print(f"   ❌ Quantum result JSON error: {e}")
            return False
        
        # Test LLaVA model
        print("3. Testing LLaVA model...")
        llava_model = LLaVAMedicalModel(config)
        
        # Test analysis
        print("4. Testing full analysis...")
        analysis_result = llava_model.analyze_medical_image(
            'test_medical_image.png', 
            'Test query',
            {}
        )
        print(f"   Analysis result keys: {analysis_result.keys()}")
        
        # Test JSON serialization of full result
        try:
            json_str = json.dumps(analysis_result)
            print("   ✅ Full analysis result is JSON serializable")
            return True
        except Exception as e:
            print(f"   ❌ Full analysis result JSON error: {e}")
            
            # Debug the problematic object
            def debug_object(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        debug_object(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        debug_object(item, f"{path}[{i}]")
                elif isinstance(obj, np.ndarray):
                    print(f"   Found numpy array at {path}: shape={obj.shape}, dtype={obj.dtype}")
                elif hasattr(obj, '__dict__'):
                    print(f"   Found object at {path}: {type(obj)}")
            
            debug_object(analysis_result)
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_analysis()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")

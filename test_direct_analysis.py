#!/usr/bin/env python3
"""
Direct test of the analysis pipeline to isolate the JSON serialization issue
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from quantum_circuits import QuantumMedicalImageProcessor
from llava_medical import LLaVAMedicalModel, MedicalImagePreprocessor
import numpy as np
import json
import tempfile
from PIL import Image

def test_direct_analysis():
    """Test the analysis pipeline directly"""
    print("Testing direct analysis pipeline...")
    
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
        # Create a test image file
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            img.save(temp_file.name)
            temp_image_path = temp_file.name
        
        print(f"Created test image: {temp_image_path}")
        
        # Initialize models
        print("1. Initializing LLaVA model...")
        llava_model = LLaVAMedicalModel(config)
        
        print("2. Running analysis...")
        analysis_result = llava_model.analyze_medical_image(
            temp_image_path, 
            'Please analyze this medical image for any abnormalities',
            {}
        )
        
        print(f"3. Analysis result keys: {analysis_result.keys()}")
        
        # Test JSON serialization
        print("4. Testing JSON serialization...")
        try:
            json_str = json.dumps(analysis_result)
            print("   ✅ Analysis result is JSON serializable")
            print(f"   JSON length: {len(json_str)} characters")
            return True
        except Exception as e:
            print(f"   ❌ JSON serialization error: {e}")
            
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
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    pass  # These are fine
                else:
                    print(f"   Found unknown type at {path}: {type(obj)}")
            
            debug_object(analysis_result)
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

if __name__ == "__main__":
    success = test_direct_analysis()
    if success:
        print("\n✅ Direct analysis test passed!")
    else:
        print("\n❌ Direct analysis test failed!")

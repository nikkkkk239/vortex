#!/usr/bin/env python3
"""
Test script for medical image analysis
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from medical_analyzer import MedicalImageAnalyzer, PatientContext

def test_analysis():
    """Test the medical image analysis"""
    print("🧪 Testing Medical Image Analysis")
    print("=" * 40)
    
    # Create test configuration
    config = {
        'device': 'cpu',
        'max_image_size': 1024,
        'supported_formats': ['jpg', 'jpeg', 'png', 'dicom', 'nii'],
        'quantum_qubits': 4,
        'quantum_layers': 2
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
    
    # Analyze image
    print("🔍 Analyzing medical image...")
    try:
        result = analyzer.analyze_image(image_path, "Analyze this medical image", patient_context)
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Session ID: {result.session_id}")
        print(f"🏥 Imaging Modality: {result.imaging_modality.value}")
        print(f"📈 Confidence Scores: {result.confidence_scores}")
        print(f"🔬 Primary Findings: {len(result.primary_findings)} findings")
        
        for i, finding in enumerate(result.primary_findings):
            print(f"  {i+1}. {finding.description} (Confidence: {finding.confidence:.2f})")
        
        print(f"⚛️ Quantum Enhanced: {result.quantum_metrics.get('quantum_enhanced', False)}")
        print(f"⏱️ Processing Time: {result.processing_metadata.get('processing_time', 0):.2f}s")
        
        # Test JSON serialization
        import json
        try:
            json_str = json.dumps({
                'session_id': result.session_id,
                'confidence_scores': result.confidence_scores,
                'quantum_enhanced': result.quantum_metrics.get('quantum_enhanced', False)
            })
            print("✅ JSON serialization test passed")
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analysis()
"""
Medical Image Analyzer - Core Analysis Engine
Quantum-enhanced medical image processing with LLaVA-Med integration
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import cv2
from PIL import Image
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quantum_circuits import QuantumMedicalImageProcessor, HybridQuantumClassicalNetwork
from llava_medical import LLaVAMedicalModel, MedicalImagePreprocessor, MedicalTerminologyProcessor
from explainable_ai import MedicalExplainabilityEngine

logger = logging.getLogger(__name__)


class DiagnosisConfidence(Enum):
    """Enumeration for diagnosis confidence levels"""
    HIGH = "high"
    MODERATE = "moderate" 
    LOW = "low"
    UNCERTAIN = "uncertain"


class ImagingModality(Enum):
    """Enumeration for medical imaging modalities"""
    XRAY = "x-ray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    PET = "pet"
    FLUOROSCOPY = "fluoroscopy"
    UNKNOWN = "unknown"


@dataclass
class PatientContext:
    """Patient context information"""
    patient_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[List[str]] = None
    current_symptoms: Optional[List[str]] = None
    medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    previous_imaging: Optional[List[Dict[str, Any]]] = None


@dataclass
class DiagnosticFinding:
    """Represents a diagnostic finding"""
    finding_id: str
    description: str
    confidence: float
    location: Optional[Dict[str, Any]] = None
    severity: Optional[str] = None
    recommendations: Optional[List[str]] = None
    differential_diagnoses: Optional[List[str]] = None


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    session_id: str
    patient_context: PatientContext
    imaging_modality: ImagingModality
    primary_findings: List[DiagnosticFinding]
    confidence_scores: Dict[str, float]
    quantum_metrics: Dict[str, Any]
    explainability_data: Dict[str, Any]
    recommendations: List[str]
    risk_stratification: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    timestamp: datetime


class MedicalImageAnalyzer:
    """
    Core medical image analyzer with quantum enhancement and explainable AI
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Medical Image Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Initialize components
        self._initialize_components()
        
        # Load medical knowledge base
        self.medical_knowledge = self._load_medical_knowledge()
        
        # Initialize diagnostic rules
        self.diagnostic_rules = self._load_diagnostic_rules()
        
        logger.info("Initialized MedicalImageAnalyzer")
    
    def _initialize_components(self):
        """Initialize all analyzer components"""
        try:
            # Initialize LLaVA-Med model
            self.llava_model = LLaVAMedicalModel(self.config)
            
            # Initialize quantum processor
            self.quantum_processor = QuantumMedicalImageProcessor(self.config)
            
            # Initialize image preprocessor
            self.image_preprocessor = MedicalImagePreprocessor(self.config)
            
            # Initialize terminology processor
            self.terminology_processor = MedicalTerminologyProcessor()
            
            # Initialize explainability engine (mock for now)
            self.explainability_engine = None
            
            # Initialize hybrid quantum-classical network
            if self.config.get('use_hybrid_network', False):
                self.hybrid_network = HybridQuantumClassicalNetwork(
                    classical_input_size=self.config.get('feature_size', 2048),
                    quantum_qubits=self.config.get('quantum_qubits', 8),
                    num_classes=self.config.get('num_classes', 20)
                )
            else:
                self.hybrid_network = None
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            # Initialize with minimal functionality
            self.llava_model = None
            self.quantum_processor = None
            self.image_preprocessor = MedicalImagePreprocessor(self.config)
            self.terminology_processor = MedicalTerminologyProcessor()
            self.explainability_engine = None
            self.hybrid_network = None
    
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load medical knowledge base"""
        # In production, this would load from a comprehensive medical database
        return {
            'common_findings': {
                'pneumonia': {
                    'description': 'Infection of the lungs',
                    'typical_locations': ['lung_fields', 'lower_lobes'],
                    'modalities': ['x-ray', 'ct'],
                    'severity_levels': ['mild', 'moderate', 'severe']
                },
                'fracture': {
                    'description': 'Break in bone continuity',
                    'typical_locations': ['extremities', 'spine', 'ribs'],
                    'modalities': ['x-ray', 'ct'],
                    'severity_levels': ['hairline', 'simple', 'complex', 'compound']
                },
                'tumor': {
                    'description': 'Abnormal tissue growth',
                    'typical_locations': ['brain', 'lung', 'abdomen', 'bone'],
                    'modalities': ['ct', 'mri', 'pet'],
                    'severity_levels': ['benign', 'malignant']
                }
            },
            'anatomical_regions': {
                'thorax': ['heart', 'lungs', 'mediastinum', 'pleura'],
                'abdomen': ['liver', 'kidneys', 'spleen', 'pancreas', 'bowel'],
                'head': ['brain', 'skull', 'sinuses', 'orbits'],
                'spine': ['cervical', 'thoracic', 'lumbar', 'sacral'],
                'extremities': ['arms', 'legs', 'joints', 'bones']
            }
        }
    
    def _load_diagnostic_rules(self) -> Dict[str, Any]:
        """Load diagnostic decision rules"""
        return {
            'confidence_thresholds': {
                'high_confidence': 0.8,
                'moderate_confidence': 0.6,
                'low_confidence': 0.4
            },
            'modality_specific_rules': {
                'x-ray': {
                    'optimal_conditions': ['proper_positioning', 'adequate_exposure'],
                    'common_artifacts': ['motion_blur', 'overlapping_structures']
                },
                'ct': {
                    'optimal_conditions': ['contrast_timing', 'slice_thickness'],
                    'common_artifacts': ['beam_hardening', 'metal_artifacts']
                },
                'mri': {
                    'optimal_conditions': ['sequence_selection', 'field_strength'],
                    'common_artifacts': ['motion', 'susceptibility', 'chemical_shift']
                }
            }
        }
    
    def analyze_image(self, image_path: str, query: str,
                     patient_context: Optional[PatientContext] = None) -> AnalysisResult:
        """
        Comprehensive medical image analysis
        
        Args:
            image_path: Path to medical image
            query: Clinical query
            patient_context: Patient context information
            
        Returns:
            Complete analysis result
        """
        try:
            session_id = self._generate_session_id()
            start_time = datetime.now()
            
            logger.info(f"Starting analysis for session {session_id}")
            
            # Step 1: Preprocess image and detect modality
            image, metadata = self.image_preprocessor.preprocess_image(image_path)
            modality = self._detect_imaging_modality(image, metadata)
            
            # Step 2: Extract classical features
            classical_features = self._extract_classical_features(image, metadata)
            
            # Step 3: Apply quantum enhancement
            quantum_results = self._apply_quantum_enhancement(classical_features, patient_context)
            
            # Step 4: Generate LLaVA-Med analysis
            llava_analysis = self._generate_llava_analysis(image_path, query, patient_context)
            
            # Step 5: Apply hybrid quantum-classical network if available
            hybrid_predictions = self._apply_hybrid_network(classical_features)
            
            # Step 6: Generate explainable AI insights
            explainability_data = self._generate_explainability(image_path, classical_features)
            
            # Step 7: Extract diagnostic findings
            findings = self._extract_diagnostic_findings(
                llava_analysis, quantum_results, hybrid_predictions, metadata
            )
            
            # Step 8: Calculate confidence scores
            confidence_scores = self._calculate_comprehensive_confidence(
                quantum_results, llava_analysis, hybrid_predictions
            )
            
            # Step 9: Generate recommendations
            recommendations = self._generate_clinical_recommendations(
                findings, confidence_scores, patient_context, modality
            )
            
            # Step 10: Perform risk stratification
            risk_stratification = self._perform_risk_stratification(
                findings, patient_context, confidence_scores
            )
            
            # Compile results
            analysis_result = AnalysisResult(
                session_id=session_id,
                patient_context=patient_context or PatientContext(),
                imaging_modality=modality,
                primary_findings=findings,
                confidence_scores=confidence_scores,
                quantum_metrics=quantum_results,
                explainability_data=explainability_data,
                recommendations=recommendations,
                risk_stratification=risk_stratification,
                processing_metadata={
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'image_metadata': metadata,
                    'llava_response': llava_analysis.get('llava_response', ''),
                    'quantum_enhanced': quantum_results.get('quantum_enhanced', False),
                    'hybrid_network_used': hybrid_predictions is not None
                },
                timestamp=datetime.now()
            )
            
            logger.info(f"Analysis completed for session {session_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            # Return error result
            return self._create_error_result(str(e), patient_context)
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _detect_imaging_modality(self, image: Image.Image, metadata: Dict[str, Any]) -> ImagingModality:
        """Detect imaging modality from image and metadata"""
        # Check metadata first
        if 'modality' in metadata:
            modality_str = metadata['modality'].lower()
            for modality in ImagingModality:
                if modality.value in modality_str:
                    return modality
        
        # Fallback to image analysis
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Simple heuristics for modality detection
        mean_intensity = np.mean(img_array)
        intensity_std = np.std(img_array)
        
        if mean_intensity < 50 and intensity_std > 30:
            return ImagingModality.CT
        elif mean_intensity > 150 and intensity_std < 50:
            return ImagingModality.XRAY
        else:
            return ImagingModality.UNKNOWN
    
    def _extract_classical_features(self, image: Image.Image, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract classical image features"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Basic image features
            features = []
            
            # Intensity statistics
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.median(gray),
                np.min(gray),
                np.max(gray)
            ])
            
            # Texture features (using Local Binary Patterns approximation)
            texture_features = self._extract_texture_features(gray)
            features.extend(texture_features)
            
            # Shape features
            shape_features = self._extract_shape_features(gray)
            features.extend(shape_features)
            
            # Frequency domain features
            freq_features = self._extract_frequency_features(gray)
            features.extend(freq_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting classical features: {str(e)}")
            return np.zeros(256, dtype=np.float32)  # Return zeros as fallback
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract texture features from grayscale image"""
        # Simple texture measures
        features = []
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.sum(magnitude > np.mean(magnitude)) / magnitude.size  # Edge density
        ])
        
        return features
    
    def _extract_shape_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract shape-based features"""
        features = []
        
        # Find contours for shape analysis
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour features
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape descriptors
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                features.append(circularity)
            else:
                features.append(0.0)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            features.append(aspect_ratio)
            
        else:
            features.extend([0.0, 0.0])  # No contours found
        
        return features
    
    def _extract_frequency_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract frequency domain features using FFT"""
        features = []
        
        # Apply FFT
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Frequency domain statistics
        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.sum(magnitude_spectrum > np.mean(magnitude_spectrum)) / magnitude_spectrum.size
        ])
        
        return features
    
    def _apply_quantum_enhancement(self, classical_features: np.ndarray,
                                  patient_context: Optional[PatientContext]) -> Dict[str, Any]:
        """Apply quantum enhancement to features"""
        if self.quantum_processor:
            context_dict = {}
            if patient_context:
                context_dict = {
                    'age': patient_context.age,
                    'medical_history': patient_context.medical_history or [],
                    'symptoms': patient_context.current_symptoms or []
                }
            
            return self.quantum_processor.process_medical_image(classical_features, context_dict)
        else:
            # Mock quantum results
            return {
                'quantum_features': classical_features,
                'uncertainty_metrics': {
                    'confidence': 0.75,
                    'entropy': 0.5,
                    'quantum_uncertainty': 0.25
                },
                'quantum_enhanced': False
            }
    
    def _generate_llava_analysis(self, image_path: str, query: str,
                                patient_context: Optional[PatientContext]) -> Dict[str, Any]:
        """Generate analysis using LLaVA-Med"""
        if self.llava_model:
            context_dict = {}
            if patient_context:
                context_dict = {
                    'patient_id': patient_context.patient_id,
                    'age': patient_context.age,
                    'gender': patient_context.gender,
                    'medical_history': patient_context.medical_history or [],
                    'symptoms': patient_context.current_symptoms or []
                }
            
            return self.llava_model.analyze_medical_image(image_path, query, context_dict)
        else:
            # Mock LLaVA response
            return {
                'llava_response': f"Mock medical analysis: {query}. Please correlate with clinical findings.",
                'processing_info': {'quantum_enhanced': False}
            }
    
    def _apply_hybrid_network(self, classical_features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Apply hybrid quantum-classical network if available"""
        if self.hybrid_network is None:
            return None
        
        try:
            # Convert to tensor
            features_tensor = torch.tensor(classical_features).unsqueeze(0).float()
            
            # Forward pass
            with torch.no_grad():
                predictions = self.hybrid_network(features_tensor)
                uncertainty = self.hybrid_network.get_quantum_uncertainty(features_tensor)
            
            return {
                'predictions': predictions.numpy(),
                'uncertainty': {k: v.numpy() for k, v in uncertainty.items()}
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid network: {str(e)}")
            return None
    
    def _generate_explainability(self, image_path: str, features: np.ndarray) -> Dict[str, Any]:
        """Generate explainability data"""
        if self.explainability_engine:
            # Would generate comprehensive explanations
            pass
        
        # Mock explainability data
        return {
            'feature_importance': np.random.rand(len(features)).tolist(),
            'attention_regions': [
                {'x': 100, 'y': 150, 'width': 50, 'height': 60, 'confidence': 0.8},
                {'x': 200, 'y': 250, 'width': 40, 'height': 45, 'confidence': 0.6}
            ],
            'explanation_available': False
        }
    
    def _extract_diagnostic_findings(self, llava_analysis: Dict[str, Any],
                                   quantum_results: Dict[str, Any],
                                   hybrid_predictions: Optional[Dict[str, Any]],
                                   metadata: Dict[str, Any]) -> List[DiagnosticFinding]:
        """Extract diagnostic findings from analysis results"""
        findings = []
        
        # Extract from LLaVA response
        llava_response = llava_analysis.get('llava_response', '')
        
        # Simple keyword-based finding extraction
        finding_keywords = {
            'pneumonia': ['pneumonia', 'consolidation', 'infiltrate', 'opacity'],
            'fracture': ['fracture', 'break', 'crack', 'displacement'],
            'mass': ['mass', 'lesion', 'tumor', 'nodule'],
            'normal': ['normal', 'unremarkable', 'no acute', 'clear']
        }
        
        response_lower = llava_response.lower()
        
        for finding_type, keywords in finding_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    confidence = quantum_results.get('uncertainty_metrics', {}).get('confidence', 0.5)
                    
                    finding = DiagnosticFinding(
                        finding_id=f"{finding_type}_{len(findings)}",
                        description=f"Possible {finding_type} detected based on {keyword}",
                        confidence=confidence,
                        recommendations=self._get_finding_recommendations(finding_type),
                        differential_diagnoses=self._get_differential_diagnoses(finding_type)
                    )
                    findings.append(finding)
                    break  # Only add one finding per type
        
        # If no findings, add normal finding
        if not findings:
            findings.append(DiagnosticFinding(
                finding_id="normal_0",
                description="No significant pathological findings identified",
                confidence=quantum_results.get('uncertainty_metrics', {}).get('confidence', 0.7),
                recommendations=["Continue routine care", "Clinical correlation recommended"]
            ))
        
        return findings
    
    def _get_finding_recommendations(self, finding_type: str) -> List[str]:
        """Get recommendations for specific finding type"""
        recommendations_map = {
            'pneumonia': [
                "Consider antibiotic therapy",
                "Monitor oxygen saturation",
                "Follow-up chest imaging",
                "Assess for complications"
            ],
            'fracture': [
                "Orthopedic consultation",
                "Immobilization as appropriate",
                "Pain management",
                "Follow-up imaging"
            ],
            'mass': [
                "Further characterization needed",
                "Consider CT or MRI",
                "Oncology consultation if malignant",
                "Biopsy may be indicated"
            ],
            'normal': [
                "Continue routine care",
                "Clinical correlation",
                "Return if symptoms worsen"
            ]
        }
        
        return recommendations_map.get(finding_type, ["Clinical correlation recommended"])
    
    def _get_differential_diagnoses(self, finding_type: str) -> List[str]:
        """Get differential diagnoses for finding type"""
        differentials_map = {
            'pneumonia': ["Viral pneumonia", "Bacterial pneumonia", "Atypical pneumonia", "Pulmonary edema"],
            'fracture': ["Simple fracture", "Comminuted fracture", "Pathological fracture", "Stress fracture"],
            'mass': ["Benign mass", "Primary malignancy", "Metastatic disease", "Inflammatory lesion"],
            'normal': ["Normal variant", "Early pathology", "Technical factors"]
        }
        
        return differentials_map.get(finding_type, ["Further evaluation needed"])
    
    def _calculate_comprehensive_confidence(self, quantum_results: Dict[str, Any],
                                          llava_analysis: Dict[str, Any],
                                          hybrid_predictions: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive confidence scores"""
        # Extract individual confidence scores
        quantum_confidence = quantum_results.get('uncertainty_metrics', {}).get('confidence', 0.5)
        
        # LLaVA confidence (estimated from response length and certainty words)
        llava_response = llava_analysis.get('llava_response', '')
        llava_confidence = self._estimate_llava_confidence(llava_response)
        
        # Hybrid network confidence
        hybrid_confidence = 0.5
        if hybrid_predictions:
            uncertainty_data = hybrid_predictions.get('uncertainty', {})
            hybrid_confidence = uncertainty_data.get('confidence', [0.5])[0]
        
        # Calculate overall confidence
        confidences = [quantum_confidence, llava_confidence, hybrid_confidence]
        weights = [0.4, 0.4, 0.2]  # Weight quantum and LLaVA more heavily
        
        overall_confidence = sum(c * w for c, w in zip(confidences, weights))
        
        return {
            'overall_confidence': overall_confidence,
            'quantum_confidence': quantum_confidence,
            'llava_confidence': llava_confidence,
            'hybrid_confidence': hybrid_confidence,
            'confidence_level': self._categorize_confidence(overall_confidence)
        }
    
    def _estimate_llava_confidence(self, response: str) -> float:
        """Estimate confidence from LLaVA response text"""
        if not response:
            return 0.5
        
        # Count certainty indicators
        high_confidence_words = ['clearly', 'definitely', 'obvious', 'evident', 'certain']
        low_confidence_words = ['possibly', 'maybe', 'might', 'could', 'uncertain', 'unclear']
        
        response_lower = response.lower()
        
        high_count = sum(1 for word in high_confidence_words if word in response_lower)
        low_count = sum(1 for word in low_confidence_words if word in response_lower)
        
        # Base confidence
        base_confidence = 0.6
        
        # Adjust based on word counts
        if high_count > low_count:
            return min(0.95, base_confidence + 0.1 * high_count)
        elif low_count > high_count:
            return max(0.2, base_confidence - 0.1 * low_count)
        else:
            return base_confidence
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= 0.8:
            return DiagnosisConfidence.HIGH.value
        elif confidence >= 0.6:
            return DiagnosisConfidence.MODERATE.value
        elif confidence >= 0.4:
            return DiagnosisConfidence.LOW.value
        else:
            return DiagnosisConfidence.UNCERTAIN.value
    
    def _generate_clinical_recommendations(self, findings: List[DiagnosticFinding],
                                         confidence_scores: Dict[str, float],
                                         patient_context: Optional[PatientContext],
                                         modality: ImagingModality) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        # Base recommendations
        recommendations.append("Correlate findings with clinical presentation")
        
        # Confidence-based recommendations
        overall_confidence = confidence_scores.get('overall_confidence', 0.5)
        
        if overall_confidence < 0.6:
            recommendations.extend([
                "Consider additional imaging modalities",
                "Seek specialist consultation",
                "Clinical correlation strongly recommended"
            ])
        elif overall_confidence < 0.8:
            recommendations.append("Follow-up imaging may be beneficial")
        
        # Finding-specific recommendations
        for finding in findings:
            if finding.recommendations:
                recommendations.extend(finding.recommendations)
        
        # Patient context-based recommendations
        if patient_context:
            if patient_context.age and patient_context.age > 65:
                recommendations.append("Consider age-related factors in interpretation")
            
            if patient_context.medical_history:
                recommendations.append("Consider past medical history in diagnostic workup")
        
        # Modality-specific recommendations
        if modality == ImagingModality.XRAY:
            recommendations.append("Consider CT if further detail needed")
        elif modality == ImagingModality.CT:
            recommendations.append("Consider MRI for soft tissue evaluation")
        
        # Remove duplicates and sort
        unique_recommendations = list(set(recommendations))
        return sorted(unique_recommendations)
    
    def _perform_risk_stratification(self, findings: List[DiagnosticFinding],
                                   patient_context: Optional[PatientContext],
                                   confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Perform risk stratification"""
        risk_factors = []
        risk_score = 0.0
        
        # Finding-based risk
        for finding in findings:
            if 'tumor' in finding.description.lower() or 'mass' in finding.description.lower():
                risk_score += 0.3
                risk_factors.append("Possible neoplastic process")
            elif 'fracture' in finding.description.lower():
                risk_score += 0.2
                risk_factors.append("Traumatic injury")
            elif 'pneumonia' in finding.description.lower():
                risk_score += 0.25
                risk_factors.append("Infectious process")
        
        # Confidence-based risk
        if confidence_scores.get('overall_confidence', 0.5) < 0.6:
            risk_score += 0.1
            risk_factors.append("Diagnostic uncertainty")
        
        # Patient context-based risk
        if patient_context:
            if patient_context.age and patient_context.age > 70:
                risk_score += 0.1
                risk_factors.append("Advanced age")
            
            if patient_context.medical_history:
                high_risk_conditions = ['cancer', 'diabetes', 'heart disease', 'copd']
                for condition in high_risk_conditions:
                    if any(condition in (history or '').lower() for history in patient_context.medical_history):
                        risk_score += 0.05
                        risk_factors.append(f"History of {condition}")
        
        # Categorize risk
        if risk_score >= 0.5:
            risk_category = "High"
        elif risk_score >= 0.3:
            risk_category = "Moderate"
        else:
            risk_category = "Low"
        
        return {
            'risk_score': min(risk_score, 1.0),  # Cap at 1.0
            'risk_category': risk_category,
            'risk_factors': risk_factors,
            'recommendations': self._get_risk_based_recommendations(risk_category)
        }
    
    def _get_risk_based_recommendations(self, risk_category: str) -> List[str]:
        """Get recommendations based on risk category"""
        if risk_category == "High":
            return [
                "Urgent clinical evaluation recommended",
                "Consider immediate specialist consultation",
                "Close monitoring required",
                "Expedited follow-up imaging"
            ]
        elif risk_category == "Moderate":
            return [
                "Timely clinical correlation recommended",
                "Consider specialist consultation",
                "Schedule appropriate follow-up",
                "Monitor for symptom progression"
            ]
        else:
            return [
                "Routine clinical follow-up",
                "Standard monitoring protocols",
                "Patient education regarding symptoms"
            ]
    
    def _create_error_result(self, error_message: str, 
                           patient_context: Optional[PatientContext]) -> AnalysisResult:
        """Create error analysis result"""
        return AnalysisResult(
            session_id=self._generate_session_id(),
            patient_context=patient_context or PatientContext(),
            imaging_modality=ImagingModality.UNKNOWN,
            primary_findings=[
                DiagnosticFinding(
                    finding_id="error_0",
                    description=f"Analysis error: {error_message}",
                    confidence=0.0
                )
            ],
            confidence_scores={'overall_confidence': 0.0},
            quantum_metrics={'error': error_message},
            explainability_data={'error': error_message},
            recommendations=["Please retry analysis", "Check image quality and format"],
            risk_stratification={'risk_category': 'Unknown', 'risk_score': 0.0},
            processing_metadata={'error': error_message},
            timestamp=datetime.now()
        )
    
    def batch_analyze_images(self, image_paths: List[str], queries: List[str],
                           patient_contexts: Optional[List[PatientContext]] = None) -> List[AnalysisResult]:
        """
        Batch analyze multiple medical images
        
        Args:
            image_paths: List of image paths
            queries: List of queries for each image
            patient_contexts: Optional list of patient contexts
            
        Returns:
            List of analysis results
        """
        if patient_contexts is None:
            patient_contexts = [None] * len(image_paths)
        
        results = []
        for i, (image_path, query) in enumerate(zip(image_paths, queries)):
            context = patient_contexts[i] if i < len(patient_contexts) else None
            result = self.analyze_image(image_path, query, context)
            results.append(result)
        
        return results
    
    def export_analysis_results(self, results: List[AnalysisResult], output_path: str) -> str:
        """
        Export analysis results to various formats
        
        Args:
            results: List of analysis results
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        try:
            output_path = Path(output_path)
            
            if output_path.suffix.lower() == '.json':
                return self._export_to_json(results, output_path)
            elif output_path.suffix.lower() == '.csv':
                return self._export_to_csv(results, output_path)
            else:
                # Default to JSON
                json_path = output_path.with_suffix('.json')
                return self._export_to_json(results, json_path)
                
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return ""
    
    def _export_to_json(self, results: List[AnalysisResult], output_path: Path) -> str:
        """Export results to JSON format"""
        export_data = []
        
        for result in results:
            result_dict = {
                'session_id': result.session_id,
                'timestamp': result.timestamp.isoformat(),
                'imaging_modality': result.imaging_modality.value,
                'confidence_scores': result.confidence_scores,
                'primary_findings': [
                    {
                        'finding_id': f.finding_id,
                        'description': f.description,
                        'confidence': f.confidence,
                        'recommendations': f.recommendations or []
                    }
                    for f in result.primary_findings
                ],
                'recommendations': result.recommendations,
                'risk_stratification': result.risk_stratification,
                'quantum_enhanced': result.quantum_metrics.get('quantum_enhanced', False)
            }
            export_data.append(result_dict)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(output_path)
    
    def _export_to_csv(self, results: List[AnalysisResult], output_path: Path) -> str:
        """Export results to CSV format"""
        rows = []
        
        for result in results:
            for finding in result.primary_findings:
                row = {
                    'session_id': result.session_id,
                    'timestamp': result.timestamp.isoformat(),
                    'imaging_modality': result.imaging_modality.value,
                    'finding_id': finding.finding_id,
                    'finding_description': finding.description,
                    'finding_confidence': finding.confidence,
                    'overall_confidence': result.confidence_scores.get('overall_confidence', 0.0),
                    'risk_category': result.risk_stratification.get('risk_category', 'Unknown'),
                    'quantum_enhanced': result.quantum_metrics.get('quantum_enhanced', False)
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return str(output_path)
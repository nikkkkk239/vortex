"""
LLaVA-Med Integration Module
Integrates Microsoft LLaVA-Med with quantum enhancements and medical imaging pipeline
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from PIL import Image
import cv2
from pathlib import Path
import json
import warnings

# Add the parent llava directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    LLAVA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LLaVA imports not available: {e}")
    LLAVA_AVAILABLE = False
    # Create placeholder classes for development
    class MockLLaVAModel:
        def __init__(self): pass
        def generate(self, *args, **kwargs): return "Mock LLaVA response"

from quantum_circuits import QuantumMedicalImageProcessor

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Preprocessor for medical images compatible with LLaVA-Med
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize medical image preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_image_size = config.get('max_image_size', 1024)
        self.supported_formats = config.get('supported_formats', ['jpg', 'jpeg', 'png', 'dicom', 'nii'])
        
        logger.info("Initialized MedicalImagePreprocessor")
    
    def preprocess_image(self, image_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Preprocess medical image for LLaVA-Med analysis
        
        Args:
            image_path: Path to the medical image
            
        Returns:
            Tuple of (preprocessed_image, metadata)
        """
        try:
            # Determine image type and load accordingly
            image_path = Path(image_path)
            extension = image_path.suffix.lower().lstrip('.')
            
            if extension in ['jpg', 'jpeg', 'png']:
                image, metadata = self._load_standard_image(image_path)
            elif extension == 'dicom':
                image, metadata = self._load_dicom_image(image_path)
            elif extension in ['nii', 'nii.gz']:
                image, metadata = self._load_nifti_image(image_path)
            else:
                raise ValueError(f"Unsupported image format: {extension}")
            
            # Apply medical image enhancements
            enhanced_image = self._enhance_medical_image(image, metadata)
            
            # Resize if necessary
            if max(enhanced_image.size) > self.max_image_size:
                enhanced_image = self._resize_image(enhanced_image, self.max_image_size)
            
            return enhanced_image, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def _load_standard_image(self, image_path: Path) -> Tuple[Image.Image, Dict[str, Any]]:
        """Load standard image formats (JPG, PNG)"""
        image = Image.open(image_path).convert('RGB')
        
        metadata = {
            'format': image_path.suffix.lower().lstrip('.'),
            'size': image.size,
            'mode': image.mode,
            'source_path': str(image_path)
        }
        
        return image, metadata
    
    def _load_dicom_image(self, image_path: Path) -> Tuple[Image.Image, Dict[str, Any]]:
        """Load DICOM medical images"""
        try:
            import pydicom
            
            # Read DICOM file
            dicom_data = pydicom.dcmread(str(image_path))
            
            # Extract pixel array
            pixel_array = dicom_data.pixel_array
            
            # Normalize to 0-255 range
            pixel_array = pixel_array.astype(np.float32)
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
            pixel_array = pixel_array.astype(np.uint8)
            
            # Convert to RGB PIL Image
            if len(pixel_array.shape) == 2:
                # Grayscale to RGB
                image = Image.fromarray(pixel_array, mode='L').convert('RGB')
            else:
                image = Image.fromarray(pixel_array)
            
            # Extract DICOM metadata
            metadata = {
                'format': 'dicom',
                'size': image.size,
                'modality': getattr(dicom_data, 'Modality', 'Unknown'),
                'patient_id': getattr(dicom_data, 'PatientID', 'Unknown'),
                'study_date': getattr(dicom_data, 'StudyDate', 'Unknown'),
                'body_part': getattr(dicom_data, 'BodyPartExamined', 'Unknown'),
                'source_path': str(image_path)
            }
            
            return image, metadata
            
        except ImportError:
            logger.error("pydicom not available for DICOM processing")
            raise
        except Exception as e:
            logger.error(f"Error loading DICOM image: {str(e)}")
            raise
    
    def _load_nifti_image(self, image_path: Path) -> Tuple[Image.Image, Dict[str, Any]]:
        """Load NIfTI medical images"""
        try:
            import nibabel as nib
            
            # Load NIfTI file
            nii_img = nib.load(str(image_path))
            data = nii_img.get_fdata()
            
            # Take middle slice for 3D volumes
            if len(data.shape) == 3:
                middle_slice = data.shape[2] // 2
                slice_data = data[:, :, middle_slice]
            else:
                slice_data = data
            
            # Normalize to 0-255 range
            slice_data = slice_data.astype(np.float32)
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            slice_data = slice_data.astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(slice_data, mode='L').convert('RGB')
            
            # Extract NIfTI metadata
            header = nii_img.header
            metadata = {
                'format': 'nifti',
                'size': image.size,
                'original_shape': data.shape,
                'voxel_size': header.get_zooms(),
                'source_path': str(image_path)
            }
            
            return image, metadata
            
        except ImportError:
            logger.error("nibabel not available for NIfTI processing")
            raise
        except Exception as e:
            logger.error(f"Error loading NIfTI image: {str(e)}")
            raise
    
    def _enhance_medical_image(self, image: Image.Image, metadata: Dict[str, Any]) -> Image.Image:
        """Apply medical image enhancements"""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(img_array.shape) == 3:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_array = clahe.apply(img_array)
        
        # Apply denoising for medical images
        if metadata.get('format') in ['dicom', 'nifti']:
            enhanced_array = cv2.bilateralFilter(enhanced_array, 9, 75, 75)
        
        return Image.fromarray(enhanced_array)
    
    def _resize_image(self, image: Image.Image, max_size: int) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class LLaVAMedicalModel:
    """
    Enhanced LLaVA-Med model with quantum integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLaVA-Med model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_path = config.get('model_path', 'microsoft/llava-med-v1.5-mistral-7b')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize quantum processor
        self.quantum_processor = QuantumMedicalImageProcessor(config)
        
        # Initialize image preprocessor
        self.image_preprocessor = MedicalImagePreprocessor(config)
        
        # Load LLaVA-Med model
        self._load_model()
        
        logger.info(f"Initialized LLaVAMedicalModel on device: {self.device}")
    
    def _load_model(self):
        """Load the LLaVA-Med model"""
        if not LLAVA_AVAILABLE:
            logger.warning("LLaVA libraries not available - using mock model")
            self.model = MockLLaVAModel()
            self.tokenizer = None
            self.image_processor = None
            self.context_len = 2048
            self.conv_mode = "llava_v1"
            return
            
        try:
            disable_torch_init()
            
            # Get model name
            model_name = get_model_name_from_path(self.model_path)
            
            # Load model components
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=False,
                load_4bit=False,
                device=self.device
            )
            
            # Set up conversation template
            if 'llama-2' in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"
            
            self.conv_mode = conv_mode
            
        except Exception as e:
            logger.error(f"Error loading LLaVA-Med model: {str(e)}")
            # Use mock model for development
            self.model = MockLLaVAModel()
            self.tokenizer = None
            self.image_processor = None
            self.context_len = 2048
            self.conv_mode = "llava_v1"
    
    def analyze_medical_image(self, image_path: str, query: str, 
                            patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze medical image with quantum-enhanced LLaVA-Med
        
        Args:
            image_path: Path to medical image
            query: Medical query about the image
            patient_context: Optional patient context
            
        Returns:
            Analysis results with quantum enhancements
        """
        try:
            # Preprocess medical image
            image, metadata = self.image_preprocessor.preprocess_image(image_path)
            
            # Extract classical features
            image_features = self._extract_image_features(image)
            
            # Apply quantum enhancement
            quantum_results = self.quantum_processor.process_medical_image(
                image_features, patient_context
            )
            
            # Generate LLaVA-Med response
            llava_response = self._generate_llava_response(image, query, metadata)
            
            # Combine results
            analysis_results = {
                'llava_response': llava_response,
                'quantum_features': quantum_results['quantum_features'],
                'uncertainty_metrics': quantum_results['uncertainty_metrics'],
                'image_metadata': metadata,
                'processing_info': {
                    'quantum_enhanced': quantum_results['quantum_enhanced'],
                    'model_used': self.model_path,
                    'device': self.device,
                    'timestamp': str(np.datetime64('now'))
                }
            }
            
            # Ensure all numpy arrays are converted to lists for JSON serialization
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
            
            return convert_numpy(analysis_results)
            
        except Exception as e:
            logger.error(f"Error analyzing medical image: {str(e)}")
            return {
                'error': str(e),
                'llava_response': 'Error occurred during analysis',
                'quantum_features': [],
                'uncertainty_metrics': {'confidence': 0.0},
                'processing_info': {'quantum_enhanced': False}
            }
    
    def _extract_image_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from medical image"""
        try:
            if self.image_processor is not None:
                # Use LLaVA image processor
                processed_image = self.image_processor.preprocess(image, return_tensors='pt')
                image_tensor = processed_image['pixel_values'].to(self.device)
                
                # Extract features using model's vision encoder
                with torch.no_grad():
                    features = self.model.get_model().get_vision_tower()(image_tensor)
                    features = features.cpu().numpy().flatten()
                
                return features
            else:
                # Fallback: simple feature extraction
                img_array = np.array(image.resize((224, 224)))
                return img_array.flatten().astype(np.float32) / 255.0
                
        except Exception as e:
            logger.error(f"Error extracting image features: {str(e)}")
            # Fallback feature extraction
            img_array = np.array(image.resize((224, 224)))
            return img_array.flatten().astype(np.float32) / 255.0
    
    def _generate_llava_response(self, image: Image.Image, query: str, 
                               metadata: Dict[str, Any]) -> str:
        """Generate response using LLaVA-Med model"""
        try:
            if isinstance(self.model, MockLLaVAModel):
                return f"Mock LLaVA-Med analysis for {metadata.get('format', 'unknown')} image: {query}"
            
            # Prepare conversation
            conv = conv_templates[self.conv_mode].copy()
            
            # Add medical context to the query
            medical_query = self._enhance_query_with_context(query, metadata)
            
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + medical_query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            
            # Process image
            if self.image_processor is not None:
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
            else:
                # Fallback
                image_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                    use_cache=True
                )
            
            # Decode response
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Clean up the response
            if conv.sep_style == SeparatorStyle.TWO:
                outputs = outputs.split(conv.sep2)[-1].strip()
            else:
                outputs = outputs.split(conv.sep)[-1].strip()
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error generating LLaVA response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _enhance_query_with_context(self, query: str, metadata: Dict[str, Any]) -> str:
        """Enhance query with medical context"""
        enhanced_query = query
        
        # Add modality information if available
        if 'modality' in metadata:
            enhanced_query = f"This is a {metadata['modality']} medical image. " + enhanced_query
        
        # Add body part information if available
        if 'body_part' in metadata:
            enhanced_query += f" The image shows the {metadata['body_part']}."
        
        # Add medical imaging specific instructions
        enhanced_query += " Please provide a detailed medical analysis focusing on any abnormalities, pathological findings, and diagnostic insights. Include confidence levels where appropriate."
        
        return enhanced_query
    
    def batch_analyze_images(self, image_paths: List[str], queries: List[str],
                           patient_contexts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
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
            result = self.analyze_medical_image(image_path, query, context)
            results.append(result)
        
        return results


class MedicalTerminologyProcessor:
    """
    Processor for medical terminology and context understanding
    """
    
    def __init__(self):
        """Initialize medical terminology processor"""
        self.medical_terms = self._load_medical_terminology()
        logger.info("Initialized MedicalTerminologyProcessor")
    
    def _load_medical_terminology(self) -> Dict[str, Any]:
        """Load medical terminology database"""
        # This would typically load from a comprehensive medical database
        # For now, we'll use a subset of common medical terms
        return {
            'anatomical_terms': [
                'thorax', 'abdomen', 'pelvis', 'cranium', 'spine', 'extremities',
                'heart', 'lungs', 'liver', 'kidney', 'brain', 'bones'
            ],
            'pathological_terms': [
                'fracture', 'pneumonia', 'tumor', 'lesion', 'inflammation',
                'atelectasis', 'consolidation', 'effusion', 'mass', 'nodule'
            ],
            'imaging_terms': [
                'radiopaque', 'radiolucent', 'contrast', 'enhancement',
                'artifact', 'resolution', 'signal intensity'
            ]
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted medical entities
        """
        text_lower = text.lower()
        extracted = {
            'anatomical_terms': [],
            'pathological_terms': [],
            'imaging_terms': []
        }
        
        for category, terms in self.medical_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    extracted[category].append(term)
        
        return extracted
    
    def generate_medical_context(self, extracted_entities: Dict[str, List[str]]) -> str:
        """
        Generate medical context from extracted entities
        
        Args:
            extracted_entities: Extracted medical entities
            
        Returns:
            Generated medical context string
        """
        context_parts = []
        
        if extracted_entities['anatomical_terms']:
            context_parts.append(f"Anatomical regions of interest: {', '.join(extracted_entities['anatomical_terms'])}")
        
        if extracted_entities['pathological_terms']:
            context_parts.append(f"Potential pathological findings: {', '.join(extracted_entities['pathological_terms'])}")
        
        if extracted_entities['imaging_terms']:
            context_parts.append(f"Imaging characteristics: {', '.join(extracted_entities['imaging_terms'])}")
        
        return ". ".join(context_parts) if context_parts else "No specific medical context identified."
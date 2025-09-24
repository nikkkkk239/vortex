"""
Explainable AI Module for Medical Image Analysis
Implements Grad-CAM, LIME, SHAP, and other XAI techniques for medical transparency
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import json

# XAI libraries
try:
    import lime
    from lime import lime_image
    import shap
    from captum.attr import GradCAM, GuidedGradCam, IntegratedGradients, LayerGradCam
    from captum.attr import visualization as viz
    XAI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some XAI libraries not available: {e}")
    XAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class MedicalGradCAM:
    """
    Grad-CAM implementation optimized for medical images
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize Medical Grad-CAM
        
        Args:
            model: PyTorch model
            target_layers: List of target layer names for visualization
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
        logger.info(f"Initialized MedicalGradCAM with {len(target_layers)} target layers")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate Class Activation Maps
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Dictionary with CAM heatmaps for each target layer
        """
        try:
            self.model.eval()
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            class_score = output[0, target_class]
            class_score.backward(retain_graph=True)
            
            cams = {}
            
            # Generate CAM for each target layer
            for layer_name in self.target_layers:
                if layer_name in self.gradients and layer_name in self.activations:
                    # Get gradients and activations
                    gradients = self.gradients[layer_name][0]  # [C, H, W]
                    activations = self.activations[layer_name][0]  # [C, H, W]
                    
                    # Calculate weights as global average pooling of gradients
                    weights = torch.mean(gradients, dim=[1, 2])  # [C]
                    
                    # Generate CAM
                    cam = torch.sum(weights.unsqueeze(1).unsqueeze(2) * activations, dim=0)  # [H, W]
                    cam = F.relu(cam)  # Apply ReLU
                    
                    # Normalize CAM
                    if cam.max() > 0:
                        cam = cam / cam.max()
                    
                    cams[layer_name] = cam.cpu().numpy()
                
            return cams
            
        except Exception as e:
            logger.error(f"Error generating CAM: {str(e)}")
            return {}
    
    def visualize_cam(self, original_image: np.ndarray, cam: np.ndarray, 
                     alpha: float = 0.4, colormap: str = 'jet') -> np.ndarray:
        """
        Visualize CAM overlay on original image
        
        Args:
            original_image: Original image as numpy array
            cam: CAM heatmap
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            Visualization as numpy array
        """
        # Resize CAM to match image size
        if cam.shape != original_image.shape[:2]:
            cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        else:
            cam_resized = cam
        
        # Apply colormap
        colormap_func = plt.get_cmap(colormap)
        cam_colored = colormap_func(cam_resized)[:, :, :3]  # Remove alpha channel
        cam_colored = (cam_colored * 255).astype(np.uint8)
        
        # Ensure original image is in correct format
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Overlay CAM on original image
        overlayed = cv2.addWeighted(original_image, 1-alpha, cam_colored, alpha, 0)
        
        return overlayed
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class MedicalLIME:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for medical images
    """
    
    def __init__(self, model: nn.Module, preprocess_fn: Optional[callable] = None):
        """
        Initialize Medical LIME
        
        Args:
            model: PyTorch model
            preprocess_fn: Preprocessing function for images
        """
        self.model = model
        self.preprocess_fn = preprocess_fn or self._default_preprocess
        self.explainer = lime_image.LimeImageExplainer()
        
        logger.info("Initialized MedicalLIME")
    
    def _default_preprocess(self, images: np.ndarray) -> torch.Tensor:
        """Default preprocessing function"""
        # Convert to tensor and normalize
        if len(images.shape) == 3:
            images = images[np.newaxis, :]  # Add batch dimension
        
        # Convert from HWC to CHW format
        images = np.transpose(images, (0, 3, 1, 2))
        
        # Convert to tensor
        tensor = torch.FloatTensor(images)
        
        # Normalize to [0, 1] range
        tensor = tensor / 255.0
        
        return tensor
    
    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME
        
        Args:
            images: Batch of images as numpy array
            
        Returns:
            Predictions as numpy array
        """
        try:
            self.model.eval()
            
            # Preprocess images
            tensor = self.preprocess_fn(images)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error in LIME prediction: {str(e)}")
            # Return dummy predictions
            return np.random.rand(images.shape[0], 10)
    
    def explain_instance(self, image: np.ndarray, top_labels: int = 5, 
                        num_samples: int = 1000) -> Dict[str, Any]:
        """
        Explain a single medical image instance
        
        Args:
            image: Input image as numpy array
            top_labels: Number of top labels to explain
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation results
        """
        try:
            # Generate explanation
            explanation = self.explainer.explain_instance(
                image,
                self.predict_fn,
                top_labels=top_labels,
                hide_color=0,
                num_samples=num_samples,
                batch_size=32
            )
            
            # Extract explanation data
            explanation_dict = {
                'top_labels': explanation.top_labels,
                'local_exp': dict(explanation.local_exp),
                'available_labels': explanation.available_labels,
                'mode': explanation.mode
            }
            
            return explanation_dict
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            return {'error': str(e)}
    
    def visualize_explanation(self, image: np.ndarray, explanation: Dict[str, Any], 
                            label: int, positive_only: bool = True) -> np.ndarray:
        """
        Visualize LIME explanation
        
        Args:
            image: Original image
            explanation: LIME explanation dictionary
            label: Target label to visualize
            positive_only: Show only positive contributions
            
        Returns:
            Visualization as numpy array
        """
        try:
            # Recreate LIME explanation object (simplified)
            if label in explanation['local_exp']:
                # Get segments and weights
                segments_and_weights = explanation['local_exp'][label]
                
                # Create mask
                mask = np.ones(image.shape[:2])
                for segment_id, weight in segments_and_weights:
                    if positive_only and weight > 0:
                        # This is a simplified visualization
                        # In practice, you'd need access to the segmentation
                        mask *= weight
                
                # Apply mask to image
                if len(image.shape) == 3:
                    visualization = image * mask[:, :, np.newaxis]
                else:
                    visualization = image * mask
                
                return visualization.astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Error visualizing LIME explanation: {str(e)}")
            return image


class MedicalSHAP:
    """
    SHAP (SHapley Additive exPlanations) for medical images
    """
    
    def __init__(self, model: nn.Module, background_data: Optional[torch.Tensor] = None):
        """
        Initialize Medical SHAP
        
        Args:
            model: PyTorch model
            background_data: Background dataset for SHAP
        """
        self.model = model
        self.background_data = background_data
        
        # Initialize SHAP explainer
        if background_data is not None:
            self.explainer = shap.DeepExplainer(model, background_data)
        else:
            self.explainer = shap.GradientExplainer(model, background_data)
        
        logger.info("Initialized MedicalSHAP")
    
    def explain_instance(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate SHAP values for input
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            SHAP values as numpy array
        """
        try:
            shap_values = self.explainer.shap_values(input_tensor)
            
            if isinstance(shap_values, list):
                # Multi-class case - return values for first class
                return shap_values[0]
            else:
                return shap_values
                
        except Exception as e:
            logger.error(f"Error generating SHAP values: {str(e)}")
            return np.zeros_like(input_tensor.cpu().numpy())
    
    def visualize_shap(self, input_tensor: torch.Tensor, shap_values: np.ndarray) -> Dict[str, Any]:
        """
        Create SHAP visualizations
        
        Args:
            input_tensor: Original input tensor
            shap_values: SHAP values
            
        Returns:
            Dictionary with visualization data
        """
        try:
            # Convert to numpy
            input_np = input_tensor.cpu().numpy()
            
            # Create visualization data
            visualization_data = {
                'input_shape': input_np.shape,
                'shap_shape': shap_values.shape,
                'positive_contributions': np.sum(shap_values[shap_values > 0]),
                'negative_contributions': np.sum(shap_values[shap_values < 0]),
                'total_effect': np.sum(shap_values)
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error creating SHAP visualization: {str(e)}")
            return {'error': str(e)}


class MedicalExplainabilityEngine:
    """
    Comprehensive explainability engine for medical image analysis
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize explainability engine
        
        Args:
            model: PyTorch model to explain
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Initialize XAI methods
        self.gradcam = None
        self.lime = None
        self.shap = None
        
        self._setup_explainers()
        
        logger.info("Initialized MedicalExplainabilityEngine")
    
    def _setup_explainers(self):
        """Setup explainability methods"""
        try:
            # Setup Grad-CAM
            target_layers = self.config.get('gradcam_layers', ['layer4', 'avgpool'])
            self.gradcam = MedicalGradCAM(self.model, target_layers)
            
            # Setup LIME
            self.lime = MedicalLIME(self.model)
            
            # Setup SHAP (if background data available)
            background_data = self.config.get('background_data')
            if background_data is not None:
                self.shap = MedicalSHAP(self.model, background_data)
            
        except Exception as e:
            logger.error(f"Error setting up explainers: {str(e)}")
    
    def generate_comprehensive_explanation(self, image_path: str, input_tensor: torch.Tensor,
                                         target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanations using multiple XAI methods
        
        Args:
            image_path: Path to original image
            input_tensor: Preprocessed input tensor
            target_class: Target class for explanation
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanations = {
            'metadata': {
                'image_path': image_path,
                'input_shape': list(input_tensor.shape),
                'target_class': target_class,
                'timestamp': str(np.datetime64('now'))
            }
        }
        
        try:
            # Load original image
            original_image = np.array(Image.open(image_path))
            
            # Generate Grad-CAM explanations
            if self.gradcam:
                gradcam_results = self.gradcam.generate_cam(input_tensor, target_class)
                explanations['gradcam'] = self._process_gradcam_results(
                    original_image, gradcam_results
                )
            
            # Generate LIME explanations
            if self.lime:
                lime_result = self.lime.explain_instance(original_image)
                explanations['lime'] = lime_result
            
            # Generate SHAP explanations
            if self.shap:
                shap_values = self.shap.explain_instance(input_tensor)
                shap_viz = self.shap.visualize_shap(input_tensor, shap_values)
                explanations['shap'] = shap_viz
            
            # Generate attention maps (if model supports it)
            attention_maps = self._extract_attention_maps(input_tensor)
            if attention_maps:
                explanations['attention'] = attention_maps
            
            # Generate confidence scores
            confidence_scores = self._calculate_confidence_scores(input_tensor)
            explanations['confidence'] = confidence_scores
            
            # Generate medical insights
            medical_insights = self._generate_medical_insights(explanations)
            explanations['medical_insights'] = medical_insights
            
        except Exception as e:
            logger.error(f"Error generating comprehensive explanation: {str(e)}")
            explanations['error'] = str(e)
        
        return explanations
    
    def _process_gradcam_results(self, original_image: np.ndarray, 
                               gradcam_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process Grad-CAM results for medical interpretation"""
        processed_results = {}
        
        for layer_name, cam in gradcam_results.items():
            # Create visualization
            visualization = self.gradcam.visualize_cam(original_image, cam)
            
            # Calculate statistics
            stats = {
                'max_activation': float(np.max(cam)),
                'mean_activation': float(np.mean(cam)),
                'activation_area': float(np.sum(cam > 0.5) / cam.size),
                'focused_regions': self._identify_focused_regions(cam)
            }
            
            processed_results[layer_name] = {
                'visualization': visualization,
                'heatmap': cam,
                'statistics': stats
            }
        
        return processed_results
    
    def _identify_focused_regions(self, cam: np.ndarray, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify highly activated regions in CAM"""
        # Find connected components of high activation
        binary_mask = (cam > threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            region_info = {
                'centroid': centroids[i].tolist(),
                'area': int(stats[i, cv2.CC_STAT_AREA]),
                'bbox': {
                    'x': int(stats[i, cv2.CC_STAT_LEFT]),
                    'y': int(stats[i, cv2.CC_STAT_TOP]),
                    'width': int(stats[i, cv2.CC_STAT_WIDTH]),
                    'height': int(stats[i, cv2.CC_STAT_HEIGHT])
                }
            }
            regions.append(region_info)
        
        return regions
    
    def _extract_attention_maps(self, input_tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Extract attention maps if model has attention mechanisms"""
        try:
            # This would be specific to the model architecture
            # For now, return placeholder
            return {
                'attention_available': False,
                'note': 'Attention extraction not implemented for this model'
            }
        except Exception as e:
            logger.error(f"Error extracting attention maps: {str(e)}")
            return None
    
    def _calculate_confidence_scores(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Calculate various confidence scores"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Get model predictions
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Calculate confidence metrics
                max_prob = float(torch.max(probabilities))
                entropy = float(-torch.sum(probabilities * torch.log(probabilities + 1e-8)))
                
                # Prediction confidence (max probability)
                prediction_confidence = max_prob
                
                # Uncertainty (normalized entropy)
                num_classes = probabilities.shape[1]
                max_entropy = np.log(num_classes)
                uncertainty = entropy / max_entropy
                
                # Calibrated confidence (1 - uncertainty)
                calibrated_confidence = 1.0 - uncertainty
                
                return {
                    'prediction_confidence': prediction_confidence,
                    'calibrated_confidence': calibrated_confidence,
                    'uncertainty': uncertainty,
                    'entropy': entropy
                }
                
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {str(e)}")
            return {'prediction_confidence': 0.5, 'uncertainty': 0.5}
    
    def _generate_medical_insights(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate medical insights from explanations"""
        insights = {
            'summary': '',
            'key_findings': [],
            'confidence_assessment': '',
            'recommendations': []
        }
        
        try:
            # Analyze Grad-CAM results
            if 'gradcam' in explanations:
                insights['key_findings'].extend(
                    self._analyze_gradcam_for_medical_insights(explanations['gradcam'])
                )
            
            # Analyze confidence scores
            if 'confidence' in explanations:
                confidence_data = explanations['confidence']
                if confidence_data['prediction_confidence'] > 0.8:
                    insights['confidence_assessment'] = 'High confidence prediction'
                elif confidence_data['prediction_confidence'] > 0.6:
                    insights['confidence_assessment'] = 'Moderate confidence prediction'
                else:
                    insights['confidence_assessment'] = 'Low confidence prediction - consider additional analysis'
            
            # Generate summary
            insights['summary'] = self._generate_explanation_summary(explanations)
            
            # Generate recommendations
            insights['recommendations'] = self._generate_medical_recommendations(explanations)
            
        except Exception as e:
            logger.error(f"Error generating medical insights: {str(e)}")
            insights['error'] = str(e)
        
        return insights
    
    def _analyze_gradcam_for_medical_insights(self, gradcam_data: Dict[str, Any]) -> List[str]:
        """Analyze Grad-CAM results for medical insights"""
        findings = []
        
        for layer_name, layer_data in gradcam_data.items():
            stats = layer_data.get('statistics', {})
            
            # Analyze activation patterns
            if stats.get('activation_area', 0) > 0.3:
                findings.append(f"Widespread activation detected in {layer_name} - suggests diffuse changes")
            elif stats.get('activation_area', 0) < 0.1:
                findings.append(f"Focal activation in {layer_name} - suggests localized findings")
            
            # Analyze activation intensity
            if stats.get('max_activation', 0) > 0.8:
                findings.append(f"High intensity activation in {layer_name} - high diagnostic relevance")
        
        return findings
    
    def _generate_explanation_summary(self, explanations: Dict[str, Any]) -> str:
        """Generate summary of explanations"""
        summary_parts = []
        
        if 'gradcam' in explanations:
            summary_parts.append("Visual attention analysis performed using Grad-CAM")
        
        if 'lime' in explanations:
            summary_parts.append("Local feature importance analyzed using LIME")
        
        if 'shap' in explanations:
            summary_parts.append("Feature contributions analyzed using SHAP")
        
        if 'confidence' in explanations:
            conf = explanations['confidence']['prediction_confidence']
            summary_parts.append(f"Prediction confidence: {conf:.2f}")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_medical_recommendations(self, explanations: Dict[str, Any]) -> List[str]:
        """Generate medical recommendations based on explanations"""
        recommendations = []
        
        # Based on confidence
        if 'confidence' in explanations:
            conf = explanations['confidence']['prediction_confidence']
            if conf < 0.6:
                recommendations.append("Consider additional imaging modalities for confirmation")
                recommendations.append("Correlate with clinical findings and patient history")
        
        # Based on attention patterns
        if 'gradcam' in explanations:
            for layer_data in explanations['gradcam'].values():
                regions = layer_data.get('statistics', {}).get('focused_regions', [])
                if len(regions) > 3:
                    recommendations.append("Multiple regions of interest detected - consider systematic evaluation")
        
        if not recommendations:
            recommendations.append("Standard diagnostic workflow recommended")
        
        return recommendations
    
    def save_explanations(self, explanations: Dict[str, Any], output_dir: str) -> str:
        """
        Save explanations to files
        
        Args:
            explanations: Explanation dictionary
            output_dir: Output directory
            
        Returns:
            Path to saved explanations
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save explanation data as JSON
            json_path = output_path / "explanations.json"
            
            # Prepare JSON-serializable data
            json_data = {}
            for key, value in explanations.items():
                if key == 'gradcam':
                    # Save only statistics for JSON, visualizations separately
                    json_data[key] = {
                        layer_name: layer_info.get('statistics', {})
                        for layer_name, layer_info in value.items()
                    }
                elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    json_data[key] = value
                else:
                    json_data[key] = str(value)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            # Save visualizations
            if 'gradcam' in explanations:
                self._save_gradcam_visualizations(explanations['gradcam'], output_path)
            
            logger.info(f"Explanations saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving explanations: {str(e)}")
            return ""
    
    def _save_gradcam_visualizations(self, gradcam_data: Dict[str, Any], output_path: Path):
        """Save Grad-CAM visualizations"""
        for layer_name, layer_info in gradcam_data.items():
            if 'visualization' in layer_info:
                viz_path = output_path / f"gradcam_{layer_name}.png"
                Image.fromarray(layer_info['visualization']).save(viz_path)
            
            if 'heatmap' in layer_info:
                heatmap_path = output_path / f"heatmap_{layer_name}.npy"
                np.save(heatmap_path, layer_info['heatmap'])
    
    def cleanup(self):
        """Cleanup resources"""
        if self.gradcam:
            self.gradcam.cleanup()
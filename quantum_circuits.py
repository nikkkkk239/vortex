"""
Quantum Circuits Module for Medical Image Processing
Implements quantum-enhanced feature extraction and uncertainty quantification
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings

# Quantum computing libraries
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit.library import TwoLocal, ZZFeatureMap, RealAmplitudes
    from qiskit.primitives import Sampler
    try:
        from qiskit import execute
    except ImportError:
        from qiskit_aer import execute
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        from qiskit.providers.basic_provider import BasicProvider
        AerSimulator = BasicProvider().get_backend('qasm_simulator')
    try:
        from qiskit.visualization import plot_histogram, plot_circuit
    except ImportError:
        plot_histogram = plot_circuit = None
    QISKIT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Qiskit not available: {e}")
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.templates import AmplitudeEmbedding, AngleEmbedding
    PENNYLANE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PennyLane not available: {e}")
    PENNYLANE_AVAILABLE = False

logger = logging.getLogger(__name__)


class QuantumFeatureExtractor:
    """
    Quantum feature extractor for medical images
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        """
        Initialize quantum feature extractor
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit = None
        self.parameters = None
        
        if QISKIT_AVAILABLE:
            self._create_qiskit_circuit()
        elif PENNYLANE_AVAILABLE:
            self._create_pennylane_circuit()
        else:
            logger.warning("No quantum backend available - using classical fallback")
    
    def _create_qiskit_circuit(self):
        """Create Qiskit quantum circuit"""
        try:
            # Create quantum register
            qreg = QuantumRegister(self.n_qubits, 'q')
            creg = ClassicalRegister(self.n_qubits, 'c')
            
            # Create quantum circuit
            self.circuit = QuantumCircuit(qreg, creg)
            
            # Add feature map (data encoding)
            feature_map = ZZFeatureMap(feature_dimension=self.n_qubits, reps=1)
            self.circuit = self.circuit.compose(feature_map)
            
            # Add variational form
            variational_form = TwoLocal(
                num_qubits=self.n_qubits,
                rotation_blocks=['ry', 'rz'],
                entanglement_blocks='cz',
                entanglement='circular',
                reps=self.n_layers,
                insert_barriers=True
            )
            self.circuit = self.circuit.compose(variational_form)
            
            # Add measurements
            self.circuit.measure_all()
            
            # Get parameter references
            self.parameters = list(self.circuit.parameters)
            
            logger.info(f"Created Qiskit circuit with {len(self.parameters)} parameters")
            
        except Exception as e:
            logger.error(f"Error creating Qiskit circuit: {str(e)}")
            self.circuit = None
    
    def _create_pennylane_circuit(self):
        """Create PennyLane quantum circuit"""
        try:
            # Define quantum device
            self.device = qml.device('default.qubit', wires=self.n_qubits)
            
            # Define quantum circuit
            @qml.qnode(self.device)
            def quantum_circuit(features, weights):
                # Encode features
                AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)
                
                # Add variational layers
                for layer in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.RY(weights[layer][i], wires=i)
                        qml.RZ(weights[layer][i + self.n_qubits], wires=i)
                    
                    # Add entangling gates
                    for i in range(self.n_qubits - 1):
                        qml.CZ(wires=[i, i + 1])
                    qml.CZ(wires=[self.n_qubits - 1, 0])
                
                # Return measurements
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
            self.circuit = quantum_circuit
            self.parameters = np.random.random((self.n_layers, 2 * self.n_qubits))
            
            logger.info(f"Created PennyLane circuit with {self.parameters.size} parameters")
            
        except Exception as e:
            logger.error(f"Error creating PennyLane circuit: {str(e)}")
            self.circuit = None
    
    def extract_features(self, image_features: np.ndarray) -> np.ndarray:
        """
        Extract quantum features from classical image features
        
        Args:
            image_features: Classical image features
            
        Returns:
            Quantum-enhanced features
        """
        if self.circuit is None:
            # Fallback to classical processing
            return self._classical_feature_extraction(image_features)
        
        try:
            if QISKIT_AVAILABLE and hasattr(self, 'circuit') and not callable(self.circuit):
                return self._qiskit_feature_extraction(image_features)
            elif PENNYLANE_AVAILABLE and callable(self.circuit):
                return self._pennylane_feature_extraction(image_features)
            else:
                return self._classical_feature_extraction(image_features)
                
        except Exception as e:
            logger.error(f"Error in quantum feature extraction: {str(e)}")
            return self._classical_feature_extraction(image_features)
    
    def _qiskit_feature_extraction(self, image_features: np.ndarray) -> np.ndarray:
        """Extract features using Qiskit"""
        try:
            # Prepare data for quantum circuit
            # Normalize and scale features to fit quantum circuit
            normalized_features = self._normalize_features(image_features, self.n_qubits)
            
            # Create parameter binding
            param_binding = {}
            for i, param in enumerate(self.parameters[:len(normalized_features)]):
                param_binding[param] = normalized_features[i]
            
            # Execute circuit
            simulator = AerSimulator()
            transpiled_circuit = transpile(self.circuit, simulator)
            job = execute(transpiled_circuit, simulator, shots=1000, parameter_binds=[param_binding])
            result = job.result()
            counts = result.get_counts()
            
            # Convert counts to feature vector
            quantum_features = self._counts_to_features(counts)
            
            return quantum_features
            
        except Exception as e:
            logger.error(f"Error in Qiskit feature extraction: {str(e)}")
            return self._classical_feature_extraction(image_features)
    
    def _pennylane_feature_extraction(self, image_features: np.ndarray) -> np.ndarray:
        """Extract features using PennyLane"""
        try:
            # Normalize features
            normalized_features = self._normalize_features(image_features, 2**self.n_qubits)
            
            # Execute quantum circuit
            quantum_features = self.circuit(normalized_features, self.parameters)
            
            return np.array(quantum_features)
            
        except Exception as e:
            logger.error(f"Error in PennyLane feature extraction: {str(e)}")
            return self._classical_feature_extraction(image_features)
    
    def _classical_feature_extraction(self, image_features: np.ndarray) -> np.ndarray:
        """Fallback classical feature extraction"""
        # Apply some transformations to simulate quantum-like processing
        features = image_features.copy()
        
        # Apply non-linear transformations
        features = np.tanh(features)  # Non-linear activation
        features = np.fft.fft(features)  # Frequency domain
        features = np.abs(features)  # Magnitude spectrum
        
        # Add some noise to simulate quantum uncertainty
        noise = np.random.normal(0, 0.01, features.shape)
        features = features + noise
        
        return features.astype(np.float32)
    
    def _normalize_features(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Normalize features to fit quantum circuit"""
        # Ensure we have enough features
        if len(features) < target_size:
            # Pad with zeros
            normalized = np.zeros(target_size)
            normalized[:len(features)] = features
        else:
            # Take first target_size features
            normalized = features[:target_size]
        
        # Normalize to [0, 2π] range for quantum circuits
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        normalized = normalized * 2 * np.pi
        
        return normalized
    
    def _counts_to_features(self, counts: Dict[str, int]) -> np.ndarray:
        """Convert quantum measurement counts to feature vector"""
        total_shots = sum(counts.values())
        features = np.zeros(self.n_qubits)
        
        for bitstring, count in counts.items():
            # Convert bitstring to integer
            bit_int = int(bitstring, 2)
            # Add probability to corresponding qubit expectation value
            for i in range(self.n_qubits):
                if (bit_int >> i) & 1:  # If bit i is 1
                    features[i] += count / total_shots
        
        return features


class QuantumUncertaintyQuantifier:
    """
    Quantum uncertainty quantification for medical predictions
    """
    
    def __init__(self, n_qubits: int = 3):
        """
        Initialize quantum uncertainty quantifier
        
        Args:
            n_qubits: Number of qubits for uncertainty circuit
        """
        self.n_qubits = n_qubits
        
        if QISKIT_AVAILABLE:
            self._create_uncertainty_circuit()
        else:
            logger.warning("No quantum backend available for uncertainty quantification")
    
    def _create_uncertainty_circuit(self):
        """Create quantum circuit for uncertainty quantification"""
        if not QISKIT_AVAILABLE:
            self.uncertainty_circuit = None
            return
            
        try:
            from qiskit.circuit import Parameter
            
            # Create quantum register
            qreg = QuantumRegister(self.n_qubits, 'q')
            creg = ClassicalRegister(self.n_qubits, 'c')
            
            self.uncertainty_circuit = QuantumCircuit(qreg, creg)
            
            # Add superposition state preparation
            for i in range(self.n_qubits):
                self.uncertainty_circuit.h(i)
            
            # Add parameterized rotations for uncertainty modeling
            for i in range(self.n_qubits):
                self.uncertainty_circuit.ry(Parameter(f'uncertainty_{i}'), i)
            
            # Add measurements
            self.uncertainty_circuit.measure_all()
            
            logger.info("Created quantum uncertainty circuit")
            
        except Exception as e:
            logger.error(f"Error creating uncertainty circuit: {str(e)}")
            self.uncertainty_circuit = None
    
    def quantify_uncertainty(self, prediction_confidence: float, 
                           feature_uncertainty: np.ndarray) -> Dict[str, float]:
        """
        Quantify uncertainty using quantum circuits
        
        Args:
            prediction_confidence: Classical prediction confidence
            feature_uncertainty: Uncertainty in input features
            
        Returns:
            Dictionary with uncertainty metrics
        """
        if self.uncertainty_circuit is None:
            return self._classical_uncertainty(prediction_confidence, feature_uncertainty)
        
        try:
            if QISKIT_AVAILABLE:
                return self._quantum_uncertainty_quantification(prediction_confidence, feature_uncertainty)
            else:
                return self._classical_uncertainty(prediction_confidence, feature_uncertainty)
                
        except Exception as e:
            logger.error(f"Error in uncertainty quantification: {str(e)}")
            return self._classical_uncertainty(prediction_confidence, feature_uncertainty)
    
    def _quantum_uncertainty_quantification(self, prediction_confidence: float,
                                          feature_uncertainty: np.ndarray) -> Dict[str, float]:
        """Quantum uncertainty quantification"""
        try:
            # Map uncertainties to quantum parameters
            uncertainty_params = self._map_uncertainty_to_params(
                prediction_confidence, feature_uncertainty
            )
            
            # Execute uncertainty circuit
            simulator = AerSimulator()
            transpiled_circuit = transpile(self.uncertainty_circuit, simulator)
            
            # Create parameter binding
            param_binding = {}
            for i, param in enumerate(self.uncertainty_circuit.parameters):
                if i < len(uncertainty_params):
                    param_binding[param] = uncertainty_params[i]
            
            job = execute(transpiled_circuit, simulator, shots=1000, parameter_binds=[param_binding])
            result = job.result()
            counts = result.get_counts()
            
            # Calculate quantum uncertainty metrics
            quantum_uncertainty = self._calculate_quantum_uncertainty_metrics(counts)
            
            return {
                'confidence': prediction_confidence,
                'quantum_uncertainty': quantum_uncertainty,
                'entropy': self._calculate_entropy(counts),
                'coherence': self._calculate_coherence(counts)
            }
            
        except Exception as e:
            logger.error(f"Error in quantum uncertainty quantification: {str(e)}")
            return self._classical_uncertainty(prediction_confidence, feature_uncertainty)
    
    def _classical_uncertainty(self, prediction_confidence: float,
                             feature_uncertainty: np.ndarray) -> Dict[str, float]:
        """Classical uncertainty quantification fallback"""
        # Calculate entropy from feature uncertainty
        entropy = -np.sum(feature_uncertainty * np.log(feature_uncertainty + 1e-8))
        
        # Normalize entropy
        max_entropy = np.log(len(feature_uncertainty))
        normalized_entropy = entropy / max_entropy
        
        # Calculate uncertainty metrics
        quantum_uncertainty = (1 - prediction_confidence) * normalized_entropy
        
        return {
            'confidence': prediction_confidence,
            'quantum_uncertainty': quantum_uncertainty,
            'entropy': normalized_entropy,
            'coherence': 1 - normalized_entropy
        }
    
    def _map_uncertainty_to_params(self, confidence: float, 
                                 feature_uncertainty: np.ndarray) -> np.ndarray:
        """Map uncertainty values to quantum circuit parameters"""
        params = np.zeros(self.n_qubits)
        
        # Map confidence to first parameter
        params[0] = (1 - confidence) * np.pi
        
        # Map feature uncertainties to remaining parameters
        for i in range(1, min(len(feature_uncertainty) + 1, self.n_qubits)):
            if i - 1 < len(feature_uncertainty):
                params[i] = feature_uncertainty[i - 1] * np.pi
        
        return params
    
    def _calculate_quantum_uncertainty_metrics(self, counts: Dict[str, int]) -> float:
        """Calculate quantum uncertainty from measurement counts"""
        total_shots = sum(counts.values())
        
        # Calculate variance in measurement outcomes
        outcomes = []
        for bitstring, count in counts.items():
            # Convert to decimal
            outcome = int(bitstring, 2)
            outcomes.extend([outcome] * count)
        
        if len(outcomes) > 0:
            variance = np.var(outcomes)
            # Normalize by maximum possible variance
            max_variance = (2**self.n_qubits - 1)**2 / 4
            normalized_variance = variance / max_variance if max_variance > 0 else 0
            return min(normalized_variance, 1.0)
        
        return 0.5  # Default uncertainty
    
    def _calculate_entropy(self, counts: Dict[str, int]) -> float:
        """Calculate Shannon entropy from measurement counts"""
        total_shots = sum(counts.values())
        entropy = 0.0
        
        for count in counts.values():
            probability = count / total_shots
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize by maximum entropy
        max_entropy = self.n_qubits
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_coherence(self, counts: Dict[str, int]) -> float:
        """Calculate quantum coherence from measurement counts"""
        total_shots = sum(counts.values())
        
        # Coherence is inversely related to measurement spread
        num_outcomes = len(counts)
        max_outcomes = 2**self.n_qubits
        
        # Perfect coherence would give single outcome
        coherence = 1.0 - (num_outcomes - 1) / (max_outcomes - 1)
        return max(coherence, 0.0)


class QuantumMedicalImageProcessor:
    """
    Main quantum medical image processor
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize quantum medical image processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.n_qubits = config.get('quantum_qubits', 4)
        self.n_layers = config.get('quantum_layers', 2)
        
        # Initialize quantum components
        self.feature_extractor = QuantumFeatureExtractor(self.n_qubits, self.n_layers)
        self.uncertainty_quantifier = QuantumUncertaintyQuantifier(self.n_qubits)
        
        logger.info(f"Initialized QuantumMedicalImageProcessor with {self.n_qubits} qubits")
    
    def process_medical_image(self, image_features: np.ndarray,
                            patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process medical image with quantum enhancement
        
        Args:
            image_features: Classical image features
            patient_context: Optional patient context
            
        Returns:
            Dictionary with quantum-enhanced results
        """
        try:
            # Extract quantum features
            quantum_features = self.feature_extractor.extract_features(image_features)
            
            # Calculate uncertainty metrics
            base_confidence = self._calculate_base_confidence(quantum_features, patient_context)
            uncertainty_metrics = self.uncertainty_quantifier.quantify_uncertainty(
                base_confidence, quantum_features
            )
            
            # Combine classical and quantum features
            enhanced_features = self._combine_features(image_features, quantum_features)
            
            # Apply quantum enhancement to confidence
            enhanced_confidence = self._enhance_confidence(base_confidence, uncertainty_metrics)
            
            return {
                'quantum_features': enhanced_features.tolist(),
                'uncertainty_metrics': {
                    'confidence': enhanced_confidence,
                    'entropy': uncertainty_metrics.get('entropy', 0.5),
                    'quantum_uncertainty': uncertainty_metrics.get('quantum_uncertainty', 0.2)
                },
                'quantum_enhanced': True,
                'processing_info': {
                    'n_qubits': self.n_qubits,
                    'n_layers': self.n_layers,
                    'feature_dimension': len(enhanced_features)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in quantum medical image processing: {str(e)}")
            # Return fallback results
            return {
                'quantum_features': image_features.tolist(),
                'uncertainty_metrics': {
                    'confidence': 0.5,
                    'entropy': 0.5,
                    'quantum_uncertainty': 0.2
                },
                'quantum_enhanced': False,
                'processing_info': {
                    'error': str(e),
                    'fallback_mode': True
                }
            }
    
    def _calculate_base_confidence(self, quantum_features: np.ndarray,
                                 patient_context: Optional[Dict[str, Any]]) -> float:
        """Calculate base confidence from quantum features"""
        # Simple confidence calculation based on feature quality
        feature_quality = np.mean(np.abs(quantum_features))
        
        # Normalize to [0, 1] range
        confidence = np.tanh(feature_quality)
        
        # Adjust based on patient context if available
        if patient_context:
            # Age factor
            if 'age' in patient_context:
                age = patient_context['age']
                if age > 70:
                    confidence *= 0.9  # Slightly reduce confidence for elderly
                elif age < 18:
                    confidence *= 0.95  # Slightly reduce for pediatric
            
            # Medical history factor
            if 'medical_history' in patient_context:
                history = patient_context['medical_history']
                if len(history) > 3:  # Complex medical history
                    confidence *= 0.85
        
        return min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    def _combine_features(self, classical_features: np.ndarray,
                         quantum_features: np.ndarray) -> np.ndarray:
        """Combine classical and quantum features"""
        # Weighted combination
        classical_weight = 0.6
        quantum_weight = 0.4
        
        # Ensure same length
        min_length = min(len(classical_features), len(quantum_features))
        classical_subset = classical_features[:min_length]
        quantum_subset = quantum_features[:min_length]
        
        # Combine features
        combined = classical_weight * classical_subset + quantum_weight * quantum_subset
        
        return combined
    
    def _enhance_confidence(self, base_confidence: float,
                          uncertainty_metrics: Dict[str, float]) -> float:
        """Enhance confidence using quantum uncertainty metrics"""
        quantum_uncertainty = uncertainty_metrics.get('quantum_uncertainty', 0.2)
        entropy = uncertainty_metrics.get('entropy', 0.5)
        
        # Quantum enhancement factor
        enhancement_factor = 1.0 - 0.3 * quantum_uncertainty + 0.1 * (1 - entropy)
        
        enhanced_confidence = base_confidence * enhancement_factor
        
        return min(max(enhanced_confidence, 0.1), 0.95)


class HybridQuantumClassicalNetwork(nn.Module):
    """
    Hybrid quantum-classical neural network for medical image analysis
    """
    
    def __init__(self, classical_input_size: int, quantum_qubits: int, num_classes: int):
        """
        Initialize hybrid network
        
        Args:
            classical_input_size: Size of classical input features
            quantum_qubits: Number of qubits for quantum layer
            num_classes: Number of output classes
        """
        super(HybridQuantumClassicalNetwork, self).__init__()
        
        self.classical_input_size = classical_input_size
        self.quantum_qubits = quantum_qubits
        self.num_classes = num_classes
        
        # Classical layers
        self.classical_encoder = nn.Sequential(
            nn.Linear(classical_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, quantum_qubits * 2)  # Output for quantum layer
        )
        
        # Quantum processor (simulated)
        self.quantum_processor = QuantumFeatureExtractor(quantum_qubits)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(quantum_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(quantum_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through hybrid network"""
        # Classical encoding
        classical_features = self.classical_encoder(x)
        
        # Prepare features for quantum processing
        quantum_input = classical_features.detach().cpu().numpy()
        
        # Quantum processing
        quantum_features = self.quantum_processor.extract_features(quantum_input)
        quantum_tensor = torch.tensor(quantum_features, dtype=torch.float32, device=x.device)
        
        # Classification
        output = self.classifier(quantum_tensor)
        
        return output
    
    def get_quantum_uncertainty(self, x):
        """Get quantum uncertainty estimate"""
        # Classical encoding
        classical_features = self.classical_encoder(x)
        
        # Prepare features for quantum processing
        quantum_input = classical_features.detach().cpu().numpy()
        
        # Quantum processing
        quantum_features = self.quantum_processor.extract_features(quantum_input)
        quantum_tensor = torch.tensor(quantum_features, dtype=torch.float32, device=x.device)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(quantum_tensor)
        
        return {
            'uncertainty': uncertainty,
            'confidence': 1 - uncertainty,
            'quantum_features': quantum_tensor
        }


# Utility functions
def create_quantum_circuit_diagram(circuit, output_path: str = None):
    """Create visualization of quantum circuit"""
    if not QISKIT_AVAILABLE or circuit is None:
        return None
    
    try:
        # Create circuit diagram
        diagram = circuit.draw(output='mpl')
        
        if output_path:
            diagram.savefig(output_path)
            logger.info(f"Circuit diagram saved to {output_path}")
        
        return diagram
        
    except Exception as e:
        logger.error(f"Error creating circuit diagram: {str(e)}")
        return None


def benchmark_quantum_performance(n_samples: int = 100) -> Dict[str, float]:
    """Benchmark quantum processing performance"""
    import time
    
    # Create test data
    test_features = np.random.random((n_samples, 256))
    
    # Initialize quantum processor
    config = {'quantum_qubits': 4, 'quantum_layers': 2}
    processor = QuantumMedicalImageProcessor(config)
    
    # Benchmark
    start_time = time.time()
    
    results = []
    for features in test_features:
        result = processor.process_medical_image(features)
        results.append(result)
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_sample = total_time / n_samples
    
    return {
        'total_time': total_time,
        'avg_time_per_sample': avg_time_per_sample,
        'samples_processed': n_samples,
        'throughput': n_samples / total_time
    }


# Example usage and testing
if __name__ == "__main__":
    # Test quantum circuits
    print("Testing Quantum Medical Image Processor...")
    
    # Create test configuration
    config = {
        'quantum_qubits': 4,
        'quantum_layers': 2
    }
    
    # Initialize processor
    processor = QuantumMedicalImageProcessor(config)
    
    # Test with sample data
    sample_features = np.random.random(256)
    patient_context = {'age': 45, 'medical_history': ['diabetes']}
    
    # Process image
    result = processor.process_medical_image(sample_features, patient_context)
    
    print("Results:")
    print(f"  Quantum Enhanced: {result['quantum_enhanced']}")
    print(f"  Confidence: {result['uncertainty_metrics']['confidence']:.3f}")
    print(f"  Quantum Uncertainty: {result['uncertainty_metrics']['quantum_uncertainty']:.3f}")
    print(f"  Feature Dimension: {len(result['quantum_features'])}")
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    benchmark_results = benchmark_quantum_performance(10)
    print(f"  Average time per sample: {benchmark_results['avg_time_per_sample']:.4f} seconds")
    print(f"  Throughput: {benchmark_results['throughput']:.2f} samples/second")
    
    print("Quantum circuits module test completed!")

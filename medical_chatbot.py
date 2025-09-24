"""
Medical Chatbot with OpenAI GPT-4 Vision Integration
Doctor-perspective chatbot for medical consultation and image analysis
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import base64
from io import BytesIO
from PIL import Image
import numpy as np

try:
    import openai
    from openai import OpenAI
except ImportError:
    logging.warning("OpenAI library not available")
    openai = None
    OpenAI = None

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from medical_analyzer import MedicalImageAnalyzer, PatientContext, AnalysisResult
from llava_medical import MedicalTerminologyProcessor

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    message_id: str
    image_data: Optional[str] = None
    analysis_data: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None


@dataclass
class ChatSession:
    """Represents a complete chat session"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[ChatMessage]
    patient_context: Optional[PatientContext] = None
    session_metadata: Optional[Dict[str, Any]] = None


class MedicalKnowledgeBase:
    """
    Medical knowledge base for contextual information
    """
    
    def __init__(self):
        """Initialize medical knowledge base"""
        self.knowledge = self._load_medical_knowledge()
        self.terminology_processor = MedicalTerminologyProcessor()
        
        logger.info("Initialized MedicalKnowledgeBase")
    
    def _load_medical_knowledge(self) -> Dict[str, Any]:
        """Load comprehensive medical knowledge"""
        return {
            'medical_specialties': {
                'radiology': {
                    'description': 'Medical imaging and interpretation',
                    'common_procedures': ['X-ray', 'CT', 'MRI', 'Ultrasound', 'PET'],
                    'common_findings': ['Normal', 'Abnormal opacity', 'Mass', 'Fracture']
                },
                'emergency_medicine': {
                    'description': 'Acute medical care',
                    'common_presentations': ['Chest pain', 'Dyspnea', 'Trauma', 'Abdominal pain'],
                    'urgency_levels': ['Emergency', 'Urgent', 'Semi-urgent', 'Non-urgent']
                },
                'internal_medicine': {
                    'description': 'General adult medical care',
                    'common_conditions': ['Diabetes', 'Hypertension', 'COPD', 'CAD'],
                    'diagnostic_approaches': ['History', 'Physical exam', 'Laboratory', 'Imaging']
                }
            },
            'diagnostic_criteria': {
                'pneumonia': {
                    'clinical_features': ['Fever', 'Cough', 'Dyspnea', 'Chest pain'],
                    'imaging_findings': ['Consolidation', 'Air bronchograms', 'Pleural effusion'],
                    'laboratory_findings': ['Elevated WBC', 'Positive cultures']
                },
                'myocardial_infarction': {
                    'clinical_features': ['Chest pain', 'Dyspnea', 'Diaphoresis', 'Nausea'],
                    'ecg_findings': ['ST elevation', 'Q waves', 'T wave inversion'],
                    'laboratory_findings': ['Elevated troponins', 'Elevated CK-MB']
                }
            },
            'treatment_guidelines': {
                'antibiotics': {
                    'indications': ['Bacterial infections', 'Prophylaxis'],
                    'contraindications': ['Allergy', 'Viral infections'],
                    'monitoring': ['Renal function', 'Hepatic function']
                },
                'imaging_protocols': {
                    'ct_chest': {
                        'indications': ['Pulmonary embolism', 'Lung nodules', 'Trauma'],
                        'contraindications': ['Pregnancy', 'Contrast allergy'],
                        'preparation': ['IV contrast', 'Breath hold instructions']
                    }
                }
            }
        }
    
    def get_medical_context(self, query: str, specialty: Optional[str] = None) -> Dict[str, Any]:
        """
        Get relevant medical context for a query
        
        Args:
            query: Medical query
            specialty: Medical specialty context
            
        Returns:
            Relevant medical context
        """
        context = {'relevant_knowledge': []}
        
        # Extract medical entities from query
        entities = self.terminology_processor.extract_medical_entities(query)
        
        # Add specialty-specific context
        if specialty and specialty in self.knowledge['medical_specialties']:
            context['specialty_info'] = self.knowledge['medical_specialties'][specialty]
        
        # Add relevant diagnostic criteria
        query_lower = query.lower()
        for condition, criteria in self.knowledge['diagnostic_criteria'].items():
            if condition in query_lower:
                context['diagnostic_criteria'] = {condition: criteria}
                break
        
        # Add relevant treatment guidelines
        for treatment, guidelines in self.knowledge['treatment_guidelines'].items():
            if treatment in query_lower:
                context['treatment_guidelines'] = {treatment: guidelines}
                break
        
        context['extracted_entities'] = entities
        
        return context


class MedicalChatbot:
    """
    Doctor-perspective medical chatbot with OpenAI GPT-4 Vision integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Medical Chatbot
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.openai_api_key = config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        
        # Initialize OpenAI client
        if self.openai_api_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI client not available - using mock responses")
        
        # Initialize medical components
        self.medical_analyzer = MedicalImageAnalyzer(config)
        self.knowledge_base = MedicalKnowledgeBase()
        
        # Chat session storage (in production, use Redis or database)
        self.active_sessions: Dict[str, ChatSession] = {}
        
        # System prompt for medical consultation
        self.system_prompt = self._create_system_prompt()
        
        logger.info("Initialized MedicalChatbot")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for medical consultation"""
        return """You are an advanced AI medical consultant assistant designed to help healthcare professionals with medical image analysis and clinical decision-making. 

Your role and capabilities:
- Provide doctor-level medical insights and analysis
- Analyze medical images using state-of-the-art AI including quantum-enhanced processing
- Offer differential diagnoses based on imaging findings and clinical context
- Suggest appropriate treatment protocols and follow-up recommendations
- Maintain the highest standards of medical accuracy and ethical practice

Guidelines for responses:
1. Always emphasize that AI analysis should complement, not replace, clinical judgment
2. Encourage correlation with patient history, physical examination, and other diagnostic tests
3. Provide confidence levels for your assessments when possible
4. Suggest appropriate follow-up actions and specialist consultations when indicated
5. Maintain professional medical terminology while ensuring clarity
6. Always consider patient safety as the primary concern
7. Acknowledge limitations and uncertainties in AI analysis

Ethical considerations:
- Never provide definitive diagnoses without appropriate clinical context
- Always recommend physician oversight for treatment decisions
- Respect patient privacy and confidentiality
- Adhere to medical best practices and evidence-based guidelines

When analyzing medical images:
- Describe relevant anatomical structures and their appearance
- Identify any abnormalities or pathological findings
- Assess image quality and technical factors
- Provide differential diagnoses ranked by likelihood
- Suggest additional imaging or tests if needed
- Include confidence scores for major findings

Remember: You are assisting healthcare professionals, not replacing them. Always emphasize the importance of clinical correlation and physician oversight."""
    
    def create_chat_session(self, user_id: str, patient_context: Optional[PatientContext] = None) -> str:
        """
        Create a new chat session
        
        Args:
            user_id: User identifier
            patient_context: Optional patient context
            
        Returns:
            Session ID
        """
        import uuid
        
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            messages=[],
            patient_context=patient_context,
            session_metadata={'created_with_patient_context': patient_context is not None}
        )
        
        self.active_sessions[session_id] = session
        
        # Add welcome message
        welcome_message = self._create_welcome_message(patient_context)
        self._add_message_to_session(session_id, ChatMessage(
            role='assistant',
            content=welcome_message,
            timestamp=current_time,
            message_id=str(uuid.uuid4())
        ))
        
        logger.info(f"Created chat session {session_id} for user {user_id}")
        return session_id
    
    def _create_welcome_message(self, patient_context: Optional[PatientContext]) -> str:
        """Create welcome message for new session"""
        base_message = """Welcome to the Quantum-Enhanced Medical Imaging AI Consultation System.
        
I'm here to assist you with medical image analysis and clinical decision-making. I can:

• Analyze medical images (X-ray, CT, MRI, etc.) with quantum-enhanced processing
• Provide differential diagnoses and clinical insights
• Generate confidence scores and uncertainty quantification
• Offer treatment recommendations and follow-up suggestions
• Create detailed medical reports with explainable AI visualizations

Please share your medical images and clinical questions. I'll provide comprehensive analysis while emphasizing the importance of clinical correlation."""
        
        if patient_context and patient_context.patient_id:
            base_message += f"\n\nI see we have patient context available (Patient ID: {patient_context.patient_id}). This will help provide more personalized analysis."
        
        return base_message
    
    def process_medical_query(self, session_id: str, query: str, 
                            image_data: Optional[str] = None,
                            image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process medical query with optional image
        
        Args:
            session_id: Chat session ID
            query: Medical query
            image_data: Base64 encoded image data
            image_path: Path to image file
            
        Returns:
            Response dictionary with analysis results
        """
        try:
            if session_id not in self.active_sessions:
                return {'error': 'Invalid session ID', 'session_id': session_id}
            
            session = self.active_sessions[session_id]
            session.last_activity = datetime.now()
            
            # Add user message to session
            import uuid
            user_message = ChatMessage(
                role='user',
                content=query,
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4()),
                image_data=image_data[:100] + '...truncated' if image_data else None  # Store truncated for logging
            )
            self._add_message_to_session(session_id, user_message)
            
            # Process image if provided
            image_analysis = None
            if image_data or image_path:
                image_analysis = self._process_image_with_query(
                    query, image_data, image_path, session.patient_context
                )
            
            # Generate response
            response_content = self._generate_medical_response(
                session, query, image_analysis
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_response_confidence(response_content, image_analysis)
            
            # Add assistant message to session
            assistant_message = ChatMessage(
                role='assistant',
                content=response_content,
                timestamp=datetime.now(),
                message_id=str(uuid.uuid4()),
                analysis_data=image_analysis,
                confidence_score=confidence_score
            )
            self._add_message_to_session(session_id, assistant_message)
            
            return {
                'session_id': session_id,
                'response': response_content,
                'confidence_score': confidence_score,
                'image_analysis': image_analysis is not None,
                'message_count': len(session.messages),
                'analysis_data': image_analysis
            }
            
        except Exception as e:
            logger.error(f"Error processing medical query: {str(e)}")
            return {
                'error': 'Failed to process query',
                'message': str(e),
                'session_id': session_id
            }
    
    def _process_image_with_query(self, query: str, image_data: Optional[str],
                                image_path: Optional[str], 
                                patient_context: Optional[PatientContext]) -> Optional[Dict[str, Any]]:
        """Process medical image with query"""
        try:
            # Handle image data
            temp_image_path = None
            
            if image_data:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                image = Image.open(BytesIO(image_bytes))
                
                # Save temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    image.save(temp_file.name)
                    temp_image_path = temp_file.name
            
            elif image_path:
                temp_image_path = image_path
            
            if not temp_image_path:
                return None
            
            # Analyze image using medical analyzer
            analysis_result = self.medical_analyzer.analyze_image(
                temp_image_path, query, patient_context
            )
            
            # Clean up temporary file if created
            if image_data and temp_image_path:
                os.unlink(temp_image_path)
            
            # Convert analysis result to dictionary
            analysis_dict = {
                'session_id': analysis_result.session_id,
                'imaging_modality': analysis_result.imaging_modality.value,
                'primary_findings': [
                    {
                        'finding_id': f.finding_id,
                        'description': f.description,
                        'confidence': f.confidence,
                        'recommendations': f.recommendations or []
                    }
                    for f in analysis_result.primary_findings
                ],
                'confidence_scores': analysis_result.confidence_scores,
                'recommendations': analysis_result.recommendations,
                'risk_stratification': analysis_result.risk_stratification,
                'quantum_enhanced': analysis_result.quantum_metrics.get('quantum_enhanced', False),
                'processing_time': analysis_result.processing_metadata.get('processing_time', 0)
            }
            
            return analysis_dict
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {'error': str(e)}
    
    def _generate_medical_response(self, session: ChatSession, query: str,
                                 image_analysis: Optional[Dict[str, Any]]) -> str:
        """Generate medical response using OpenAI or fallback logic"""
        
        # Prepare context
        medical_context = self.knowledge_base.get_medical_context(query)
        
        # Build conversation history
        conversation_history = self._build_conversation_history(session)
        
        if self.openai_client:
            return self._generate_openai_response(
                conversation_history, query, image_analysis, medical_context, session.patient_context
            )
        else:
            return self._generate_fallback_response(query, image_analysis, medical_context)
    
    def _build_conversation_history(self, session: ChatSession) -> List[Dict[str, str]]:
        """Build conversation history for OpenAI"""
        history = [{'role': 'system', 'content': self.system_prompt}]
        
        # Add recent messages (last 10 to manage token limit)
        recent_messages = session.messages[-10:]
        
        for message in recent_messages:
            if message.role in ['user', 'assistant']:
                history.append({
                    'role': message.role,
                    'content': message.content
                })
        
        return history
    
    def _generate_openai_response(self, conversation_history: List[Dict[str, str]],
                                query: str, image_analysis: Optional[Dict[str, Any]],
                                medical_context: Dict[str, Any],
                                patient_context: Optional[PatientContext]) -> str:
        """Generate response using OpenAI GPT-4"""
        try:
            # Enhance the current query with analysis results
            enhanced_query = query
            
            if image_analysis:
                enhanced_query += f"""

MEDICAL IMAGE ANALYSIS RESULTS:
- Imaging Modality: {image_analysis.get('imaging_modality', 'Unknown')}
- Quantum Enhancement: {'Enabled' if image_analysis.get('quantum_enhanced') else 'Disabled'}
- Overall Confidence: {image_analysis.get('confidence_scores', {}).get('overall_confidence', 0.5):.2f}

PRIMARY FINDINGS:
"""
                for finding in image_analysis.get('primary_findings', []):
                    enhanced_query += f"- {finding['description']} (Confidence: {finding['confidence']:.2f})\n"
                
                enhanced_query += f"\nRECOMMendations:\n"
                for rec in image_analysis.get('recommendations', []):
                    enhanced_query += f"- {rec}\n"
                
                enhanced_query += f"\nRisk Assessment: {image_analysis.get('risk_stratification', {}).get('risk_category', 'Unknown')}"
            
            # Add patient context if available
            if patient_context:
                context_info = []
                if patient_context.age:
                    context_info.append(f"Age: {patient_context.age}")
                if patient_context.gender:
                    context_info.append(f"Gender: {patient_context.gender}")
                if patient_context.medical_history:
                    context_info.append(f"Medical History: {', '.join(patient_context.medical_history)}")
                
                if context_info:
                    enhanced_query += f"\n\nPATIENT CONTEXT:\n{'; '.join(context_info)}"
            
            # Add medical knowledge context
            if medical_context.get('diagnostic_criteria'):
                enhanced_query += f"\n\nRELEVANT DIAGNOSTIC CRITERIA:\n{json.dumps(medical_context['diagnostic_criteria'], indent=2)}"
            
            # Update conversation history with enhanced query
            conversation_history[-1]['content'] = enhanced_query
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=conversation_history,
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more consistent medical responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            return self._generate_fallback_response(query, image_analysis, medical_context)
    
    def _generate_fallback_response(self, query: str, image_analysis: Optional[Dict[str, Any]],
                                  medical_context: Dict[str, Any]) -> str:
        """Generate fallback response when OpenAI is not available"""
        
        response_parts = []
        
        # Acknowledge the query
        response_parts.append(f"Thank you for your medical consultation query: '{query}'")
        
        # Include image analysis if available
        if image_analysis:
            response_parts.append("\n**MEDICAL IMAGE ANALYSIS:**")
            
            modality = image_analysis.get('imaging_modality', 'Unknown')
            response_parts.append(f"- Imaging Modality: {modality}")
            
            confidence = image_analysis.get('confidence_scores', {}).get('overall_confidence', 0.5)
            response_parts.append(f"- Overall Analysis Confidence: {confidence:.2f} ({self._categorize_confidence_text(confidence)})")
            
            if image_analysis.get('quantum_enhanced'):
                response_parts.append("- Quantum Enhancement: Applied for improved accuracy")
            
            # Primary findings
            findings = image_analysis.get('primary_findings', [])
            if findings:
                response_parts.append("\n**PRIMARY FINDINGS:**")
                for i, finding in enumerate(findings[:3], 1):  # Limit to top 3 findings
                    response_parts.append(f"{i}. {finding['description']} (Confidence: {finding['confidence']:.2f})")
            
            # Recommendations
            recommendations = image_analysis.get('recommendations', [])
            if recommendations:
                response_parts.append("\n**CLINICAL RECOMMENDATIONS:**")
                for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5 recommendations
                    response_parts.append(f"{i}. {rec}")
            
            # Risk assessment
            risk_info = image_analysis.get('risk_stratification', {})
            if risk_info:
                risk_category = risk_info.get('risk_category', 'Unknown')
                response_parts.append(f"\n**RISK ASSESSMENT:** {risk_category} Risk")
        
        # Add medical context insights
        if medical_context.get('extracted_entities'):
            entities = medical_context['extracted_entities']
            relevant_terms = []
            for category, terms in entities.items():
                if terms:
                    relevant_terms.extend(terms)
            
            if relevant_terms:
                response_parts.append(f"\n**RELEVANT MEDICAL TERMS IDENTIFIED:** {', '.join(relevant_terms[:5])}")
        
        # Standard medical disclaimer
        response_parts.append("""
        
**IMPORTANT MEDICAL DISCLAIMER:**
This AI analysis is designed to assist healthcare professionals and should not replace clinical judgment. Please ensure:
- Correlation with patient history and physical examination
- Consideration of clinical context and symptoms  
- Appropriate specialist consultation when indicated
- Follow-up imaging or testing as clinically warranted

The analysis confidence scores reflect the AI system's certainty and should be interpreted alongside clinical expertise.""")
        
        return "\n".join(response_parts)
    
    def _categorize_confidence_text(self, confidence: float) -> str:
        """Convert confidence score to text description"""
        if confidence >= 0.8:
            return "High Confidence"
        elif confidence >= 0.6:
            return "Moderate Confidence"
        elif confidence >= 0.4:
            return "Low Confidence"
        else:
            return "Very Low Confidence"
    
    def _calculate_response_confidence(self, response_content: str, 
                                     image_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for response"""
        base_confidence = 0.7
        
        # Adjust based on image analysis availability
        if image_analysis:
            analysis_confidence = image_analysis.get('confidence_scores', {}).get('overall_confidence', 0.5)
            base_confidence = (base_confidence + analysis_confidence) / 2
        
        # Adjust based on response content certainty indicators
        response_lower = response_content.lower()
        
        # High confidence indicators
        high_confidence_words = ['clearly', 'definitely', 'evident', 'obvious', 'consistent with']
        low_confidence_words = ['possibly', 'might', 'could', 'uncertain', 'unclear', 'consider']
        
        high_count = sum(1 for word in high_confidence_words if word in response_lower)
        low_count = sum(1 for word in low_confidence_words if word in response_lower)
        
        if high_count > low_count:
            base_confidence = min(0.95, base_confidence + 0.05 * high_count)
        elif low_count > high_count:
            base_confidence = max(0.3, base_confidence - 0.05 * low_count)
        
        return base_confidence
    
    def _add_message_to_session(self, session_id: str, message: ChatMessage):
        """Add message to session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].messages.append(message)
    
    def get_session_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chat session history
        
        Args:
            session_id: Session ID
            
        Returns:
            Session history or None if not found
        """
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'message_count': len(session.messages),
            'messages': [
                {
                    'message_id': msg.message_id,
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'has_image': msg.image_data is not None,
                    'has_analysis': msg.analysis_data is not None,
                    'confidence_score': msg.confidence_score
                }
                for msg in session.messages
            ],
            'patient_context': asdict(session.patient_context) if session.patient_context else None
        }
    
    def update_patient_context(self, session_id: str, patient_context: PatientContext) -> bool:
        """
        Update patient context for session
        
        Args:
            session_id: Session ID
            patient_context: Updated patient context
            
        Returns:
            Success status
        """
        if session_id not in self.active_sessions:
            return False
        
        self.active_sessions[session_id].patient_context = patient_context
        self.active_sessions[session_id].last_activity = datetime.now()
        
        return True
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old chat sessions
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        old_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session.last_activity < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self.active_sessions[session_id]
        
        logger.info(f"Cleaned up {len(old_sessions)} old chat sessions")
        return len(old_sessions)
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics for all sessions"""
        total_sessions = len(self.active_sessions)
        total_messages = sum(len(session.messages) for session in self.active_sessions.values())
        
        # Calculate average confidence scores
        confidence_scores = []
        image_analyses = 0
        
        for session in self.active_sessions.values():
            for message in session.messages:
                if message.confidence_score:
                    confidence_scores.append(message.confidence_score)
                if message.analysis_data:
                    image_analyses += 1
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_active_sessions': total_sessions,
            'total_messages': total_messages,
            'total_image_analyses': image_analyses,
            'average_confidence_score': avg_confidence,
            'openai_integration': self.openai_client is not None,
            'timestamp': datetime.now().isoformat()
        }
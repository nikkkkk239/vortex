# Quantum-Enhanced Medical Imaging AI System Configuration

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json


@dataclass
class FlaskConfig:
    """Flask application configuration"""
    ENV: str = 'development'
    DEBUG: bool = True
    HOST: str = '0.0.0.0'
    PORT: int = 5000
    SECRET_KEY: str = 'dev-secret-key-change-in-production'
    

@dataclass
class JWTConfig:
    """JWT configuration"""
    SECRET_KEY: str = 'jwt-secret-key-change-in-production'
    ACCESS_TOKEN_EXPIRES: int = 3600  # 1 hour
    

@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    API_KEY: str = ''
    MODEL: str = 'gpt-4-vision-preview'
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.1
    

@dataclass
class LLaVAConfig:
    """LLaVA model configuration"""
    MODEL_PATH: str = 'microsoft/llava-med-v1.5-mistral-7b'
    MODEL_NAME: str = 'llava-v1.5-7b'
    DEVICE: str = 'cuda'
    LOAD_8BIT: bool = False
    LOAD_4BIT: bool = False
    

@dataclass
class QuantumConfig:
    """Quantum computing configuration"""
    BACKEND: str = 'qasm_simulator'
    SHOTS: int = 1000
    OPTIMIZATION_LEVEL: int = 1
    SEED: int = 42
    

@dataclass
class FileConfig:
    """File handling configuration"""
    UPLOAD_FOLDER: str = './uploads'
    MAX_CONTENT_LENGTH: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: Optional[set] = None
    MAX_IMAGE_SIZE: int = 1024
    CONFIDENCE_THRESHOLD: float = 0.7
    
    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'dcm', 'nii', 'nii.gz'}


@dataclass
class DatabaseConfig:
    """Database configuration"""
    URL: str = 'sqlite:///medical_analysis.db'
    TRACK_MODIFICATIONS: bool = False
    

@dataclass
class RedisConfig:
    """Redis configuration"""
    URL: str = 'redis://localhost:6379/0'
    CACHE_TYPE: str = 'redis'
    

@dataclass
class SecurityConfig:
    """Security configuration"""
    CORS_ORIGINS: Optional[list] = None
    RATE_LIMIT_GLOBAL: str = '1000 per hour'
    RATE_LIMIT_ANALYZE: str = '10 per minute'
    RATE_LIMIT_CHAT: str = '20 per minute'
    ENCRYPT_PATIENT_DATA: bool = True
    ENCRYPTION_KEY: str = 'change-this-32-character-key-now!'
    AUDIT_LOG_ENABLED: bool = True
    SESSION_TIMEOUT: int = 1800  # 30 minutes
    
    def __post_init__(self):
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8080']


@dataclass
class LoggingConfig:
    """Logging configuration"""
    LEVEL: str = 'INFO'
    FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    FILE: str = 'logs/medical_analysis.log'
    

@dataclass
class ReportConfig:
    """Report generation configuration"""
    OUTPUT_DIR: str = './reports'
    TEMPLATE_DIR: str = './templates/reports'
    CLINIC_NAME: str = 'Quantum Medical Imaging Center'
    CLINIC_ADDRESS: str = '123 Medical Center Drive, Healthcare City, HC 12345'
    CLINIC_PHONE: str = '+1 (555) 123-4567'
    CLINIC_EMAIL: str = 'info@quantummedical.com'
    

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    SENTRY_DSN: str = ''
    PROMETHEUS_METRICS_ENABLED: bool = True
    HEALTH_CHECK_ENABLED: bool = True
    

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    CELERY_BROKER_URL: str = 'redis://localhost:6379/2'
    CELERY_RESULT_BACKEND: str = 'redis://localhost:6379/3'
    WORKER_PROCESSES: int = 4
    WORKER_THREADS: int = 2


class ConfigManager:
    """
    Configuration manager for the Quantum-Enhanced Medical Imaging AI System
    """
    
    def __init__(self, env_file: str = '.env'):
        """
        Initialize configuration manager
        
        Args:
            env_file: Path to environment file
        """
        self.env_file = env_file
        self._load_environment()
        
        # Initialize configuration sections
        self.flask = self._create_flask_config()
        self.jwt = self._create_jwt_config()
        self.openai = self._create_openai_config()
        self.llava = self._create_llava_config()
        self.quantum = self._create_quantum_config()
        self.file = self._create_file_config()
        self.database = self._create_database_config()
        self.redis = self._create_redis_config()
        self.security = self._create_security_config()
        self.logging = self._create_logging_config()
        self.report = self._create_report_config()
        self.monitoring = self._create_monitoring_config()
        self.performance = self._create_performance_config()
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    
    def _get_env(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        """Get environment variable with type casting"""
        value = os.environ.get(key, default)
        
        if value is None:
            return default
        
        if cast_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif cast_type == int:
            try:
                return int(value)
            except ValueError:
                return default
        elif cast_type == float:
            try:
                return float(value)
            except ValueError:
                return default
        elif cast_type == list:
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            return value
        
        return cast_type(value)
    
    def _create_flask_config(self) -> FlaskConfig:
        """Create Flask configuration"""
        return FlaskConfig(
            ENV=self._get_env('FLASK_ENV', 'development'),
            DEBUG=self._get_env('FLASK_DEBUG', True, bool),
            HOST=self._get_env('FLASK_HOST', '0.0.0.0'),
            PORT=self._get_env('FLASK_PORT', 5000, int),
            SECRET_KEY=self._get_env('SECRET_KEY', 'dev-secret-key-change-in-production')
        )
    
    def _create_jwt_config(self) -> JWTConfig:
        """Create JWT configuration"""
        return JWTConfig(
            SECRET_KEY=self._get_env('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production'),
            ACCESS_TOKEN_EXPIRES=self._get_env('JWT_ACCESS_TOKEN_EXPIRES', 3600, int)
        )
    
    def _create_openai_config(self) -> OpenAIConfig:
        """Create OpenAI configuration"""
        return OpenAIConfig(
            API_KEY=self._get_env('OPENAI_API_KEY', ''),
            MODEL=self._get_env('OPENAI_MODEL', 'gpt-4-vision-preview'),
            MAX_TOKENS=self._get_env('OPENAI_MAX_TOKENS', 4096, int),
            TEMPERATURE=self._get_env('OPENAI_TEMPERATURE', 0.1, float)
        )
    
    def _create_llava_config(self) -> LLaVAConfig:
        """Create LLaVA configuration"""
        return LLaVAConfig(
            MODEL_PATH=self._get_env('LLAVA_MODEL_PATH', 'microsoft/llava-med-v1.5-mistral-7b'),
            MODEL_NAME=self._get_env('LLAVA_MODEL_NAME', 'llava-v1.5-7b'),
            DEVICE=self._get_env('LLAVA_DEVICE', 'cuda'),
            LOAD_8BIT=self._get_env('LLAVA_LOAD_8BIT', False, bool),
            LOAD_4BIT=self._get_env('LLAVA_LOAD_4BIT', False, bool)
        )
    
    def _create_quantum_config(self) -> QuantumConfig:
        """Create Quantum configuration"""
        return QuantumConfig(
            BACKEND=self._get_env('QUANTUM_BACKEND', 'qasm_simulator'),
            SHOTS=self._get_env('QUANTUM_SHOTS', 1000, int),
            OPTIMIZATION_LEVEL=self._get_env('QUANTUM_OPTIMIZATION_LEVEL', 1, int),
            SEED=self._get_env('QUANTUM_SEED', 42, int)
        )
    
    def _create_file_config(self) -> FileConfig:
        """Create File configuration"""
        allowed_extensions = self._get_env('ALLOWED_EXTENSIONS', 'jpg,jpeg,png,dcm,nii,nii.gz', list)
        return FileConfig(
            UPLOAD_FOLDER=self._get_env('UPLOAD_FOLDER', './uploads'),
            MAX_CONTENT_LENGTH=self._get_env('MAX_CONTENT_LENGTH', 50485760, int),
            ALLOWED_EXTENSIONS=set(allowed_extensions),
            MAX_IMAGE_SIZE=self._get_env('MAX_IMAGE_SIZE', 1024, int),
            CONFIDENCE_THRESHOLD=self._get_env('CONFIDENCE_THRESHOLD', 0.7, float)
        )
    
    def _create_database_config(self) -> DatabaseConfig:
        """Create Database configuration"""
        return DatabaseConfig(
            URL=self._get_env('DATABASE_URL', 'sqlite:///medical_analysis.db'),
            TRACK_MODIFICATIONS=self._get_env('SQLALCHEMY_TRACK_MODIFICATIONS', False, bool)
        )
    
    def _create_redis_config(self) -> RedisConfig:
        """Create Redis configuration"""
        return RedisConfig(
            URL=self._get_env('REDIS_URL', 'redis://localhost:6379/0'),
            CACHE_TYPE=self._get_env('CACHE_TYPE', 'redis')
        )
    
    def _create_security_config(self) -> SecurityConfig:
        """Create Security configuration"""
        cors_origins = self._get_env('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080', list)
        return SecurityConfig(
            CORS_ORIGINS=cors_origins,
            RATE_LIMIT_GLOBAL=self._get_env('RATE_LIMIT_GLOBAL', '1000 per hour'),
            RATE_LIMIT_ANALYZE=self._get_env('RATE_LIMIT_ANALYZE', '10 per minute'),
            RATE_LIMIT_CHAT=self._get_env('RATE_LIMIT_CHAT', '20 per minute'),
            ENCRYPT_PATIENT_DATA=self._get_env('ENCRYPT_PATIENT_DATA', True, bool),
            ENCRYPTION_KEY=self._get_env('ENCRYPTION_KEY', 'change-this-32-character-key-now!'),
            AUDIT_LOG_ENABLED=self._get_env('AUDIT_LOG_ENABLED', True, bool),
            SESSION_TIMEOUT=self._get_env('SESSION_TIMEOUT', 1800, int)
        )
    
    def _create_logging_config(self) -> LoggingConfig:
        """Create Logging configuration"""
        return LoggingConfig(
            LEVEL=self._get_env('LOG_LEVEL', 'INFO'),
            FORMAT=self._get_env('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            FILE=self._get_env('LOG_FILE', 'logs/medical_analysis.log')
        )
    
    def _create_report_config(self) -> ReportConfig:
        """Create Report configuration"""
        return ReportConfig(
            OUTPUT_DIR=self._get_env('REPORT_OUTPUT_DIR', './reports'),
            TEMPLATE_DIR=self._get_env('REPORT_TEMPLATE_DIR', './templates/reports'),
            CLINIC_NAME=self._get_env('CLINIC_NAME', 'Quantum Medical Imaging Center'),
            CLINIC_ADDRESS=self._get_env('CLINIC_ADDRESS', '123 Medical Center Drive, Healthcare City, HC 12345'),
            CLINIC_PHONE=self._get_env('CLINIC_PHONE', '+1 (555) 123-4567'),
            CLINIC_EMAIL=self._get_env('CLINIC_EMAIL', 'info@quantummedical.com')
        )
    
    def _create_monitoring_config(self) -> MonitoringConfig:
        """Create Monitoring configuration"""
        return MonitoringConfig(
            SENTRY_DSN=self._get_env('SENTRY_DSN', ''),
            PROMETHEUS_METRICS_ENABLED=self._get_env('PROMETHEUS_METRICS_ENABLED', True, bool),
            HEALTH_CHECK_ENABLED=self._get_env('HEALTH_CHECK_ENABLED', True, bool)
        )
    
    def _create_performance_config(self) -> PerformanceConfig:
        """Create Performance configuration"""
        return PerformanceConfig(
            CELERY_BROKER_URL=self._get_env('CELERY_BROKER_URL', 'redis://localhost:6379/2'),
            CELERY_RESULT_BACKEND=self._get_env('CELERY_RESULT_BACKEND', 'redis://localhost:6379/3'),
            WORKER_PROCESSES=self._get_env('WORKER_PROCESSES', 4, int),
            WORKER_THREADS=self._get_env('WORKER_THREADS', 2, int)
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.logging.FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.LEVEL.upper()),
            format=self.logging.FORMAT,
            handlers=[
                logging.FileHandler(self.logging.FILE),
                logging.StreamHandler()
            ]
        )
    
    def _validate_config(self):
        """Validate configuration"""
        logger = logging.getLogger(__name__)
        
        # Check required directories
        required_dirs = [
            self.file.UPLOAD_FOLDER,
            self.report.OUTPUT_DIR,
            Path(self.logging.FILE).parent
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate OpenAI API key
        if not self.openai.API_KEY or self.openai.API_KEY == 'your-openai-api-key-here':
            logger.warning("OpenAI API key not configured - chatbot functionality will be limited")
        
        # Validate security settings
        if self.flask.ENV == 'production':
            if self.flask.SECRET_KEY == 'dev-secret-key-change-in-production':
                logger.error("Production environment detected but SECRET_KEY is still default!")
                raise ValueError("Change SECRET_KEY in production environment")
            
            if self.jwt.SECRET_KEY == 'jwt-secret-key-change-in-production':
                logger.error("Production environment detected but JWT_SECRET_KEY is still default!")
                raise ValueError("Change JWT_SECRET_KEY in production environment")
        
        logger.info("Configuration validation completed")
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask-compatible configuration dictionary"""
        return {
            'SECRET_KEY': self.flask.SECRET_KEY,
            'JWT_SECRET_KEY': self.jwt.SECRET_KEY,
            'JWT_ACCESS_TOKEN_EXPIRES': self.jwt.ACCESS_TOKEN_EXPIRES,
            'UPLOAD_FOLDER': self.file.UPLOAD_FOLDER,
            'MAX_CONTENT_LENGTH': self.file.MAX_CONTENT_LENGTH,
            'SQLALCHEMY_DATABASE_URI': self.database.URL,
            'SQLALCHEMY_TRACK_MODIFICATIONS': self.database.TRACK_MODIFICATIONS,
            'CACHE_TYPE': self.redis.CACHE_TYPE,
            'CACHE_REDIS_URL': self.redis.URL,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'flask': self.flask.__dict__,
            'jwt': self.jwt.__dict__,
            'openai': self.openai.__dict__,
            'llava': self.llava.__dict__,
            'quantum': self.quantum.__dict__,
            'file': self.file.__dict__,
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'security': self.security.__dict__,
            'logging': self.logging.__dict__,
            'report': self.report.__dict__,
            'monitoring': self.monitoring.__dict__,
            'performance': self.performance.__dict__
        }
    
    def save_config_file(self, output_path: str = 'config.json'):
        """Save current configuration to JSON file"""
        config_dict = self.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration saved to {output_path}")


# Global configuration instance
config = ConfigManager()


# Configuration validation functions
def validate_production_config():
    """Validate configuration for production deployment"""
    errors = []
    warnings = []
    
    if config.flask.ENV == 'production':
        if config.flask.SECRET_KEY == 'dev-secret-key-change-in-production':
            errors.append("SECRET_KEY must be changed in production")
        
        if config.jwt.SECRET_KEY == 'jwt-secret-key-change-in-production':
            errors.append("JWT_SECRET_KEY must be changed in production")
        
        if config.security.ENCRYPTION_KEY == 'change-this-32-character-key-now!':
            errors.append("ENCRYPTION_KEY must be changed in production")
        
        if not config.openai.API_KEY:
            warnings.append("OpenAI API key not configured - chatbot functionality will be limited")
        
        if config.flask.DEBUG:
            warnings.append("DEBUG mode is enabled in production")
    
    return errors, warnings


def create_default_config_file():
    """Create default configuration file from environment"""
    config.save_config_file('default_config.json')
    print("Default configuration saved to default_config.json")


if __name__ == "__main__":
    # Create default configuration file
    create_default_config_file()
    
    # Validate configuration
    errors, warnings = validate_production_config()
    
    if warnings:
        print("Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed!")
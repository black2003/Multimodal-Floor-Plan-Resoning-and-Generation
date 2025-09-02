"""
Configuration settings for the application
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Model settings
    MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', './models/cache')
    DEVICE = os.environ.get('DEVICE', 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    
    # Image processing settings
    IMAGE_SIZE = (224, 224)
    MAX_IMAGE_SIZE = (1024, 1024)
    
    # Text processing settings
    MAX_TEXT_LENGTH = 512
    VOCAB_SIZE = 30522  # DistilBERT vocab size
    
    # Graph settings
    MAX_NODES = 100
    MAX_EDGES = 200
    NODE_FEATURE_DIM = 64
    EDGE_FEATURE_DIM = 32
    
    # Fusion settings
    FUSION_HIDDEN_DIM = 768
    NUM_ATTENTION_HEADS = 12
    NUM_FUSION_LAYERS = 6
    
    # Task-specific settings
    VQA_MAX_ANSWER_LENGTH = 50
    RETRIEVAL_TOP_K = 10
    VALIDITY_THRESHOLD = 0.7
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Dataset settings
    DATASET_PATH = os.environ.get('DATASET_PATH', './data')
    PROCESSED_DATA_PATH = os.environ.get('PROCESSED_DATA_PATH', './data/processed')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', './logs/app.log')
    
    # WebSocket settings
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        Path(Config.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.DATASET_PATH).mkdir(parents=True, exist_ok=True)
        Path(Config.PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(Config.LOG_FILE)).mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

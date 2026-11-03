"""Application configuration - separate from ML config"""
import os

class AppConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'predictions.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    TEMP_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    
    # Model checkpoints - use the project's checkpoint directory
    CHECKPOINT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'checkpoints')
    
    # Ensure directories exist
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)
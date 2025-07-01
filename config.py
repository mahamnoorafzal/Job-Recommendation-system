import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask Configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev_key_please_change')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Database Configuration
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DB = os.getenv('MONGODB_DB', 'job_prediction_db')
    
    # Debug Configuration
    DEBUG = True
    
    # Allowed file extensions for uploads
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

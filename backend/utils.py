# Utility functions for the Flask application

import os
import uuid
from werkzeug.utils import secure_filename
from config import Config

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """Generate a unique filename while preserving the extension"""
    file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    unique_id = str(uuid.uuid4())
    return f"{unique_id}.{file_extension}" if file_extension else unique_id

def ensure_upload_folder():
    """Ensure the upload folder exists"""
    if not os.path.exists(Config.UPLOAD_FOLDER):
        os.makedirs(Config.UPLOAD_FOLDER)
    return Config.UPLOAD_FOLDER

def save_uploaded_file(file):
    """Save uploaded file to the upload folder"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        upload_folder = ensure_upload_folder()
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        return {
            'original_filename': filename,
            'saved_filename': unique_filename,
            'file_path': file_path,
            'success': True
        }
    return {'success': False, 'error': 'Invalid file type or no file provided'}

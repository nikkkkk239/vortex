# Vortex Flask Backend

A Python Flask backend application for the Vortex project, designed to handle image uploads and API endpoints.

## Project Structure

```
backend/
├── app.py                 # Main Flask application entry point
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── env.example          # Environment variables template
├── routes/              # API route blueprints
│   ├── __init__.py
│   └── api.py           # Main API routes
└── uploads/             # Directory for uploaded files (created automatically)
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### 2. Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Configuration

1. Copy the environment template:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` file with your configuration:
   ```env
   FLASK_DEBUG=True
   SECRET_KEY=your-secret-key-here
   PORT=5000
   CORS_ORIGINS=http://localhost:3000,http://localhost:5173
   ```

### 4. Running the Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. The server will start on `http://localhost:5000` (or the port specified in your `.env` file)

3. Test the API by visiting:
   - Health check: `http://localhost:5000/health`
   - API test: `http://localhost:5000/api/test`

## API Endpoints

### Health Check
- **GET** `/health` - Returns server health status

### API Routes
- **GET** `/api/test` - Test endpoint to verify API is working
- **POST** `/api/upload` - Upload image files (placeholder implementation)
- **GET** `/api/images` - Get list of uploaded images (placeholder)
- **GET** `/api/images/<image_id>` - Get specific image (placeholder)
- **DELETE** `/api/images/<image_id>` - Delete specific image (placeholder)

## Development

### Adding New Routes

1. Create new route functions in `routes/api.py`
2. Use the `@api_bp.route()` decorator
3. Import and register the blueprint in `app.py` if creating new blueprints

### Configuration

- Modify `config.py` for application-wide settings
- Use environment variables for sensitive data
- Update `requirements.txt` when adding new dependencies

### File Uploads

The application includes utility functions in `utils.py` for handling file uploads:
- `allowed_file()` - Check file extensions
- `generate_unique_filename()` - Create unique filenames
- `save_uploaded_file()` - Save files to upload directory

## Production Deployment

1. Set `FLASK_DEBUG=False` in your environment
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
3. Configure proper CORS origins for your frontend domain
4. Set a strong `SECRET_KEY` in production

## Troubleshooting

- **Import errors**: Make sure you're in the backend directory and virtual environment is activated
- **Port already in use**: Change the PORT in your `.env` file
- **CORS issues**: Update `CORS_ORIGINS` in your `.env` file to include your frontend URL
- **File upload issues**: Ensure the uploads directory has proper write permissions

## Next Steps

This is a basic Flask structure. You can extend it by:
- Adding a database (SQLAlchemy)
- Implementing actual file upload handling
- Adding authentication and authorization
- Creating data models
- Adding API documentation (Flask-RESTX)
- Implementing image processing features

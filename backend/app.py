from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime, timezone
from PIL import Image
import io
import base64

# Try to import ML dependencies, handle gracefully if not available
try:
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    print("Install with: pip install torch transformers accelerate bitsandbytes")
    ML_AVAILABLE = False

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Enable CORS for frontend communication - more permissive for development
    CORS(app, 
         origins="*",  # Allow all origins in development
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With', 'Accept'],
         supports_credentials=False)  # Set to False when using wildcard origins
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Skip ML model loading for now to avoid startup delays
    model = None
    processor = None
    print("Skipping ML model loading for faster startup...")
    
    # Register blueprints
    from routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Image Processing endpoint (handles both upscale and original)
    @app.route('/api/upscale', methods=['POST', 'OPTIONS'])
    def process_image():
        # Handle preflight OPTIONS request
        if request.method == 'OPTIONS':
            return '', 200
            
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        try:
            file = request.files['image']
            if not file.filename:
                return jsonify({"error": "No file selected"}), 400
                
            image_bytes = file.read()
            if not image_bytes:
                return jsonify({"error": "Empty file uploaded"}), 400
                
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as img_error:
                return jsonify({"error": f"Invalid image file: {str(img_error)}"}), 400

            # Get processing type from request (default to upscale for backward compatibility)
            processing_type = request.form.get('type', 'upscale')
            
            # Convert image to base64 for response
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_uri = "data:image/jpeg;base64," + img_str

            if processing_type == 'original':
                # For original processing, just return the image as-is
                return jsonify({
                    "upscaledImageUrl": img_uri, 
                    "message": "Image processed without upscaling",
                    "status": "original",
                    "processingType": "original"
                })
            
            # AI Upscaling processing
            if model is None or processor is None:
                # Fallback: return the original image with a message
                return jsonify({
                    "upscaledImageUrl": img_uri, 
                    "message": "ML dependencies not available. Install torch, transformers, accelerate, and bitsandbytes to enable AI upscaling.",
                    "status": "fallback",
                    "processingType": "upscale"
                })

            # ML processing (when dependencies are available)
            prompt = "Enhance and upscale this image without losing details."
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=100)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # For real image upscaling, you may need to call a dedicated upscaling module or model here
            # For demonstration, encode the original image back as base64 (replace with upscale result)
            # In a real implementation, you would apply actual upscaling here
            buffered_upscaled = io.BytesIO()
            image.save(buffered_upscaled, format="JPEG", quality=95)
            img_str_upscaled = base64.b64encode(buffered_upscaled.getvalue()).decode()
            img_uri_upscaled = "data:image/jpeg;base64," + img_str_upscaled

            return jsonify({
                "upscaledImageUrl": img_uri_upscaled, 
                "message": output_text, 
                "status": "processed",
                "processingType": "upscale"
            })
            
        except Exception as e:
            return jsonify({"error": f"Processing error: {str(e)}"}), 500
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0'
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))  # Running on port 5000
    app.run(host='0.0.0.0', port=port, debug=True)

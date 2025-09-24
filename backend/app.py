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
    
    # Enable CORS for frontend communication
    CORS(app, origins=[
        'http://localhost:3000',  # React default port
        'http://localhost:5173',  # Vite default port
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5173'
    ])
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Load LLaVA model and processor (adjust to your local model path)
    model = None
    processor = None
    
    if ML_AVAILABLE:
        print("Loading LLaVA model...")
        try:
            model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            print("LLaVA model loaded successfully!")
        except Exception as e:
            print(f"Error loading LLaVA model: {e}")
            model = None
            processor = None
    else:
        print("ML dependencies not available. LLaVA functionality disabled.")
    
    # Register blueprints
    from routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # LLaVA Image Upscaling endpoint
    @app.route('/api/upscale', methods=['POST'])
    def upscale_image():
        if model is None or processor is None:
            return jsonify({"error": "LLaVA model not loaded"}), 500
            
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        try:
            file = request.files['image']
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Prepare input for LLaVA - here you prompt for upscaling or enhancement
            # This is an indicative prompt, you may need to customize based on the model
            prompt = "Enhance and upscale this medical image without losing details."

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=100)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # For real image upscaling, you may need to call a dedicated upscaling module or model here
            # For demonstration, encode the original image back as base64 (replace with upscale result)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_uri = "data:image/jpeg;base64," + img_str

            return jsonify({"upscaledImageUrl": img_uri, "message": output_text})
            
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

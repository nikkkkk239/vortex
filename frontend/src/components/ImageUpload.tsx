import { useState, useRef, useCallback } from 'react';
import axios from 'axios';

function ImageUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [processedUrl, setProcessedUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showOptions, setShowOptions] = useState(false);
  const [processingType, setProcessingType] = useState(''); // 'upscale' or 'original'
  const fileInputRef = useRef<HTMLInputElement>(null);

  const onFileChange = useCallback((file: File) => {
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }
    
    setSelectedFile(file);
    setError('');
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setProcessedUrl(''); // Clear previous result
    setShowOptions(true); // Show options after successful upload
    setProcessingType(''); // Reset processing type
  }, []);

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) onFileChange(file);
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileChange(e.dataTransfer.files[0]);
    }
  }, [onFileChange]);

  const processImage = async (type: 'upscale' | 'original') => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }
    
    setProcessingType(type);
    
    // For "Original Quality", skip processing and show original image directly
    if (type === 'original') {
      const originalImageUrl = URL.createObjectURL(selectedFile);
      setProcessedUrl(originalImageUrl);
      return;
    }
    
    // For "AI Upscale", process the image
    setLoading(true);
    setError('');
    setUploadProgress(0);
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('type', type);

    const apiUrl = 'http://localhost:5000/api/upscale';

    try {
      const response = await axios.post(apiUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        }
      });
      
      // Use the response from backend for upscaled image
      setProcessedUrl(response.data.upscaledImageUrl);
    } catch (error) {
      console.error('Error processing image:', error);
      setError(`Failed to process image. Please try again.`);
    }
    
    setLoading(false);
    setUploadProgress(0);
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setPreviewUrl('');
    setProcessedUrl('');
    setError('');
    setUploadProgress(0);
    setShowOptions(false);
    setProcessingType('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="image-upload-container">
      <div className="upload-header">
        <h2 className="upload-title">AI Image Processor</h2>
        <p className="upload-subtitle">Upload an image and choose your processing option</p>
      </div>

      <div className="upload-section">
        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''} ${previewUrl ? 'has-preview' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !showOptions && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            className="file-input"
          />
          
          {!previewUrl ? (
            <div className="upload-content">
              <div className="upload-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7,10 12,15 17,10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
              </div>
              <h3 className="upload-text">Drop your image here</h3>
              <p className="upload-hint">or click to browse files</p>
              <p className="upload-requirements">Supports JPG, PNG, GIF up to 10MB</p>
            </div>
          ) : (
            <div className="preview-container">
              <img src={previewUrl} alt="Preview" className="preview-image" />
              <div className="preview-overlay">
                <button className="change-image-btn" onClick={(e) => e.stopPropagation()}>
                  Change Image
                </button>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <line x1="15" y1="9" x2="9" y2="15"/>
              <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            {error}
          </div>
        )}

        {/* Processing Options */}
        {showOptions && !processedUrl && !loading && (
          <div className="processing-options">
            <h3 className="options-title">Choose Processing Option</h3>
            <div className="options-container">
              <button 
                className="option-btn upscale-option"
                onClick={() => processImage('upscale')}
              >
                <div className="option-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 3h18v18H3z"/>
                    <path d="M9 9h6v6H9z"/>
                    <path d="M15 3v6"/>
                    <path d="M21 9h-6"/>
                    <path d="M9 15v6"/>
                    <path d="M3 15h6"/>
                  </svg>
                </div>
                <div className="option-content">
                  <h4 className="option-title">AI Upscale</h4>
                  <p className="option-description">Enhance image quality with AI-powered upscaling</p>
                </div>
              </button>

              <button 
                className="option-btn original-option"
                onClick={() => processImage('original')}
              >
                <div className="option-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21,15 16,10 5,21"/>
                  </svg>
                </div>
                <div className="option-content">
                  <h4 className="option-title">Original Quality</h4>
                  <p className="option-description">Show image without any processing</p>
                </div>
              </button>
            </div>
          </div>
        )}

        {/* Processing State */}
        {loading && (
          <div className="processing-state">
            <div className="processing-content">
              <div className="spinner"></div>
              <h3 className="processing-title">
                {processingType === 'upscale' ? 'AI Upscaling...' : 'Processing...'}
              </h3>
              <p className="processing-subtitle">
                {processingType === 'upscale' 
                  ? 'Enhancing your image with AI technology' 
                  : 'Processing your image'}
              </p>
              {uploadProgress > 0 && (
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <span className="progress-text">{uploadProgress}% complete</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Reset Button */}
        {selectedFile && !loading && (
          <div className="upload-actions">
            <button className="reset-btn" onClick={resetUpload}>
              Upload New Image
            </button>
          </div>
        )}
      </div>

      {/* Results Section */}
      {processedUrl && (
        <div className="result-section">
          <div className="result-header">
            <h3 className="result-title">
              {processingType === 'upscale' ? 'AI Enhanced Image' : 'Original Image'}
            </h3>
            <div className="result-actions">
              <a 
                href={processedUrl} 
                download={`${processingType}-image.jpg`} 
                className="download-btn"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="7,10 12,15 17,10"/>
                  <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Download
              </a>
            </div>
          </div>
          
          {/* Show comparison only for upscaled images */}
          {processingType === 'upscale' ? (
            <div className="comparison-container">
              <div className="image-comparison">
                <div className="image-panel">
                  <h4 className="panel-title">Original</h4>
                  <img src={previewUrl} alt="Original" className="comparison-image" />
                </div>
                <div className="image-panel">
                  <h4 className="panel-title">AI Enhanced</h4>
                  <img src={processedUrl} alt="Enhanced" className="comparison-image" />
                </div>
              </div>
            </div>
          ) : (
            /* Show single image for original quality */
            <div className="single-image-container">
              <div className="single-image-panel">
                <h4 className="panel-title">Original Image</h4>
                <img src={processedUrl} alt="Original" className="single-image" />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUpload;

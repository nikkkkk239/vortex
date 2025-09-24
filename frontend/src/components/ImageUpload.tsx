import React, { useState } from 'react';
import axios from 'axios';

function ImageUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [upscaledUrl, setUpscaledUrl] = useState('');
  const [loading, setLoading] = useState(false);

  const onFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const onUpload = async () => {
    if (!selectedFile) return alert('Please upload an image first');
    setLoading(true);
    
    const formData = new FormData();
    formData.append('image', selectedFile);

    // Replace with your backend API url which calls LLaVA for upscaling
    const apiUrl = 'http://localhost:5000/api/upscale'; 

    try {
      const response = await axios.post(apiUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUpscaledUrl(response.data.upscaledImageUrl);
    } catch (error) {
      console.error('Error uploading or processing image:', error);
    }
    
    setLoading(false);
  };

  return (
    <div>
      <h2>Upload Image for Upscaling</h2>
      <input type="file" accept="image/jpeg" onChange={onFileChange} />
      {previewUrl && <img src={previewUrl} alt="uploaded" style={{ maxWidth: 300, marginTop: 10 }} />}
      <br />
      <button onClick={onUpload} disabled={loading}>
        {loading ? 'Upscaling...' : 'Upscale Image'}
      </button>
      <br />
      {upscaledUrl && (
        <>
          <h3>Upscaled Image</h3>
          <img src={upscaledUrl} alt="upscaled" style={{ maxWidth: 600, marginTop: 10 }} />
        </>
      )}
    </div>
  );
}

export default ImageUpload;

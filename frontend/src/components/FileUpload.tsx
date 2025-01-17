import React, { useState } from 'react';
import axios from 'axios';

const FileUpload: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [text, setText] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setFile(event.target.files[0]);
        }
    };

    const handleSubmit = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('/api/validation/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setText(response.data.text);
        } catch (error) {
            console.error('Error uploading file:', error);
        }
    };

    return (
        <div>
            <h1>PlanAid File Upload</h1>
            <input 
                type="file" 
                onChange={handleFileChange} 
                data-testid="file-input"
            />
            <button onClick={handleSubmit} disabled={!file}>
                Upload
            </button>
            {text && (
                <div>
                    <h3>Extracted Text:</h3>
                    <pre>{text}</pre>
                </div>
            )}
        </div>
    );
};

export default FileUpload;

import React, { useState } from 'react';
import axios from 'axios';

type PdfType = 'Regulations' | 'PlanMap';

interface ProcessedPage {
    pageNumber: number;
    content: string;
}

interface ProcessedDocument {
    documentId: string;
    pageCount: number;
    processedAt: string;
    pages: ProcessedPage[];
    extractedFields?: Record<string, string>;
}

const FileUpload: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [pdfType, setPdfType] = useState<PdfType>('Regulations');
    const [processedDoc, setProcessedDoc] = useState<ProcessedDocument | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setFile(event.target.files[0]);
        }
    };

    const handleSubmit = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);
        setError(null);

        try {
            const response = await axios.post(`/api/documents/upload?type=${pdfType}`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setProcessedDoc(response.data);
        } catch (error) {
            console.error('Error uploading file:', error);
            setError('Failed to upload file. Please try again.');
        }
    };

    return (
        <div>
            <h1>PlanAid File Upload</h1>
            <div>
                <select value={pdfType} onChange={(e) => setPdfType(e.target.value as PdfType)}>
                    <option value="Regulations">Regulations</option>
                    <option value="PlanMap">Plan Map</option>
                </select>
            </div>
            <input 
                type="file" 
                onChange={handleFileChange} 
                data-testid="file-input"
                accept=".pdf"
            />
            <button onClick={handleSubmit} disabled={!file}>
                Upload
            </button>
            
            {error && (
                <div className="error-message" style={{ color: 'red', margin: '10px 0' }}>
                    {error}
                </div>
            )}
            
            {processedDoc && (
                <div>
                    <h2>Document ID: {processedDoc.documentId}</h2>
                    <p>Pages: {processedDoc.pageCount}</p>
                    <p>Processed: {new Date(processedDoc.processedAt).toLocaleString()}</p>
                    {processedDoc.extractedFields && (
                        <div>
                            <h3>Extracted Fields:</h3>
                            <pre>{JSON.stringify(processedDoc.extractedFields, null, 2)}</pre>
                        </div>
                    )}
                    {processedDoc.pages?.map((page) => (
                        <div key={page.pageNumber}>
                            <h3>Page {page.pageNumber}</h3>
                            <pre>{page.content}</pre>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default FileUpload;

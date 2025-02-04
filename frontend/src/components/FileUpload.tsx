import React, { useState } from 'react';
import axios from 'axios';

type PdfType = 'Regulations' | 'PlanMap';

const FileUpload: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [pdfType, setPdfType] = useState<PdfType>('Regulations');
    const [processedDoc, setProcessedDoc] = useState<any>(null);

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
            const response = await axios.post(`/api/validation/upload?type=${pdfType}`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setProcessedDoc(response.data);
        } catch (error) {
            console.error('Error uploading file:', error);
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
            {processedDoc && (
                <div>
                    <h2>{processedDoc.title}</h2>
                    <p>Pages: {processedDoc.totalPages}</p>
                    <p>Processed: {new Date(processedDoc.processedDate).toLocaleString()}</p>
                    {processedDoc.metadata?.fieldIdentifiers && (
                        <div>
                            <h3>Found Field Identifiers:</h3>
                            <pre>{processedDoc.metadata.fieldIdentifiers}</pre>
                        </div>
                    )}
                    {processedDoc.pages?.map((page: any) => (
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

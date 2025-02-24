import React, { useState } from 'react';
import axios from 'axios';
import { FileType } from '../pages/Plan2'; // Keep this import

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

interface FileUploadProps {
  onFileUpload: (type: FileType, file: File) => void;
  handleSubmit: (event: React.FormEvent) => Promise<void>;
  files: {
    plankart: File | null;
    bestemmelser: File | null;
    sosi: File | null;
  };
  loading: boolean;
  error: string | null;
  progress: number;
  processingStep: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFileUpload,
  handleSubmit,
  files,
  loading,
  error,
  progress,
  processingStep,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [processedDoc, setProcessedDoc] = useState<ProcessedDocument | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleSubmitLocal = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setLocalError(null);

    try {
      const response = await axios.post('/api/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setProcessedDoc(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      setLocalError('Failed to upload file. Please try again.');
    }
  };

  return (
    <div style={styles.uploadContainer}>
      <h2 style={styles.title}>Feltsjekk for Reguleringsplan</h2>
      <form onSubmit={handleSubmitLocal} className="space-y-6">
        <div style={styles.fileInputContainer}>
          <input
            type="file"
            onChange={handleFileChange}
            data-testid="file-input"
            accept=".pdf"
          />
          <button type="submit" disabled={!file} style={styles.button}>
            Upload
          </button>
        </div>

        {localError && (
          <div style={styles.errorContainer}>
            <p style={styles.errorText}>{localError}</p>
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
      </form>
    </div>
  );
};

const styles = {
  uploadContainer: {
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    padding: '20px',
    boxShadow: '0px 0px 10px rgba(0, 0, 0, 0.1)',
    textAlign: 'left' as const,
    maxWidth: '900px',
    margin: 'auto',
  },
  title: {
    fontSize: '20px',
    fontWeight: 'bold',
    marginBottom: '20px',
    color: '#333',
  },
  fileInputContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '20px',
  },
  errorContainer: {
    marginTop: '10px',
    padding: '10px',
    backgroundColor: '#fdecea',
    borderRadius: '5px',
  },
  errorText: {
    color: '#d32f2f',
    fontSize: '14px',
  },
  button: {
    marginTop: '20px',
    backgroundColor: '#24BD76',
    color: '#fff',
    padding: '10px 15px',
    borderRadius: '5px',
    fontSize: '16px',
    cursor: 'pointer',
    border: 'none',
  },
};

export default FileUpload;
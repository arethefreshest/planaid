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
// FileUpload.tsx
import React, { useState } from "react";
import type { FileType } from "../pages/Plan2";
import CustomInput from '../components/CustomInput';
import { CircularProgress } from "./CircularProgress";

interface FileUploadProps {
  onFileUpload: (type: FileType, file: File) => void;
  handleSubmit: (event: React.FormEvent) => void;
  files: { plankart: File | null; bestemmelser: File | null; sosi: File | null };
  loading: boolean;
  error: string | null;
  progress: number;
  processingStep: string;
}

const FileUpload = ({ onFileUpload, handleSubmit, files, loading, error, progress, processingStep }: FileUploadProps) => {
  const handleChange = (type: FileType) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileUpload(type, file);
    }
  };

  return (
    <div style={styles.uploadContainer}>
      <h2 style={styles.title}>Feltsjekk for Reguleringsplan</h2>
      <CustomInput />
      <form onSubmit={handleSubmit} className="space-y-6">
        <div style={styles.fileInputContainer}>
          {Object.entries(files).map(([type, file]) => (
            <div key={type} style={styles.fileInputWrapper}>
              <label style={styles.label}>{type === 'plankart' ? 'Plankart' : type === 'bestemmelser' ? 'Bestemmelser' : 'SOSI-fil'}{type !== 'sosi' && <span style={styles.required}>*</span>}</label>
              <div style={styles.relative}>
                <input type="file" accept=".pdf,.xml,.sos" onChange={handleChange(type as FileType)} style={styles.hiddenInput} id={`file-${type}`} />
                <label htmlFor={`file-${type}`} style={styles.uploadLabel}>
                  {file ? <span>{file.name}</span> : <span>Velg fil</span>}
                </label>
              </div>
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
          ))}
        </div>

        {loading && (
          <div className="mt-4">
            <CircularProgress progress={progress} />
            <p style={styles.progressText}>{processingStep}</p>
          </div>
        )}

        {error && (
          <div style={styles.errorContainer}>
            <p style={styles.errorText}>{error}</p>
          </div>
        )}
      <button type="submit" disabled={!files.plankart || !files.bestemmelser} style={styles.button}>
        Start analyse
      </button>
      </form>
    </div>
  );
};

const styles = {
  uploadContainer: {
    backgroundColor: "#ffffff",
    borderRadius: "8px",
    padding: "20px",
    boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.1)",
    textAlign: "left" as const,
    maxWidth: "900px",
    margin: "auto",
  },
  title: {
    fontSize: "20px",
    fontWeight: "bold",
    marginBottom: "20px",
    color: "#333",
  },
  fileInputContainer: {
    display: "flex",
    justifyContent: "space-between",
    gap: "20px",
  },
  fileInputWrapper: {
    flex: "1",
    display: "flex",
    flexDirection: "column" as const,
  },
  label: {
    fontSize: "14px",
    fontWeight: "bold",
    marginBottom: "5px",
    color: "#333",
  },
  required: {
    color: "#e74c3c",
    marginLeft: "4px",
  },
  hiddenInput: {
    display: "none",
  },
  uploadLabel: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "10px",
    border: "1px solid #ccc",
    borderRadius: "4px",
    fontSize: "14px",
    width: "100%",
    cursor: "pointer",
  },
  progressText: {
    fontSize: "14px",
    color: "#666",
    marginTop: "10px",
  },
  errorContainer: {
    marginTop: "10px",
    padding: "10px",
    backgroundColor: "#fdecea",
    borderRadius: "5px",
  },
  errorText: {
    color: "#d32f2f",
    fontSize: "14px",
  },
  button: {
    marginTop: "20px",
    backgroundColor: "#24BD76",
    color: "#fff",
    padding: "10px 15px",
    borderRadius: "5px",
    fontSize: "16px",
    cursor: "pointer",
    border: "none",
  },
  relative: {
    position: 'relative' as const,
    width: '100%'
  },
};

export default FileUpload;



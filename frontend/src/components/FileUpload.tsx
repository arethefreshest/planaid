import React, { useState, CSSProperties } from "react";
import axios from "axios";
import { logger } from "../utils/logger";

// Define FileType inside this component
type FileType = "plankart" | "bestemmelser" | "sosi";

const API_URL = process.env.REACT_APP_API_URL ?? "http://localhost:5251";

// Configuration constants
const ALLOWED_TYPES = ["application/pdf", "text/xml", "text/plain", "application/octet-stream"];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Validation function
const validateFile = (file: File, type: FileType): string | null => {
  // For SOSI files, check the extension instead of MIME type
  if (type === "sosi" as FileType) {
    if (!file.name.toLowerCase().endsWith('.sos')) {
      return "SOSI filer må ha .sos filendelse.";
    }
    return null;
  }
  
  // For PDF files (plankart and bestemmelser)
  if (type !== "sosi" as FileType && !file.type.includes('pdf')) {
    return `${type.charAt(0).toUpperCase() + type.slice(1)} må være en PDF fil.`;
  }
  
  if (file.size > MAX_FILE_SIZE) {
    return "Filen er for stor. Maksimal størrelse er 10MB.";
  }
  return null;
};

const FileUpload = ({ onUploadSuccess }: { onUploadSuccess: (result: any) => void }) => {
  const [files, setFiles] = useState<{ [key in FileType]?: File }>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleFileChange = (type: FileType) => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      const validationError = validateFile(file, type);

      if (validationError) {
        setError(validationError);
        return;
      }

      setFiles((prev) => ({ ...prev, [type]: file }));
      setError(null);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (!files.plankart || !files.bestemmelser) {
        setError("Plankart og Bestemmelser må lastes opp.");
        setLoading(false);
        return;
      }

      const formData = new FormData();
      formData.append("plankart", files.plankart);
      formData.append("bestemmelser", files.bestemmelser);
      if (files.sosi) {
        formData.append("sosi", files.sosi);
        logger.info("Including SOSI file in upload");
      }

      logger.info("Sending files to backend:", {
        plankart: files.plankart.name,
        bestemmelser: files.bestemmelser.name,
        sosi: files.sosi?.name
      });

      const response = await axios.post(`${API_URL}/api/check-field-consistency`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (event) => {
          const percent = Math.round((event.loaded * 100) / (event.total || 1));
          setProgress(percent);
        },
      });

      onUploadSuccess(response.data);
    } catch (err) {
      logger.error("Feil under analyse:", err);
      setError(err instanceof Error ? err.message : "En feil oppstod");
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  return (
    <div style={styles.uploadContainer}>
      <h2 style={styles.title}>Feltsjekk for Reguleringsplan</h2>
      <form onSubmit={handleSubmit}>
        <div style={styles.fileInputContainer}>
          {["plankart", "bestemmelser", "sosi"].map((type) => (
            <div key={type} style={styles.fileInputWrapper}>
              <label style={styles.fileInput}>
                {type === "sosi" ? "SOSI (valgfri)" : type.toUpperCase()}
                <input 
                  type="file" 
                  onChange={handleFileChange(type as FileType)} 
                  accept={type === "sosi" ? ".sos" : ".pdf"} 
                  style={{ display: 'none' }} 
                />
              </label>
              {files[type as FileType] && <p style={styles.fileName}>{files[type as FileType]?.name}</p>}
            </div>
          ))}
        </div>
        {error && <p style={styles.errorText}>{error}</p>}
        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? `Laster opp... ${progress}%` : "Start Analyse"}
        </button>
      </form>
    </div>
  );
};

const styles: { [key: string]: CSSProperties } = {
  uploadContainer: { 
    padding: "24px", 
    maxWidth: "900px", 
    margin: "auto", 
    borderRadius: "12px", 
    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)", 
    backgroundColor: "#fff" },
  title: { 
    fontSize: "22px", 
    fontWeight: "bold", 
    marginBottom: "20px", 
    textAlign: "center" },
  fileInputContainer: { 
    display: "flex", 
    justifyContent: "center", 
    gap: "16px" },
  fileInputWrapper: {
    display: "flex", 
    flexDirection: "column", 
    alignItems: "center", 
    textAlign: "center" },
  fileInput: {
    display: "flex", 
    alignItems: "center", 
    justifyContent: "center", 
    backgroundColor: "#f9f9f9", 
    borderRadius: "12px", 
    padding: "12px 18px", 
    boxShadow: "0 3px 6px rgba(0, 0, 0, 0.15)", 
    cursor: "pointer", 
    border: "1px solid #ccc", 
    fontSize: "16px", 
    fontWeight: "500", 
    transition: "all 0.3s ease", 
    textAlign: "center", 
    width: "200px" },
  fileName: { 
    fontSize: "16px", 
    marginTop: "6px", 
    color: "#555" },
  errorText: { 
    color: "#d32f2f", 
    fontSize: "14px", 
    marginTop: "10px", 
    textAlign: "center" },
  button: { 
    marginTop: "24px",
    marginLeft: "30%", 
    backgroundColor: "#24BD76", 
    color: "#fff", 
    padding: "14px", 
    borderRadius: "10px", 
    width: "40%", 
    boxShadow: "0 3px 10px rgba(0, 0, 0, 0.2)", 
    border: "none", 
    cursor: "pointer", 
    transition: "background 0.3s ease" },
};

export default FileUpload;

import React, { useState } from "react";
import axios from "axios";
import { CircularProgress } from "../components/CircularProgress";
import { logger } from "../utils/logger";

// Define FileType inside this component
type FileType = "plankart" | "bestemmelser" | "sosi";

const API_URL = process.env.REACT_APP_API_URL ?? "http://localhost:5251";

// Configuration constants
const ALLOWED_TYPES = ["application/pdf", "text/xml"];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Validation function
const validateFile = (file: File): string | null => {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return "Ugyldig filformat. Vennligst last opp PDF eller XML fil.";
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
  const [processingStep, setProcessingStep] = useState<string>("");

  const handleFileChange = (type: FileType) => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      const validationError = validateFile(file);

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
      }

      const response = await axios.post(`${API_URL}/api/check-field-consistency`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (event) => {
          const percent = Math.round((event.loaded * 100) / (event.total || 1));
          setProgress(percent);
        },
      });

      onUploadSuccess(response.data.result);
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
          {(["plankart", "bestemmelser", "sosi"] as FileType[]).map((type) => (
            <div key={type}>
              <label style={styles.label}>{type.toUpperCase()}</label>
              <input type="file" onChange={handleFileChange(type)} accept=".pdf,.xml,.sos" />
              {files[type] && <p style={styles.fileName}>{files[type]?.name}</p>}
            </div>
          ))}
        </div>

        {error && <p style={styles.errorText}>{error}</p>}

        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? `Laster opp... ${progress}%` : "Start Analyse"}
        </button>

        {loading && <CircularProgress progress={progress} />}
        {processingStep && <p>{processingStep}</p>}
      </form>
    </div>
  );
};

// Styles object
const styles = {
  uploadContainer: { 
    padding: "20px", 
    maxWidth: "900px", 
    margin: "auto" },
  title: { 
    fontSize: "20px", 
    fontWeight: "bold", 
    marginBottom: "20px" },
  fileInputContainer: { 
    display: "flex", 
    justifyContent: "space-between", 
    gap: "20px" },
  label: { 
    fontWeight: "bold", 
    marginBottom: "5px" },
  fileName: { 
    fontSize: "14px", 
    marginTop: "5px" },
  errorText: { 
    color: "#d32f2f", 
    fontSize: "14px", 
    marginTop: "10px" },
  button: { 
    marginTop: "20px", 
    backgroundColor: "#24BD76", 
    color: "#fff", 
    padding: "10px", 
    borderRadius: "5px" },
};

export default FileUpload;

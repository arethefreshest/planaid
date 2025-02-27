import React, { useState } from "react";
import Layout from "../components/Layout";
import axios from "axios";
import { CircularProgress } from "../components/CircularProgress";
import { logger } from "../utils/logger";
import ConsistencyResults from '../components/ConsistencyResults';
import FileUpload from '../components/FileUpload';
import { FileType } from '../types'

const API_URL = process.env.REACT_APP_API_URL ?? 'http://backend:5251';


interface ConsistencyResult {
  is_consistent: boolean;
  matching_fields: string[];
  only_in_plankart: string[];
  only_in_bestemmelser: string[];
  only_in_sosi?: string[];
  metadata?: Record<string, string>;
}

const Plan2 = () => {
  const [files, setFiles] = useState<{
    plankart: File | null,
    bestemmelser: File | null,
    sosi: File | null
  }>({
    plankart: null,
    bestemmelser: null,
    sosi: null,
  });
  const [result, setResult] = useState<ConsistencyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState("");

  const handleFileUpload = (type: FileType, file: File) => {
    if (file) {
      setFiles((prev) => ({
        ...prev,
        [type]: file,
      }));
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setProgress(0);
    setProcessingStep("Starter opplasting...");

    const formData = new FormData();
    if (files.plankart) formData.append("plankart", files.plankart);
    if (files.bestemmelser) formData.append("bestemmelser", files.bestemmelser);
    if (files.sosi) formData.append("sosi", files.sosi);

    try {
      setProcessingStep("Laster opp filer...");

      const response = await axios.post(`${API_URL}/api/check-field-consistency`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const total = progressEvent.total ?? 1;
          const percentCompleted = Math.round((progressEvent.loaded * 100) / total);
          setProgress(percentCompleted);
          setProcessingStep(`Laster opp... ${percentCompleted}%`);
        },
      });

      setProcessingStep("Analyserer dokumenter...");
      setResult(response.data);
      setProcessingStep("Analyse fullført ✅");
    } catch (error: any) {
      console.error("Feil under opplasting:", error);
      setError(error?.response?.data?.detail || "En feil oppstod under analysen.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div style={styles.container}>
        <div style={styles.LeftContainer}>
          <FileUpload
            onFileUpload={handleFileUpload}
            handleSubmit={handleSubmit}
            files={files}
            loading={loading}
            error={error}
            progress={progress}
            processingStep={processingStep}
          />
        </div>

        <div style={styles.RightContainer}>
          <h2 style={styles.title}>Kart Analyse: </h2>
          {loading && <CircularProgress progress={progress} />}
          {result && <ConsistencyResults result={result} />}
          {error && (
            <div style={styles.errorContainer}>
              <p style={styles.errorText}>{error}</p>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

const styles = {
  container: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: "2rem",
    flexWrap: "wrap" as const,
  },
  LeftContainer: {
    flex: 1,
    textAlign: "left" as const,
  },
  RightContainer: {
    backgroundColor: "#ffffff",
    borderRadius: "8px",
    padding: "20px",
    boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.1)",
    height: "650px",
    flex: 1,
    paddingLeft: "30px",
    textAlign: "left" as const,
  },
  title: {
    fontSize: "20px",
    fontWeight: "bold",
    marginBottom: "20px",
    color: "#333",
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
};

export default Plan2;
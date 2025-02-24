import React, { useState } from "react";
import Layout from "../components/Layout";
import CustomInput from "../components/CustomInput";
import axios from "axios";
import { CircularProgress } from "../components/CircularProgress";
import { logger } from "../utils/logger";
import ConsistencyResults from '../components/ConsistencyResults';
import FileUpload from '../components/FileUpload';

const API_URL = process.env.REACT_APP_API_URL;


export type FileType = 'plankart' | 'bestemmelser' | 'sosi';

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
      setFiles((prev: { plankart: File | null, bestemmelser: File | null, sosi: File | null }) => 
        ({ ...prev, [type]: file })
      );
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
  
      const response = await axios.post("/api/consistency/check-field-consistency", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const total = progressEvent.total ?? 1;
          const percentCompleted = Math.round((progressEvent.loaded * 100) / total);
          setProgress(percentCompleted);
          setProcessingStep(`Laster opp... ${percentCompleted}%`);
        },
      });
  
      setProcessingStep("Analyserer dokumenter...");
      console.log("Respons fra backend:", response.data);
  
      // 💡 Her kan du oppdatere UI med resultatet fra backend
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
          {result && <ConsistencyResults result={result} />}
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
  tryButton: {
    backgroundColor: "#004d00",
    color: "#fff",
    border: "none",
    padding: "1rem 2rem",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "1.5rem",
  },
};

export default Plan2;

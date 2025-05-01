import React, { useState, CSSProperties } from "react";
import axios from "axios";

const API_URL = process.env.REACT_APP_API_URL ?? "http://localhost:5251";

const ComparePdf = () => {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ text1: string; text2: string; diff: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (type: "file1" | "file2") => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (type === "file1") setFile1(file);
    else setFile2(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file1 || !file2) {
      setError("Begge PDF-filer må lastes opp.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file1", file1);
    formData.append("file2", file2);

    try {
      const res = await axios.post(`${API_URL}/api/compare-pdfs`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setResult(res.data);
    } catch (err) {
      setError("Feil under sammenligning.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
      <div style={styles.container}>
        <h2 style={styles.title}>Versjonskontroll:</h2>

        <form onSubmit={handleSubmit} style={styles.form}>
          <h3 style={styles.uploadLabel}>Opplastning for Versjonskontroll</h3>

          <div style={styles.buttonRow}>
            <div>
              <label style={styles.uploadButton}>
                PDF 1
                <input type="file" accept="application/pdf" onChange={handleFileChange("file1")} style={{ display: "none" }} />
              </label>
              {file1 && <p style={styles.fileName}>{file1.name}</p>}
            </div>

            <div>
              <label style={styles.uploadButton}>
                PDF 2
                <input type="file" accept="application/pdf" onChange={handleFileChange("file2")} style={{ display: "none" }} />
              </label>
              {file2 && <p style={styles.fileName}>{file2.name}</p>}
            </div>
          </div>

          {error && <p style={styles.errorText}>{error}</p>}

          <button type="submit" disabled={loading} style={styles.compareButton}>
            {loading ? "Sammenligner..." : "Sammenlign"}
          </button>
        </form>

        {result && (
          <>
            <h3 style={styles.diffHeader}>Her er forskjellene mellom <b>Document 1</b> og <b>Document 2</b>:</h3>
            <pre style={styles.diffBox}>{result.diff}</pre>

            <div style={styles.sideBySide}>
              <div style={styles.textBlock}>
                <h4>Document 1</h4>
                <pre style={styles.textPre}>{result.text1}</pre>
              </div>

              <div style={styles.textBlock}>
                <h4>Document 2</h4>
                <pre style={styles.textPre}>{result.text2}</pre>
              </div>
            </div>
          </>
        )}
      </div>
  );
};

const styles: { [key: string]: CSSProperties } = {
  container: {
    padding: "24px",
    maxWidth: "1200px",
    margin: "auto",
  },
  title: {
    fontSize: "24px",
    fontWeight: 600,
    marginBottom: "24px",
  },
  form: {
    background: "#f9f9f9",
    padding: "20px",
    borderRadius: "12px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
    textAlign: "center",
  },
  uploadLabel: {
    fontSize: "18px",
    fontWeight: 500,
    marginBottom: "16px",
  },
  buttonRow: {
    display: "flex",
    justifyContent: "center",
    gap: "40px",
    marginBottom: "16px",
  },
  uploadButton: {
    backgroundColor: "#f0f0f0",
    padding: "12px 20px",
    borderRadius: "12px",
    boxShadow: "0 3px 6px rgba(0, 0, 0, 0.15)",
    border: "1px solid #ccc",
    cursor: "pointer",
    fontWeight: 500,
    display: "inline-block",
    width: "150px",
    textAlign: "center",
    transition: "all 0.3s ease",
  },
  fileName: {
    fontSize: "14px",
    color: "#555",
    marginTop: "6px",
  },
  errorText: {
    color: "#d32f2f",
    fontSize: "14px",
    marginTop: "10px",
    textAlign: "center",
  },
  compareButton: {
    marginTop: "20px",
    backgroundColor: "#24BD76",
    color: "#fff",
    padding: "14px",
    borderRadius: "10px",
    width: "200px",
    fontWeight: "bold",
    border: "none",
    cursor: "pointer",
    transition: "background 0.3s ease",
  },
  diffHeader: {
    fontSize: "18px",
    marginTop: "40px",
    marginBottom: "12px",
  },
  diffBox: {
    whiteSpace: "pre-wrap",
    background: "#e8f5e9",
    border: "1px solid #c8e6c9",
    padding: "16px",
    borderRadius: "8px",
  },
  sideBySide: {
    display: "flex",
    justifyContent: "space-between",
    marginTop: "24px",
    gap: "24px",
  },
  textBlock: {
    flex: 1,
    background: "#fafafa",
    borderRadius: "8px",
    padding: "12px",
    border: "1px solid #ddd",
    overflowX: "auto",
  },
  textPre: {
    whiteSpace: "pre-wrap",
    fontSize: "14px",
    lineHeight: 1.5,
  },
};

export default ComparePdf;

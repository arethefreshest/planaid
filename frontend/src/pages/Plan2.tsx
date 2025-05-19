import React, { useState } from "react";
import Layout from "../components/Layout";
import FileUpload from "../components/FileUpload";
import ConsistencyResults from "../components/ConsistencyResults";
import { Link } from "react-router-dom";

const Plan2 = () => {
  const [result, setResult] = useState<any | null>(null);

  return (
    <Layout>
      <div style={styles.container}>
        <div style={styles.leftContainer}>
          <FileUpload onUploadSuccess={setResult} />
        </div>
        <div style={styles.rightContainer}>
          {result && <ConsistencyResults result={result} />}
        </div>
        </div>
          <Link to="/page3" style={styles.button}>
            Gå til Versjonskontroll
          </Link>
    </Layout>
  );
};

const styles = {
  container: { 
    display: "flex", 
    gap: "2rem", 
    flexWrap: "wrap" as const },
  leftContainer: { 
    flex: 1 },
  rightContainer: { 
    padding: "20px", 
    backgroundColor: "#fff", 
    borderRadius: "8px", 
    flex: 1 },
  button: {
    position: 'fixed' as const,
    bottom: '80px',
    right: '20px',
    backgroundColor: '#24BD76',
    color: '#fff',
    border: 'none',
    padding: '1rem 3rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1rem',
    zIndex: 1000, 
    textDecoration: 'none'
  },
  title: { fontSize: "20px", 
    fontWeight: "bold", 
    marginBottom: "20px" },
};

export default Plan2;
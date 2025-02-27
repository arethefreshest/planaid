import React, { useState } from "react";
import Layout from "../components/Layout";
import FileUpload from "../components/FileUpload";
import ConsistencyResults from "../components/ConsistencyResults";

const Plan2 = () => {
  const [result, setResult] = useState<any | null>(null);

  return (
    <Layout>
      <div style={styles.container}>
        <div style={styles.leftContainer}>
          <FileUpload onUploadSuccess={setResult} />
        </div>
        <div style={styles.rightContainer}>
          <h2 style={styles.title}>Kart Analyse:</h2>
          {result && <ConsistencyResults result={result} />}
        </div>
      </div>
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
  title: { fontSize: "20px", 
    fontWeight: "bold", 
    marginBottom: "20px" },
};

export default Plan2;
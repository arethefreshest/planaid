import React from "react";

const ConsistencyResults = ({ result }: { result: any }) => {
  if (!result) return null;

  return (
    <div style={styles.resultContainer}>
      <h2 style={styles.title}>Analyse Resultater</h2>
      <div style={result.is_consistent ? styles.successBox : styles.errorBox}>
        <h3>{result.is_consistent ? "Ingen avvik funnet" : "Avvik funnet"}</h3>
      </div>

      {result.matching_fields.length > 0 && <p>Matchende Felter: {result.matching_fields.join(", ")}</p>}
      {result.only_in_plankart.length > 0 && <p>Kun i Plankart: {result.only_in_plankart.join(", ")}</p>}
      {result.only_in_bestemmelser.length > 0 && <p>Kun i Bestemmelser: {result.only_in_bestemmelser.join(", ")}</p>}
    </div>
  );
};

// Styles object
const styles = {
  resultContainer: { 
    padding: "20px", 
    maxWidth: "900px", 
    margin: "auto" },
  title: { 
    fontSize: "20px", 
    fontWeight: "bold", 
    marginBottom: "20px" },
  successBox: { 
    backgroundColor: "#e6f7e6", 
    padding: "10px", 
    borderRadius: "5px" },
  errorBox: { 
    backgroundColor: "#fdecea", 
    padding: "10px", 
    borderRadius: "5px" },
};

export default ConsistencyResults;


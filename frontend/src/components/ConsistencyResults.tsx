import React from "react";

interface ConsistencyResult {
  is_consistent: boolean;
  matching_fields: string[];
  only_in_plankart: string[];
  only_in_bestemmelser: string[];
  only_in_sosi?: string[];
  metadata?: Record<string, string>;
}

const ConsistencyResults = ({ result }: { result: ConsistencyResult }) => {
  if (!result) return null;

  return (
    <div>
      <h3>Analysis Results</h3>
      <div style={{ padding: "1rem", borderRadius: "8px", backgroundColor: result.is_consistent ? "#d4edda" : "#f8d7da" }}>
        <strong>{result.is_consistent ? "No inconsistencies found" : "Inconsistencies detected"}</strong>
      </div>
      <FieldList title="Matching Fields" fields={result.matching_fields} color="#d4edda" />
      <FieldList title="Only in Plankart" fields={result.only_in_plankart} color="#fff3cd" />
      <FieldList title="Only in Bestemmelser" fields={result.only_in_bestemmelser} color="#f8d7da" />
      <FieldList title="Only in SOSI" fields={result.only_in_sosi || []} color="#cce5ff" />
      {result.metadata && <MetadataDisplay metadata={result.metadata} />}
    </div>
  );
};

const FieldList = ({ title, fields, color }: { title: string; fields: string[]; color: string }) => {
  return fields.length > 0 ? (
    <div style={{ marginTop: "1rem" }}>
      <h4>{title}</h4>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
        {fields.map(field => (
          <span key={field} style={{ backgroundColor: color, padding: "0.3rem 0.6rem", borderRadius: "5px", fontSize: "0.9rem" }}>{field}</span>
        ))}
      </div>
    </div>
  ) : null;
};

const MetadataDisplay = ({ metadata }: { metadata: Record<string, string> }) => (
  <div style={{ marginTop: "1rem" }}>
    <h4>Metadata</h4>
    <div style={{ backgroundColor: "#f0f0f0", padding: "0.5rem", borderRadius: "5px" }}>
      {Object.entries(metadata).map(([key, value]) => (
        <div key={key}>
          <strong>{key.replace("_", " ").toUpperCase()}:</strong> {value || "Not available"}
        </div>
      ))}
    </div>
  </div>
);

export default ConsistencyResults;

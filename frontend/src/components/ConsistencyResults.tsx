import React from "react";

// Define the expected structure of the result prop
interface Result {
  is_consistent: boolean;
  matching_fields: string[];
  only_in_plankart: string[];
  only_in_bestemmelser: string[];
  only_in_sosi: string[];
}

const ConsistencyResults = ({ result }: { result: Result }) => {
  if (!result) return null;

  const getBadgeStyle = (field: string) => {
    const colorMap = {
      Matching: { background: "#d1fae5", color: "#065f46" }, // Green
      Plankart: { background: "#bfdbfe", color: "#1e3a8a" }, // Blue
      Bestemmelser: { background: "#fde68a", color: "#92400e" }, // Yellow
      SOSI: { background: "#e9d5ff", color: "#6b21a8" }, // Purple for SOSI
    } as const; // Use 'as const' to infer literal types

    type Category = keyof typeof colorMap; // Define a type for the keys of colorMap

    let category: Category = "Matching"; // Initialize category with a valid key
    if (result.only_in_plankart.includes(field)) category = "Plankart";
    if (result.only_in_bestemmelser.includes(field)) category = "Bestemmelser";
    if (result.only_in_sosi.includes(field)) category = "SOSI";

    return {
      display: "inline-flex",
      alignItems: "center",
      padding: "6px 12px",
      borderRadius: "999px",
      fontSize: "14px",
      fontWeight: "500",
      backgroundColor: colorMap[category].background,
      color: colorMap[category].color,
    };
  };

  return (
    <div style={styles.resultContainer}>
      <h2 style={styles.title}>Analyseresultater</h2>
      <div style={result.is_consistent ? styles.successBox : styles.errorBox}>
        <h3>{result.is_consistent ? "Ingen avvik funnet" : "Avvik funnet"}</h3>
      </div>
      {result.matching_fields.length > 0 && (
        <div style={styles.section}>
          <h4 style={styles.title}>Matching Fields</h4>
          <div style={styles.container}>
            {result.matching_fields.map((field: string) => (
              <span key={field} style={getBadgeStyle(field)}>
                {field}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.only_in_plankart.length > 0 && (
        <div style={styles.section}>
          <h4 style={styles.title}>Kun i Plankart</h4>
          <div style={styles.container}>
            {result.only_in_plankart.map((field: string) => (
              <span key={field} style={getBadgeStyle(field)}>
                {field}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.only_in_bestemmelser.length > 0 && (
        <div style={styles.section}>
          <h4 style={styles.title}>Kun i Bestemmelser</h4>
          <div style={styles.container}>
            {result.only_in_bestemmelser.map((field: string) => (
              <span key={field} style={getBadgeStyle(field)}>
                {field}
              </span>
            ))}
          </div>
        </div>
      )}

      {result.only_in_sosi && result.only_in_sosi.length > 0 && (
        <div style={styles.section}>
          <h4 style={styles.title}>Kun i SOSI</h4>
          <div style={styles.container}>
            {result.only_in_sosi.map((field: string) => (
              <span key={field} style={getBadgeStyle(field)}>
                {field}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  resultContainer: {
    padding: "20px",
    maxWidth: "900px",
    margin: "auto",
  },
  title: {
    fontSize: "20px",
    fontWeight: "bold",
    marginBottom: "20px",
  },
  successBox: {
    backgroundColor: "#e6f7e6",
    padding: "10px",
    borderRadius: "5px",
  },
  errorBox: {
    backgroundColor: "#fdecea",
    padding: "10px",
    borderRadius: "5px",
  },
  section: {
    marginBottom: "16px",
  },
  container: {
    display: "flex",
    flexWrap: "wrap" as "wrap",
    gap: "8px",
  },
};

export default ConsistencyResults;


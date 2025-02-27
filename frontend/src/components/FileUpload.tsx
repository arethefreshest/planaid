import React from 'react';

interface FileUploadProps {
}

  return (
    <div style={styles.uploadContainer}>
      <h2 style={styles.title}>Feltsjekk for Reguleringsplan</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div style={styles.fileInputContainer}>
          {(['plankart', 'bestemmelser', 'sosi'] as FileType[]).map((type) => (
            <div key={type}>
              <label style={styles.label}>{type.toUpperCase()}</label>
              <input
                type="file"
                onChange={(e) => handleFileChange(e, type)}
                accept=".pdf,.xml,.sos"
              />
              {files[type] && (
                <p style={styles.fileName}>{files[type]?.name}</p>
              )}
            </div>
          ))}
        </div>

        {error && (
          <div style={styles.errorContainer}>
            <p style={styles.errorText}>{error}</p>
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={styles.button}
        >
          {loading ? `Laster opp... ${progress}%` : 'Start Analyse'}
        </button>

        {processingStep && <p>{processingStep}</p>}
      </form>
    </div>
  );
};

const styles = {
  uploadContainer: {
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    padding: '20px',
    boxShadow: '0px 0px 10px rgba(0, 0, 0, 0.1)',
    textAlign: 'left' as const,
    maxWidth: '900px',
    margin: 'auto',
  },
  title: {
    fontSize: '20px',
    fontWeight: 'bold',
    marginBottom: '20px',
    color: '#333',
  },
  fileInputContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '20px',
    flexWrap: 'wrap' as const,
  },
  fileName: {
    fontSize: '14px',
    marginTop: '5px',
  },
  label: {
    fontWeight: 'bold',
    marginBottom: '5px',
    display: 'block',
  },
  errorContainer: {
    marginTop: '10px',
    padding: '10px',
    backgroundColor: '#fdecea',
    borderRadius: '5px',
  },
  errorText: {
    color: '#d32f2f',
    fontSize: '14px',
  },
  button: {
    marginTop: '20px',
    backgroundColor: '#24BD76',
    color: '#fff',
    padding: '10px 15px',
    borderRadius: '5px',
    fontSize: '16px',
    cursor: 'pointer',
    border: 'none',
  },
};

export default FileUpload;
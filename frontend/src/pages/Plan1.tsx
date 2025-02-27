import Layout from '../components/Layout';
import React, { useState } from "react";
import DragDrop from '../components/DragDrop';
import CustomInput from '../components/CustomInput';
import FileUpload from '../components/FileUpload';

export type FileType = 'plankart' | 'bestemmelser' | 'sosi';



const Plan1 = () => {
  const [files, setFiles] = useState<{ plankart: File | null, bestemmelser: File | null, sosi: File | null }>({
    plankart: null,
    bestemmelser: null,
    sosi: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState("");

  const handleFileUpload = (type: FileType, file: File) => {
    if (file) {
      setFiles((prev) => ({ ...prev, [type]: file }));
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    // Add your submit logic here
  };

  return (
    <Layout>
      <div style={styles.container}>
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
    </Layout>
  );
};
const styles = {
  container: {
    flex: '1', 
    paddingTop: '100px',
    textAlign: 'left' as const,
    justifyContent: 'center'
  },

  tryButton: {
    backgroundColor: '#004d00',
    color: '#fff',
    border: 'none',
    padding: '1rem 2rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.5rem',
  },
  imageContainer: {
    flex: '1 1 60%',
    maxWidth: '800px',
  },
  buttonContainer: {
    flex: '0 1 auto',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: '100%',
    borderRadius: '8px',
  },
  ButtonUpload: {
    backgroundColor: '#004d00',
    color: '#fff',
    border: 'none',
    padding: '1rem 2rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.5rem',
    marginTop: '30px',
    justifyContent: 'center'
  }
};

export default Plan1;
import React, { useState } from "react";
import dragDrop from "../images/DragDrop.png"

const fileTypes = ["PNG", "SOS"];

function DragDrop() {
  const [file, setFile] = useState(null);

  const handleChange = (file: File): void => {
  };

  return (
    <div style={styles.dropZone}>
      <div style={styles.content}>
        <img src={dragDrop} alt="upload icon" style={styles.icon} />
        <p style={styles.text}>Drag and Drop here</p>
      </div>
    </div>
  );
}

const styles = {
  dropZone: {
    width: '400px',
    padding: '30px',
    border: '2px dashed #ccc',
    borderRadius: '10px',
    backgroundColor: '#f0f0f0',
    textAlign: 'center' as const,
    cursor: 'pointer',
    position: 'relative' as const,
  },
  content: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    width: '50px',
    marginBottom: '10px',
  },
  text: {
    fontSize: '18px',
    fontWeight: 'bold',
    color: '#333',
  },
  orText: {
    fontSize: '16px',
    color: '#999',
    margin: '10px 0',
  },
  uploadButton: {
    backgroundColor: '#5cb85c',
    color: '#fff',
    padding: '12px 24px',
    borderRadius: '5px',
    cursor: 'pointer',
    fontSize: '16px',
    border: 'none',
    textDecoration: 'none',
  },
};

export default DragDrop;


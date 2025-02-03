import Layout from '../components/Layout';
import React, { useState } from "react";
import DragDrop from '../components/DragDrop';
import CustomInput from '../components/CustomInput';


const Plan1 = () => {
  return (
    <Layout>
      <div style={styles.container}>
        <h2>Last opp planforslag her:</h2>
        <CustomInput></CustomInput>
        <DragDrop></DragDrop>
        <button style={styles.ButtonUpload}>
          Upload
        </button>
      </div>
    </Layout>
  );
};
const styles = {
  container: {
    flex: '1', 
    paddingLeft: '600px',
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
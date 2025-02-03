import Layout from '../components/Layout';
import React, { useState } from "react";
import DragDrop from '../components/DragDrop';
import CustomInput from '../components/CustomInput';


const Plan2 = () => {
  return (
    <Layout>
      {/* Flexbox container for image and button */}
      <div style={styles.container}>
        {/* Left side*/}
        <div style={styles.LeftContainer}>
        <h2>Last opp planforslag her:</h2>
        <CustomInput></CustomInput>
        <DragDrop></DragDrop>
        </div>

        {/*Middle*/}
        <div style={styles.MiddleContainer}>
        <h2>Kart Analyse: </h2>
        </div>
        
        {/* Right side*/}
        <div style={styles.RightContainer}>
        <h2>Bemerkelser: </h2>
        </div>
      </div>
    </Layout>
  );
};
const styles = {
  container: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between', 
    gap: '2rem', 
    marginTop: '0rem',
    flexWrap: 'wrap' as const, 
  },
  LeftContainer: {
    flex: '1',  
    textAlign: 'left' as const,
  },
  MiddleContainer: {
    border: 'solid, 2px',
    borderRadius: '10px',
    backgroundColor: '#f0f0f0',
    height: '800px',
    flex: '1', 
    paddingLeft: '30px',
    textAlign: 'left' as const,
  },
  RightContainer: {
    border: 'solid, 2px',
    borderRadius: '10px',
    backgroundColor: '#f0f0f0',
    height: '800px',
    flex: '1',  
    paddingLeft: '30px',  
    textAlign: 'left' as const,
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
};

export default Plan2;
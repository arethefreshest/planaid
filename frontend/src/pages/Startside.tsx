import React from 'react';
import { Link } from "react-router-dom";
import Layout from '../components/Layout';
import bygg3d from '../images/bygg3d.png';

const Start = () => {
  return (
    <Layout>
      <div style={{ textAlign: 'center', marginBottom: '10rem' }}>
        <h1 style={{fontSize: '50px'}}>Velkommen til PlanAid!</h1>
        <p style={{fontSize: '25px'}}>Her kan du analysere planforslag.</p>
      </div>

      {/* Flexbox container for image and button */}
      <div style={styles.container}>
        {/* Left side: Image */}
        <div style={styles.imageContainer}>
          <img
            src={bygg3d}
            alt="3D building visualization"
            style={styles.image}
          />
        </div>

        {/* Right side: Button */}
        <div style={styles.buttonContainer}>
        <Link to="/page2" style={styles.button}>
          Start analyse
        </Link>
        </div>
      </div>
    </Layout>
  );
};

const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10rem', 
    marginTop: '2rem',
    flexWrap: 'wrap' as const, 
  },
  imageContainer: {
    flex: '1 1 60%', 
    maxWidth: '600px', 
  },
  buttonContainer: {
    flex: '0 1 auto', 
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    backgroundColor: '#24BD76',
    color: '#fff',
    border: 'none',
    padding: '1rem 3rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '2rem',
  },
  image: {
    width: '100%',
    borderRadius: '8px',
  },
};

export default Start;

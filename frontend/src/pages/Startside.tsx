import React from 'react';
import Layout from '../components/Layout';
import bygg3d from '../images/bygg3d.png';

const Start = () => {
  return (
    <Layout>
      {/* Center the title and subtitle */}
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
            alt="Example Image"
            style={styles.image}
          />
        </div>

        {/* Right side: Button */}
        <div style={styles.buttonContainer}>
          <button style={styles.tryButton}>Prøv Nå!</button>
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
    gap: '10rem', // Space between image and button
    marginTop: '2rem',
    flexWrap: 'wrap' as const, // Add type assertion here
  },
  imageContainer: {
    flex: '1 1 60%', // Allows image to take up more space
    maxWidth: '800px', // Optional: Limits max size of the image
  },
  buttonContainer: {
    flex: '0 1 auto', // Adjusts button size while maintaining proportions
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  tryButton: {
    backgroundColor: '#004d00',
    color: '#fff',
    border: 'none',
    padding: '1rem 2rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '3rem',
  },
  image: {
    width: '100%',
    borderRadius: '8px',
  },
};

export default Start;

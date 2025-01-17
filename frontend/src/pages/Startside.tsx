import React from 'react';
import Layout from '../components/Layout';
import bygg3d from '../images/bygg3d.png';

const Start = () => {
  return (
    <Layout>
      <div style={{ textAlign: 'center' }}>
        <h1>Velkommen til PlanAid!</h1>
        <p>Her kan du analysere planforslag.</p>
        <button style={styles.tryButton}>Prøv Nå!</button>
        <img
          src={bygg3d}
          alt="Example Image"
          style={styles.image}
        />
      </div>
    </Layout>
  );
};

const styles = {
  tryButton: {
    backgroundColor: '#004d00',
    color: '#fff',
    border: 'none',
    padding: '1rem 2rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.2rem',
  },
  image: {
    width: '100%',
    marginTop: '2rem',
    borderRadius: '8px',
  },
};

export default Start;
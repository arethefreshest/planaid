import React from 'react';

const CustomInput = () => {
  return (
    <div style={styles.container}>
      <label style={styles.label} htmlFor="saksId">Saks Id:</label>
      <input type="text" id="saksId" placeholder="Skriv inn Saks Id" style={styles.input} />
    </div>
  );
};

const styles = {
  container: { 
    display: 'flex', 
    flexDirection: 'row' as const, 
    gap: '8px', 
    width: '300px', 
    marginBottom: '20px' 
  },
  label: { fontSize: '15px', fontWeight: 'bold', color: '#333', width: '130px', padding: '5px' },
  input: { backgroundColor: '#f0f0f0', border: '1px solid #ccc', borderRadius: '10px', padding: '10px 10px', fontSize: '10px', outline: 'none', width: '100%' }
};

export default CustomInput;
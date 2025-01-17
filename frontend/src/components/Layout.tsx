import React from 'react';
import { Link } from 'react-router-dom';
import logo from '../images/LogoPlanAid.png';


const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logo}>
          <img src={logo} alt="PlanAid Logo" style={styles.logoImage} />
        </div>
        <button style={styles.loginButton}>Logg inn</button>
      </header>

      {/* Main Content */}
      <main style={styles.main}>{children}</main>

      {/* Footer */}
      <footer style={styles.footer}>
        <div>
          <span>© 2025 PlanAid</span>
        </div>
        <div style={styles.socialLinks}>
          <a href="https://x.com">X</a>
          <a href="https://instagram.com">Instagram</a>
          <a href="https://linkedin.com">LinkedIn</a>
        </div>
      </footer>
    </div>
  );
};

const styles = {
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1rem 2rem',
    backgroundColor: '#fff',
    borderBottom: '1px solid #ccc',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  logoImage: {
    height: '40px',
  },
  loginButton: {
    backgroundColor: '#004d00',
    color: '#fff',
    border: 'none',
    padding: '0.5rem 1rem',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  main: {
    padding: '2rem',
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1rem 2rem',
    backgroundColor: '#f0f0f0',
    borderTop: '1px solid #ccc',
  },
  socialLinks: {
    display: 'flex',
    gap: '1rem',
  },
};

export default Layout;
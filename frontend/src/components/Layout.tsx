import React from 'react';
import { Link } from 'react-router-dom';
import logo from '../images/planlogopng1.png'
import xIcon from '../images/x.png';
import instaIcon from '../images/insta.png';
import linkedIcon from '../images/in.png';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div style={styles.pageContainer}>
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
          <a href="https://x.com/thefunnytweeter/status/1668518247938023425" target="_blank" rel="noopener noreferrer">
            <img src={xIcon} alt="X" style={styles.icon} />
          </a>
          <a href="https://www.instagram.com/realfunnyreels/reels/" target="_blank" rel="noopener noreferrer">
            <img src={instaIcon} alt="Instagram" style={styles.icon} />
          </a>
          <a href="https://linkedin.com/in/olesveinungberget/" target="_blank" rel="noopener noreferrer">
            <img src={linkedIcon} alt="LinkedIn" style={styles.icon} />
          </a>
        </div>
      </footer>
    </div>
  );
};

const styles = {
  pageContainer: {
    display: 'flex',
    flexDirection: 'column' as const,
    minHeight: '100vh',
    backgroundColor: '#f9f9f9',
  },
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
    height: '60px',
  },
  loginButton: {
    backgroundColor: '#24BD76',
    color: '#fff',
    border: 'none',
    padding: '0.5rem 1rem',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  main: {
    flex: '1',
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
  icon: {
    width: '20px',
    height: '20px',
  },
};

export default Layout;
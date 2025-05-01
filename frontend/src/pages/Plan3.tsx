import ComparePdf from "../components/ComparePdf";
import Layout from "../components/Layout";
import { Link } from "react-router-dom";


const Plan3 = () => (
  <Layout>
    <ComparePdf />
    <Link to="/page2" style={styles.button}>
      Gå til Konsistenssjekk
    </Link>
  </Layout>
);
const styles = {
  button: {
    position: 'fixed' as const,
    bottom: '20px',
    right: '20px',
    backgroundColor: '#24BD76',
    color: '#fff',
    border: 'none',
    padding: '1rem 3rem',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1rem',
    zIndex: 1000, 
    textDecoration: 'none'
  }
};

export default Plan3;
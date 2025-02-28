import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Start from './pages/Startside';
import Page1 from './pages/Plan1';
import Page2 from './pages/Plan2';

function App() {
  return (
    <Router>
      <div>
        <nav style={{ padding: '1rem', backgroundColor: '#f0f0f0' }}>
          <Link to="/" style={{ marginRight: '10px' }}>Home</Link>
          <Link to="/page1" style={{ marginRight: '10px' }}>Page 1</Link>
          <Link to="/page2" style={{ marginRight: '10px' }}>Page 2</Link>
        </nav>
        <Routes>
          <Route path="/" element={<Start />} />
          <Route path="/page1" element={<Page1 />} />
          <Route path="/page2" element={<Page2 />} />

        </Routes>
      </div>
    </Router>
  );
}

export default App;

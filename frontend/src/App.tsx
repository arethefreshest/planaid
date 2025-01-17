import React from 'react';
import './App.css';
import FileUpload from './components/FileUpload';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>PlanAid</h1>
        <FileUpload />
      </header>
    </div>
  );
}

export default App;

import React from 'react';
import ChatPanel from './components/ChatPanel';
import Sidebar from './components/Sidebar';
import "./App.css"

function App() {
  return (
    <div style={{ position: 'relative', minHeight: '100vh', background: '#070b14' }}>

      {/* Animated Mesh Background */}
      <div className="bg-blobs">
        <div className="blob blob-1" />
        <div className="blob blob-2" />
        <div className="blob blob-3" />
      </div>

      {/* App Shell */}
      <div className="app-shell">
        <Sidebar />
        <ChatPanel />
      </div>

    </div>
  );
}

export default App;
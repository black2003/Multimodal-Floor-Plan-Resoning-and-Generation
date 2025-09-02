import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { SocketProvider } from './contexts/SocketContext';
import { AnalysisProvider } from './contexts/AnalysisContext';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import AnalysisPage from './pages/AnalysisPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import DocumentationPage from './pages/DocumentationPage';
import { Toaster } from './components/ui/toaster';

function App() {
  return (
    <ThemeProvider>
      <SocketProvider>
        <AnalysisProvider>
          <Router>
            <div className="min-h-screen bg-background">
              <Layout>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/analyze" element={<AnalysisPage />} />
                  <Route path="/explain" element={<ExplainabilityPage />} />
                  <Route path="/docs" element={<DocumentationPage />} />
                </Routes>
              </Layout>
              <Toaster />
            </div>
          </Router>
        </AnalysisProvider>
      </SocketProvider>
    </ThemeProvider>
  );
}

export default App;

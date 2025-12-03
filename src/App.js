import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Reports from './pages/Reports';
import Contact from './pages/Contact';
import FAQ from './pages/FAQ';
import MainPage from './pages/main_page';
import CrowdCountPhoto from './pages/crowd_count_photo';

function App() {
  return (
    <div className="App">
      <Navbar />
      <div className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/main_page" element={<MainPage />} />
          <Route path='/crowd_count_photo' element={<CrowdCountPhoto />} />
        </Routes>
      </div>
      <Footer />
    </div>
  );
}

export default App;


import React, { useState } from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  // Close menu on link click (for better UX)
  const handleLinkClick = () => setMenuOpen(false);

  return (
    <nav className="navbar">
      <div className="navbar-left">
        <Link to="/" className="logo-link">
          <img src="/logo.png" alt="Logo" className="logo-img" />
          <div className="logo-text-group">
            <div className="logo-text">VigilNet</div>
            <div className="tagline">Real-time crowd monitoring & alerts</div>
          </div>
        </Link>
      </div>
      <div
        className={`navbar-hamburger${menuOpen ? " open" : ""}`}
        onClick={() => setMenuOpen((m) => !m)}
        aria-label="Open navigation"
        tabIndex={0}
        onKeyPress={(e) => { if (e.key === 'Enter') setMenuOpen((m) => !m); }}
      >
        <div />
        <div />
        <div />
      </div>
      <div className={`navbar-links${menuOpen ? " open" : ""}`}>
        <Link to="/" onClick={handleLinkClick}>Home</Link>
        <Link to="/main_page" onClick={handleLinkClick}>Analysis</Link>
        <Link to="/crowd_count_photo" onClick={handleLinkClick}>CCPhoto</Link>
        <Link to="/reports" onClick={handleLinkClick}>Reports</Link>
        <Link to="/contact" onClick={handleLinkClick}>Contact</Link>
        <Link to="/faq" onClick={handleLinkClick}>FAQ</Link>
      </div>
    </nav>
  );
}

export default Navbar;

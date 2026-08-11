import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import AnalyzePage from "./pages/AnalyzePage";
import HistoryPage from "./pages/HistoryPage";
import MetricsPage from "./pages/MetricsPage";
import PatientsPage from "./pages/PatientsPage";
import "./App.css";

function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("lungai_theme") || "light";
  });

  useEffect(() => {
    localStorage.setItem("lungai_theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  return (
    <BrowserRouter>
      <div className="app" data-theme={theme}>
        {/* ── Sidebar ─────────────────────────────────────────── */}
        <aside className="sidebar">
          <div className="sidebar-logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect width="28" height="28" rx="8" fill="#1D9E75"/>
              <path d="M9 14c0-2.8 2-5 5-5s5 2.2 5 5" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              <path d="M8 14v4a6 6 0 0012 0v-4" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="14" cy="12" r="1.5" fill="white"/>
            </svg>
            <div className="logo-badge-container">
              <span className="logo-text">LungAI</span>
              <span className="badge-tag">Academic</span>
            </div>
          </div>

          <nav className="sidebar-nav">
            <div className="sidebar-section-header">WORKSPACE</div>
            <NavLink to="/" end className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
              </svg>
              Analyze
            </NavLink>
            <NavLink to="/history" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
              History
            </NavLink>

            <div className="sidebar-section-header" style={{ marginTop: "1rem" }}>INSIGHTS</div>
            <NavLink to="/metrics" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3v18h18M8 17l4-4 4 4 4-8"/>
              </svg>
              Model Performance
            </NavLink>
            <NavLink to="/patients" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/>
                <path d="M23 21v-2a4 4 0 010 7.75"/>
              </svg>
              Patients
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            <button className="theme-toggle-btn" onClick={toggleTheme} title="Toggle Dark/Light Mode">
              <span>{theme === "light" ? "🌙 Dark Theme" : "☀️ Light Theme"}</span>
            </button>
            <div className="sidebar-disclaimer-wrapper">
              <span className="disclaimer-badge">Academic Prototype</span>
              <p className="version-text">LungAI v1.0 · Not for clinical diagnosis</p>
            </div>
          </div>
        </aside>

        {/* ── Main Content ────────────────────────────────────── */}
        <main className="main-content">
          <Routes>
            <Route path="/"         element={<AnalyzePage />} />
            <Route path="/history"  element={<HistoryPage />} />
            <Route path="/metrics"  element={<MetricsPage />} />
            <Route path="/patients" element={<PatientsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;

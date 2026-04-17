import React from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import AnalyzePage from "./pages/AnalyzePage";
import HistoryPage from "./pages/HistoryPage";
import MetricsPage from "./pages/MetricsPage";
import PatientsPage from "./pages/PatientsPage";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        {/* ── Sidebar ─────────────────────────────────────────── */}
        <aside className="sidebar">
          <div className="sidebar-logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect width="28" height="28" rx="8" fill="#1D9E75"/>
              <path d="M9 14c0-2.8 2-5 5-5s5 2.2 5 5" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              <path d="M8 14v4a6 6 0 0012 0v-4" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="14" cy="12" r="1.5" fill="white"/>
            </svg>
            <span className="logo-text">LungAI</span>
          </div>

          <nav className="sidebar-nav">
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
            <NavLink to="/metrics" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3v18h18M8 17l4-4 4 4 4-8"/>
              </svg>
              Model Metrics
            </NavLink>
            <NavLink to="/patients" className={({isActive}) => isActive ? "nav-item active" : "nav-item"}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/>
                <path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75"/>
              </svg>
              Patients
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            <p className="disclaimer-badge">Academic use only</p>
            <p className="version-text">LungAI v1.0 · Not a medical device</p>
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

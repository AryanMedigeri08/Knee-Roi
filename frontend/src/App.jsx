import React, { useState, useRef } from 'react';
import './index.css';

const severityColors = {
  "Healthy": "#22C55E",
  "Doubtful": "#F59E0B",
  "Minimal": "#F97316",
  "Moderate": "#F5811F",
  "Severe": "#EF4444"
};

const findingIcons = {
  "Joint Space Narrowing": "🦴",
  "Osteophyte Formation": "🔬",
  "Sclerosis": "🧫",
  "Alignment": "📐"
};

function App() {
  const [activeTab, setActiveTab] = useState('Home');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResults(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.success) {
        setResults(data);
        setActiveTab('Insights');
      }
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Analysis failed. Make sure backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!results) return;
    try {
      const res = await fetch('http://localhost:8000/download_pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_text: results.results.report_text })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "knee_xray_analysis_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Download failed:", e);
    }
  };

  return (
    <div className="app-container">
      <div className="main-content-wrapper">

        {/* ── SIDEBAR ── */}
        <nav className="sidebar">
          <div className="sidebar-logo">
            <img src="/img/logo.png" alt="Logo" className="sidebar-logo-img" />
            <span className="sidebar-logo-text">OSTEO <em>VISION</em></span>
          </div>

          <div className="sidebar-nav">
            <button className={`nav-item ${activeTab === 'Home' ? 'active' : ''}`} onClick={() => setActiveTab('Home')}>
              <img src="/img/home.png" alt="" className="nav-icon" />
              <span className="nav-label">Home</span>
            </button>
            <button className={`nav-item ${activeTab === 'History' ? 'active' : ''}`} onClick={() => setActiveTab('History')}>
              <img src="/img/file.png" alt="" className="nav-icon" />
              <span className="nav-label">History</span>
            </button>
            <button className={`nav-item ${activeTab === 'Insights' ? 'active' : ''}`} onClick={() => setActiveTab('Insights')}>
              <img src="/img/report.png" alt="" className="nav-icon" />
              <span className="nav-label">Insights</span>
            </button>
          </div>

            {/* Sidebar Promo & Doctor Profile */}
          <div className="sidebar-bottom">
            <div className="promo-card">
              <div className="promo-title">AI-Powered Analysis</div>
              <p className="promo-desc">Advanced deep learning models for early detection and better outcomes.</p>
            </div>

            <div className="doc-profile">
              <img src="https://images.unsplash.com/photo-1622253692010-333f2da6031d?auto=format&fit=crop&q=80&w=100&h=100" className="doc-avatar" alt="Dr. Aditya Sharma" />
              <div className="doc-info">
                <div className="doc-name">Dr. Aryan Medigeri</div>
                <div className="doc-role">Orthopedic Surgeon</div>
              </div>
              <span className="doc-arrow">▾</span>
            </div>
          </div>
        </nav>

        {/* ── MAIN AREA ── */}
        <main className="main-area">
          {/* Header Actions */}
          <div className="header-actions">
            <div className="notif-bell">
              <img src="/img/bell.png" alt="Notifications" />
              <span>3</span>
            </div>
            <div className="user-profile">
              <div className="user-avatar">AS</div>
              <span className="user-arrow">▾</span>
            </div>
          </div>

          <div className="block-container">

            {/* ═══ HOME TAB ═══ */}
            {activeTab === 'Home' && (
              <div className="home-split-layout fade-in">
                {/* Left Side: Content */}
                <div className="home-left">
                  <div className="home-header">
                    <h1 className="home-title">Knee <span style={{ color: 'var(--blue)' }}>X-ray</span> Analysis</h1>
                    <p className="home-subtitle">AI-powered insights for better knee health</p>
                  </div>

                  {/* Feature Cards Row */}
                  <div className="feature-cards">
                    <div className="feature-card">
                      <div className="feature-icon feature-icon--blue">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>
                      </div>
                      <div>
                        <div className="feature-title">Accurate Results</div>
                        <div className="feature-desc">High precision analysis</div>
                      </div>
                    </div>
                    <div className="feature-card">
                      <div className="feature-icon feature-icon--blue">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
                      </div>
                      <div>
                        <div className="feature-title">Fast Processing</div>
                        <div className="feature-desc">Results in seconds</div>
                      </div>
                    </div>
                    <div className="feature-card">
                      <div className="feature-icon feature-icon--blue">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><rect x="3" y="11" width="18" height="11" rx="2" ry="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></svg>
                      </div>
                      <div>
                        <div className="feature-title">Secure & Private</div>
                        <div className="feature-desc">Your data is safe</div>
                      </div>
                    </div>
                  </div>

                  {/* Upload Area */}
                  <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
                    <input ref={fileInputRef} type="file" accept=".jpg,.jpeg,.png" onChange={handleFileChange} style={{ display: 'none' }} />
                    <div className="upload-circle">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <path d="M21.2 15c.7-1.2 1-2.5.7-3.9-.3-2-1.8-3.6-3.8-3.9C17 4.1 14.2 2 11 2 7.7 2 5.1 4.3 4.2 7.5c-2.3.2-4.1 2-4.2 4.3C0 14.3 2 16.5 4.5 16.5H19c1 0 1.8-.7 2.2-1.5z" />
                        <polyline points="16 12 12 8 8 12" />
                        <line x1="12" y1="8" x2="12" y2="16" />
                      </svg>
                    </div>
                    <div className="upload-title">Upload Knee X-ray</div>
                    <div className="upload-formats">JPG, PNG or DICOM</div>

                    <button className="btn-primary" style={{ width: 'auto', minWidth: '150px', padding: '0.6rem 2.2rem', margin: '0 auto 0.6rem' }} onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                      Choose File
                    </button>

                    <div className="upload-note">or drag and drop your file here</div>
                  </div>

                  {/* Preview Section */}
                  {preview && (
                    <div className="preview-section fade-in">
                      <img src={preview} alt="Preview" className="preview-img" />
                      <button className="btn-primary btn-analyze" onClick={handleAnalyze} disabled={loading}>
                        {loading ? (
                          <span className="btn-loading">
                            <span className="spinner"></span> Analyzing...
                          </span>
                        ) : '🔬 Analyze X-Ray'}
                      </button>
                    </div>
                  )}

                  {/* Recent Analyses integrated directly on Left Column */}
                  <div className="recent-analyses-container">
                    <div className="history-header">
                      <h3>Recent Analyses</h3>
                      <a href="#" className="view-all-link" onClick={(e) => { e.preventDefault(); setActiveTab('History'); }}>View All</a>
                    </div>

                    <div className="recent-row" onClick={() => setActiveTab('History')}>
                      <div className="recent-thumb"><img src="/img/minimal_xray.png" alt="Mild X-Ray" /></div>
                      <div className="recent-meta">
                        <div className="recent-date">May 20, 2024 · 10:30 AM</div>
                        <div className="recent-knee">Right Knee</div>
                      </div>
                      <div className="recent-badge mild">Mild (24%)</div>
                      <div className="recent-arrow">›</div>
                    </div>

                    <div className="recent-row" onClick={() => setActiveTab('History')}>
                      <div className="recent-thumb"><img src="/img/moderate_xray.png" alt="Moderate X-Ray" /></div>
                      <div className="recent-meta">
                        <div className="recent-date">May 18, 2024 · 02:15 PM</div>
                        <div className="recent-knee">Left Knee</div>
                      </div>
                      <div className="recent-badge moderate">Moderate (52%)</div>
                      <div className="recent-arrow">›</div>
                    </div>
                  </div>
                </div>

                {/* Right Side: Huge Knee Image occupying full vertical height */}
                <div className="home-right">
                  <img src="/img/hero_knee.png" className="full-hero-img" alt="" />
                </div>
              </div>
            )}

            {/* ═══ INSIGHTS TAB ═══ */}
            {activeTab === 'Insights' && (
              <div className="fade-in">
                {!results ? (
                  <div className="empty-state">
                    <div className="empty-icon">📊</div>
                    <h3>No Analysis Available</h3>
                    <p>Please upload and analyze an image from the Home tab first.</p>
                    <button className="btn-primary" style={{ maxWidth: '300px', margin: '1.5rem auto 0' }} onClick={() => setActiveTab('Home')}>Go to Home</button>
                  </div>
                ) : (
                  <>
                    <div className="insights-header">
                      <div className="insights-header-icon">✨</div>
                      <div>
                        <h2 style={{ marginBottom: '0.3rem' }}>Knee Osteoarthritis Severity Analysis</h2>
                        <p className="insights-subtitle">Deterministic ROI Extraction • Xception CNN • Deep Learning Classification</p>
                      </div>
                    </div>

                    <div className="columns">
                      <div className="col-left">
                        <div className="step-header">
                          <span className="step-badge">Step 1</span>
                          <h3 style={{ margin: 0 }}>ROI Extraction Pipeline</h3>
                        </div>
                        <p className="step-desc">Deterministic classical image processing – no training data required.</p>

                        <div className="pipeline-row">
                          <div className="pipeline-card">
                            <img src={`data:image/png;base64,${results.images.input}`} alt="Input" />
                            <p>Input X-Ray</p>
                          </div>
                          <div className="pipeline-arrow">›</div>
                          <div className="pipeline-card">
                            <img src={`data:image/png;base64,${results.images.norm}`} alt="Normalized" />
                            <p>Normalized</p>
                          </div>
                          <div className="pipeline-arrow">›</div>
                          <div className="pipeline-card">
                            <img src={`data:image/png;base64,${results.images.otsu}`} alt="Otsu" />
                            <p>Otsu Threshold</p>
                          </div>
                          <div className="pipeline-arrow">›</div>
                          <div className="pipeline-card">
                            <img src={`data:image/png;base64,${results.images.morph}`} alt="Morph" />
                            <p>Morphological Closing</p>
                          </div>
                          <div className="pipeline-arrow">›</div>
                          <div className="pipeline-card">
                            <img src={`data:image/png;base64,${results.images.roi}`} alt="ROI" />
                            <p>Extracted ROI <span className="checkmark">✅</span></p>
                          </div>
                        </div>

                        <div className="step-header" style={{ marginTop: '2.5rem' }}>
                          <span className="step-badge step-badge--purple">Step 2</span>
                          <h3 style={{ margin: 0 }}>OA Severity Classification</h3>
                        </div>

                        <div className="classification-row">
                          <div className="classification-card">
                            <h4>Input to CNN (ROI)</h4>
                            <div className="xray-img-wrap">
                              <img src={`data:image/png;base64,${results.images.roi}`} alt="ROI" />
                            </div>
                          </div>
                          <div className="classification-card">
                            <h4>Grad-CAM Explainability</h4>
                            <div className="xray-img-wrap">
                              <img src={`data:image/png;base64,${results.images.gradcam}`} alt="Grad-CAM" />
                            </div>
                          </div>
                        </div>

                        <div className="pipeline-footer">
                          Framework: Deterministic ROI Extraction • Xception CNN
                        </div>
                      </div>

                      <div className="col-right">
                        <div className="analysis-panel">
                          <div className="panel-header">
                            <button className="panel-back" onClick={() => setActiveTab('Home')}>‹</button>
                            <h3 style={{ margin: 0 }}>Analysis Result</h3>
                            <div></div>
                          </div>

                          <div className="severity-section">
                            <div className="severity-info">
                              <div className="sev-overview-label">Severity Overview</div>
                              <div className="sev-value" style={{ color: severityColors[results.results.grade] }}>
                                {results.results.grade}
                              </div>
                              <div className="sev-pct" style={{ color: severityColors[results.results.grade] }}>
                                {results.results.probability.toFixed(2)}%
                              </div>
                              <div className="kl-info">
                                <span className="kl-label">Kellgren–Lawrence Grade</span>
                                <span className="kl-value" style={{ color: severityColors[results.results.grade] }}>{results.results.kl_grade}</span>
                              </div>
                              <p className="sev-desc">{results.results.description}</p>
                            </div>
                            <div className="mini-ring-wrap">
                              <svg viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="45" fill="none" stroke="#F3F4F6" strokeWidth="8" />
                                <circle cx="50" cy="50" r="45" fill="none" stroke={severityColors[results.results.grade]} strokeWidth="8" strokeLinecap="round" strokeDasharray="282" strokeDashoffset={282 - (results.results.probability / 100) * 282} style={{ transition: 'stroke-dashoffset 1.5s ease-out' }} />
                              </svg>
                              <div className="ring-center">
                                <span className="ring-pct">{results.results.probability.toFixed(1)}%</span>
                                <span className="ring-sub">Severity Score</span>
                              </div>
                            </div>
                          </div>

                          <div className="prob-section">
                            <h4>Class Probability Distribution</h4>
                            {Object.entries(results.results.probabilities).map(([name, val]) => (
                              <div key={name} className="prob-bar-row">
                                <div className="prob-name">{name}</div>
                                <div className="prob-track">
                                  <div className="prob-fill" style={{ width: `${val}%`, background: severityColors[name] || '#ccc' }}></div>
                                </div>
                                <div className="prob-val">{val.toFixed(2)}%</div>
                              </div>
                            ))}
                          </div>

                          <div className="findings-section">
                            <h4>Key Findings</h4>
                            {Object.entries(results.results.findings).map(([name, status]) => (
                              <div key={name} className="find-row">
                                <div className="find-ico">{findingIcons[name] || '🔬'}</div>
                                <div className="find-name">{name}</div>
                                <div className={`find-status fs-${status}`}>{status}</div>
                              </div>
                            ))}
                          </div>

                          <div className="action-buttons">
                            <button className="btn-outline" onClick={() => {
                              const subject = "Knee X-ray Analysis Report";
                              const body = encodeURIComponent(results.results.report_text);
                              window.location.href = `mailto:?subject=${subject}&body=${body}`;
                            }}>
                              <svg viewBox="0 0 24 24" fill="none" style={{ width: 18, height: 18 }}><path d="M4 12v8a2 2 0 002 2h12a2 2 0 002-2v-8M16 6l-4-4-4 4M12 2v13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                              Share Report
                            </button>
                            <button className="btn-download" onClick={downloadReport}>
                              <svg viewBox="0 0 24 24" fill="none" style={{ width: 18, height: 18 }}><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                              Download Report
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* ═══ HISTORY TAB ═══ */}
            {activeTab === 'History' && (
              <div className="fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
                <div className="history-header">
                  <h3>Recent Analyses</h3>
                  <a href="#" className="view-all-link">View All</a>
                </div>

                <div className="recent-row">
                  <div className="recent-thumb"><img src="/img/minimal_xray.png" alt="Mild X-Ray" /></div>
                  <div className="recent-meta">
                    <div className="recent-date">May 20, 2024 · 10:30 AM</div>
                    <div className="recent-knee">Right Knee</div>
                  </div>
                  <div className="recent-badge mild">Mild (24%)</div>
                  <div className="recent-arrow">›</div>
                </div>

                <div className="recent-row">
                  <div className="recent-thumb"><img src="/img/moderate_xray.png" alt="Moderate X-Ray" /></div>
                  <div className="recent-meta">
                    <div className="recent-date">May 18, 2024 · 02:15 PM</div>
                    <div className="recent-knee">Left Knee</div>
                  </div>
                  <div className="recent-badge moderate">Moderate (52%)</div>
                  <div className="recent-arrow">›</div>
                </div>
              </div>
            )}

          </div>
        </main>
      </div>
    </div>
  );
}

export default App;

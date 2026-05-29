import { useState, useCallback } from "react";
import SequenceInput from "./components/SequenceInput";
import SSViewer from "./components/SSViewer";
import HPViewer from "./components/HPViewer";
import EnergyViewer from "./components/EnergyViewer";
import ValidationViewer from "./components/ValidationViewer";

const TABS = [
  { id: "ss",     label: "Secondary Structure" },
  { id: "hp",     label: "HP Lattice" },
  { id: "energy", label: "Energy Minimization" },
  { id: "rama",   label: "Validation" },
];

const API = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

export default function App() {
  const [tab,        setTab]     = useState("ss");
  const [sequence,   setSeq]     = useState("");
  const [loading,    setLoading] = useState(false);
  const [error,      setError]   = useState(null);
  const [ssResult,   setSS]      = useState(null);
  const [hpResult,   setHP]      = useState(null);
  const [enResult,   setEn]      = useState(null);
  const [valResult,  setVal]     = useState(null);

  const analyze = useCallback(async (seq) => {
    setSeq(seq);
    setLoading(true);
    setError(null);
    try {
      const [ssRes, hpRes, enRes, valRes] = await Promise.all([
        fetch(`${API}/predict_ss`, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ sequence: seq }) }).then(r=>r.json()),
        fetch(`${API}/hp_fold`,    { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ sequence: seq.slice(0,40), steps: 5000 }) }).then(r=>r.json()),
        fetch(`${API}/minimize`,   { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ sequence: seq.slice(0,30), max_steps: 300 }) }).then(r=>r.json()),
        fetch(`${API}/validate`,   { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ sequence: seq }) }).then(r=>r.json()),
      ]);
      if (ssRes.error)  throw new Error(ssRes.error);
      setSS(ssRes); setHP(hpRes); setEn(enRes); setVal(valRes);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem 1rem", fontFamily: "system-ui, sans-serif" }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
        <div style={{ width: 40, height: 40, borderRadius: 10, background: "#1D9E75", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>🧬</div>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 500, margin: 0 }}>ProteinIQ</h1>
          <p style={{ fontSize: 13, color: "#666", margin: 0 }}>Integrated protein structure prediction & analysis pipeline</p>
        </div>
      </div>

      {/* Sequence Input */}
      <SequenceInput onAnalyze={analyze} loading={loading} />

      {/* Error */}
      {error && (
        <div style={{ background: "#FCEBEB", border: "0.5px solid #F09595", borderRadius: 8, padding: "10px 14px", marginBottom: 16, fontSize: 13, color: "#A32D2D" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, borderBottom: "0.5px solid #ddd", marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: "8px 16px", fontSize: 13, fontWeight: 500, border: "none",
              background: "transparent", cursor: "pointer",
              color: tab === t.id ? "#1D9E75" : "#666",
              borderBottom: tab === t.id ? "2px solid #1D9E75" : "2px solid transparent",
              marginBottom: -1,
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Panels */}
      {tab === "ss"     && <SSViewer result={ssResult} />}
      {tab === "hp"     && <HPViewer result={hpResult} />}
      {tab === "energy" && <EnergyViewer result={enResult} />}
      {tab === "rama"   && <ValidationViewer result={valResult} />}

      {/* Footer */}
      <div style={{ marginTop: 32, paddingTop: 16, borderTop: "0.5px solid #eee", fontSize: 12, color: "#999", display: "flex", justifyContent: "space-between" }}>
        <span>ProteinIQ · Built with Flask + React</span>
        <span>KNN · HP Model · CHARMM Energy · Ramachandran Validation</span>
      </div>
    </div>
  );
}

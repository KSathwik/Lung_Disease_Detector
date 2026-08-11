import React, { useEffect, useState } from "react";
import { listPatients, createPatient, updatePatient } from "../services/api";

const EMPTY_FORM = { name:"", age:"", gender:"Male", contact:"", email:"", medical_history:"" };

const Field = ({ label, onChange, ...props }) => (
  <div className="form-field">
    <label className="field-label">{label}</label>
    <input className="text-input" {...props} onChange={onChange} />
  </div>
);

export default function PatientsPage() {
  const [patients,  setPatients]  = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [search,    setSearch]    = useState("");
  const [showForm,  setShowForm]  = useState(false);
  const [saving,    setSaving]    = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [form,      setForm]      = useState(EMPTY_FORM);

  const load = () => {
    setLoading(true);
    listPatients()
      .then(d => setPatients(d.patients || []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const openCreate = () => {
    setEditingId(null);
    setForm(EMPTY_FORM);
    setError(null);
    setShowForm(true);
  };

  const openEdit = (p) => {
    setEditingId(p.patient_id);
    setForm({
      name: p.name || "",
      age: p.age != null ? String(p.age) : "",
      gender: p.gender || "Male",
      contact: p.contact || "",
      email: p.email || "",
      medical_history: p.medical_history || "",
    });
    setError(null);
    setShowForm(true);
  };

  const closeForm = () => {
    setShowForm(false);
    setEditingId(null);
    setForm(EMPTY_FORM);
  };

  const handleSubmit = async () => {
    if (!form.name || !form.age) return;
    setSaving(true);
    try {
      const payload = { ...form, age: parseInt(form.age) };
      if (editingId) {
        await updatePatient(editingId, payload);
      } else {
        await createPatient(payload);
      }
      closeForm();
      load();
    } catch(e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const handleField = (e) =>
    setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const filteredPatients = patients.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    (p.contact && p.contact.toLowerCase().includes(search.toLowerCase())) ||
    (p.email && p.email.toLowerCase().includes(search.toLowerCase())) ||
    (p.patient_id && p.patient_id.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="page">
      <div className="page-header" style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start", flexWrap:"wrap", gap:"12px"}}>
        <div>
          <h1 className="page-title">Patient Directory</h1>
          <p className="page-subtitle">{patients.length} patients registered in the clinical database.</p>
        </div>
        <button className="btn-primary" onClick={() => (showForm ? closeForm() : openCreate())} style={{maxWidth:"200px"}}>
          {showForm ? "Cancel" : "+ Register Patient"}
        </button>
      </div>

      {/* ── Search Bar (Item 3) ──────────────────────────────────── */}
      <div className="toolbar-row">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Search patients by name, contact, email or ID…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {showForm && (
        <div className="card" style={{marginBottom:"1.5rem"}}>
          <p className="section-label" style={{marginBottom:"1rem"}}>
            {editingId ? "Edit Patient Details" : "Register New Patient"}
          </p>
          <div className="form-grid">
            <Field label="Full name *"    name="name"    value={form.name}    placeholder="e.g. Ravi Kumar" onChange={handleField} />
            <Field label="Age *"          name="age"     value={form.age}     placeholder="e.g. 45" type="number" onChange={handleField} />
            <div className="form-field">
              <label className="field-label">Gender</label>
              <select className="select-input" value={form.gender} onChange={e => setForm(f => ({...f, gender: e.target.value}))}>
                <option>Male</option><option>Female</option><option>Other</option>
              </select>
            </div>
            <Field label="Contact"        name="contact" value={form.contact} placeholder="+91 98765 43210" onChange={handleField} />
            <Field label="Email"          name="email"   value={form.email}   placeholder="patient@email.com" onChange={handleField} />
          </div>
          <div className="form-field">
            <label className="field-label">Medical history</label>
            <textarea
              className="text-input"
              rows={3}
              name="medical_history"
              value={form.medical_history}
              placeholder="Existing pulmonary conditions, allergies, medications..."
              onChange={e => setForm(f => ({...f, medical_history: e.target.value}))}
              style={{resize:"vertical"}}
            />
          </div>
          <div style={{display:"flex", gap:"10px", marginTop:"8px"}}>
            <button className="btn-primary" onClick={handleSubmit} disabled={saving}>
              {saving ? "Saving…" : editingId ? "Save Changes" : "Register Patient"}
            </button>
            <button className="btn-ghost" onClick={closeForm}>Cancel</button>
          </div>
        </div>
      )}

      {/* ── Skeleton Loaders (Item 2) ────────────────────────────── */}
      {loading ? (
        <div className="patient-grid">
          {[1, 2, 3, 4, 5, 6].map(n => (
            <div key={n} className="card skeleton skeleton-card" style={{height:"110px"}} />
          ))}
        </div>
      ) : filteredPatients.length === 0 ? (
        <div className="card empty-state"><p>No matching patient records found.</p></div>
      ) : (
        <div className="patient-grid">
          {filteredPatients.map(p => (
            <div key={p.patient_id || p.id} className="card patient-card">
              <div className="patient-avatar">
                {p.name.split(" ").map(w => w[0]).join("").slice(0,2).toUpperCase()}
              </div>
              <div className="patient-info">
                <p className="patient-name">{p.name}</p>
                <p className="patient-meta">{p.age} yrs · {p.gender}</p>
                {p.contact && <p className="patient-meta">📞 {p.contact}</p>}
                {p.email   && <p className="patient-meta">✉️ {p.email}</p>}
                {p.medical_history && (
                  <p className="patient-meta" style={{marginTop:"4px", fontStyle:"italic", color:"var(--text-muted)"}}>
                    "{p.medical_history.length > 50 ? p.medical_history.slice(0, 50) + "…" : p.medical_history}"
                  </p>
                )}
              </div>
              <span className="patient-id-badge">{p.patient_id}</span>
              <button
                className="btn-secondary"
                style={{position:"absolute", bottom:"12px", right:"12px", padding:"4px 12px", fontSize:"12px"}}
                onClick={() => openEdit(p)}
              >
                Edit
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

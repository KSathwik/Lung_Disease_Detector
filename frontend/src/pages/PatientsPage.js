import React, { useEffect, useState } from "react";
import { listPatients, createPatient } from "../services/api";

export default function PatientsPage() {
  const [patients, setPatients] = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [saving,   setSaving]   = useState(false);
  const [form, setForm] = useState({ name:"", age:"", gender:"Male", contact:"", email:"", medical_history:"" });

  const load = () => {
    setLoading(true);
    listPatients()
      .then(d => setPatients(d.patients || []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleSubmit = async () => {
    if (!form.name || !form.age) return;
    setSaving(true);
    try {
      await createPatient({ ...form, age: parseInt(form.age) });
      setShowForm(false);
      setForm({ name:"", age:"", gender:"Male", contact:"", email:"", medical_history:"" });
      load();
    } catch(e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const Field = ({ label, ...props }) => (
    <div className="form-field">
      <label className="field-label">{label}</label>
      <input className="text-input" {...props} onChange={e => setForm(f => ({...f, [props.name]: e.target.value}))} />
    </div>
  );

  return (
    <div className="page">
      <div className="page-header" style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start"}}>
        <div>
          <h1 className="page-title">Patients</h1>
          <p className="page-subtitle">{patients.length} patients registered.</p>
        </div>
        <button className="btn-primary" onClick={() => setShowForm(v => !v)}>
          {showForm ? "Cancel" : "+ Register Patient"}
        </button>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {showForm && (
        <div className="card" style={{marginBottom:"1.5rem"}}>
          <p className="section-label" style={{marginBottom:"1rem"}}>New Patient</p>
          <div className="form-grid">
            <Field label="Full name *"    name="name"    value={form.name}    placeholder="e.g. Ravi Kumar" />
            <Field label="Age *"          name="age"     value={form.age}     placeholder="e.g. 45" type="number" />
            <div className="form-field">
              <label className="field-label">Gender</label>
              <select className="select-input" value={form.gender} onChange={e => setForm(f => ({...f, gender: e.target.value}))}>
                <option>Male</option><option>Female</option><option>Other</option>
              </select>
            </div>
            <Field label="Contact"        name="contact" value={form.contact} placeholder="+91 98765 43210" />
            <Field label="Email"          name="email"   value={form.email}   placeholder="patient@email.com" />
          </div>
          <div className="form-field">
            <label className="field-label">Medical history</label>
            <textarea
              className="text-input"
              rows={3}
              name="medical_history"
              value={form.medical_history}
              placeholder="Existing conditions, allergies, medications..."
              onChange={e => setForm(f => ({...f, medical_history: e.target.value}))}
              style={{resize:"vertical"}}
            />
          </div>
          <button className="btn-primary" onClick={handleSubmit} disabled={saving}>
            {saving ? "Saving…" : "Register Patient"}
          </button>
        </div>
      )}

      {loading ? (
        <p className="loading-text">Loading patients…</p>
      ) : patients.length === 0 ? (
        <div className="card empty-state"><p>No patients registered yet.</p></div>
      ) : (
        <div className="patient-grid">
          {patients.map(p => (
            <div key={p.id} className="card patient-card">
              <div className="patient-avatar">
                {p.name.split(" ").map(w => w[0]).join("").slice(0,2).toUpperCase()}
              </div>
              <div className="patient-info">
                <p className="patient-name">{p.name}</p>
                <p className="patient-meta">{p.age} yrs · {p.gender}</p>
                {p.contact && <p className="patient-meta">{p.contact}</p>}
                {p.email   && <p className="patient-meta">{p.email}</p>}
              </div>
              <span className="patient-id-badge">{p.patient_id}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

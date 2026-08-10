/**
 * API Service Layer
 * Communicates with FastAPI backend
 */

import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

// ── Request / Response interceptors ──────────────────────────────────────────
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err.response?.data?.detail;
    let msg;
    if (Array.isArray(detail)) {
      // FastAPI validation errors (422) come back as a list of objects
      msg = detail
        .map((d) => {
          const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "";
          return field ? `${field}: ${d.msg}` : d.msg;
        })
        .join("; ");
    } else {
      msg = detail || err.message || "An error occurred";
    }
    return Promise.reject(new Error(msg));
  }
);

// ─── Predictions ─────────────────────────────────────────────────────────────

export const analyzeLungImage = async (file, { patientId, scanType, notes } = {}) => {
  const formData = new FormData();
  formData.append("file", file);
  if (patientId) formData.append("patient_id", patientId);
  if (scanType)  formData.append("scan_type",  scanType);
  if (notes)     formData.append("notes",       notes);

  const res = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const listPredictions = async (skip = 0, limit = 20) => {
  const res = await api.get("/predictions", { params: { skip, limit } });
  return res.data;
};

export const getPrediction = async (predictionId) => {
  const res = await api.get(`/predictions/${predictionId}`);
  return res.data;
};

export const getModelMetrics = async () => {
  const res = await api.get("/model-metrics");
  return res.data;
};

// ─── Patients ─────────────────────────────────────────────────────────────────

export const createPatient = async (data) => {
  const res = await api.post("/patients", data);
  return res.data;
};

export const updatePatient = async (patientId, data) => {
  const res = await api.put(`/patients/${patientId}`, data);
  return res.data;
};

export const listPatients = async () => {
  const res = await api.get("/patients");
  return res.data;
};

// ─── Reports ──────────────────────────────────────────────────────────────────

export const generateReport = async (predictionId) => {
  const res = await api.post(`/reports/generate/${predictionId}`);
  return res.data;
};

export const listReports = async () => {
  const res = await api.get("/reports");
  return res.data;
};

// ─── Health ───────────────────────────────────────────────────────────────────

export const healthCheck = async () => {
  const res = await api.get("/health");
  return res.data;
};

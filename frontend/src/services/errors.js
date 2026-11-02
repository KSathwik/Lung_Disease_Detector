/**
 * Turns an axios error into an Error with a user-facing title, message and kind.
 * Kept free of axios imports so it can be unit tested directly.
 */

export class ApiError extends Error {
  constructor({ title, message, kind, status }) {
    super(message);
    this.name = "ApiError";
    this.title = title;
    this.kind = kind;
    this.status = status;
  }
}

const detailToText = (detail) => {
  if (detail == null) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors (422) come back as a list of objects
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "";
        return field ? `${field}: ${d.msg}` : d.msg;
      })
      .join("; ");
  }
  if (typeof detail === "object") {
    const parts = [detail.message, detail.reason].filter(Boolean);
    return parts.length ? parts.join(" ") : JSON.stringify(detail);
  }
  return String(detail);
};

export const normalizeApiError = (err) => {
  const status = err?.response?.status;
  const detail = err?.response?.data?.detail;
  const code = detail && typeof detail === "object" && !Array.isArray(detail) ? detail.error : undefined;
  const text = detailToText(detail);

  // The CRA dev-server proxy answers 500 with a plain-text "Proxy error ... ECONNREFUSED" when the backend is down
  const proxyDown = typeof err?.response?.data === "string" && /Proxy error|ECONNREFUSED/i.test(err.response.data);

  if (!err?.response || proxyDown) {
    const timedOut = err?.code === "ECONNABORTED";
    return new ApiError({
      kind: "network",
      title: timedOut ? "Request Timed Out" : "Server Unreachable",
      message: timedOut
        ? "The server took too long to respond. Please try again."
        : "Could not reach the LungAI server. Check that the backend is running and try again.",
    });
  }
  if (code === "INVALID_CXR") {
    return new ApiError({ kind: "invalid_image", status, title: "Not a Chest X-Ray", message: text });
  }
  if (code === "MODEL_NOT_LOADED" || status === 503) {
    return new ApiError({ kind: "model_unavailable", status, title: "Model Not Loaded", message: text || "The prediction model is not available." });
  }
  if (status === 401) {
    return new ApiError({ kind: "auth", status, title: "Not Authorized", message: text || "Missing or invalid API key." });
  }
  if (status === 413) {
    return new ApiError({ kind: "too_large", status, title: "File Too Large", message: text });
  }
  if (status >= 400 && status < 500) {
    return new ApiError({ kind: "bad_request", status, title: "Request Rejected", message: text || err.message });
  }
  return new ApiError({ kind: "server", status, title: "Server Error", message: text || err.message || "An unexpected error occurred." });
};

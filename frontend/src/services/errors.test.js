import { normalizeApiError } from "./errors";

const httpError = (status, detail) => ({ message: `Request failed with status code ${status}`, response: { status, data: { detail } } });

test("CXR gate rejection (object detail) becomes readable text, not [object Object]", () => {
  const e = normalizeApiError(httpError(422, {
    error: "INVALID_CXR",
    message: "The uploaded image does not appear to be a chest radiograph.",
    reason: "Uploaded image appears to be a natural color photograph.",
  }));
  expect(e.kind).toBe("invalid_image");
  expect(e.title).toBe("Not a Chest X-Ray");
  expect(e.message).toContain("does not appear to be a chest radiograph");
  expect(e.message).toContain("natural color photograph");
  expect(e.message).not.toContain("[object Object]");
});

test("network failure is reported as a server problem, not an image problem", () => {
  const e = normalizeApiError({ message: "Network Error" });
  expect(e.kind).toBe("network");
  expect(e.title).toBe("Server Unreachable");
  expect(e.title).not.toMatch(/radiograph|validation/i);
});

test("dev-proxy 'backend down' 500 is reported as unreachable", () => {
  const e = normalizeApiError({
    message: "Request failed with status code 500",
    response: { status: 500, data: "Proxy error: Could not proxy request /api/v1/predict from localhost:3000 to http://localhost:8000 (ECONNREFUSED)." },
  });
  expect(e.kind).toBe("network");
  expect(e.title).toBe("Server Unreachable");
});

test("timeout is distinguished from an unreachable server", () => {
  expect(normalizeApiError({ code: "ECONNABORTED", message: "timeout" }).title).toBe("Request Timed Out");
});

test("503 model-not-loaded is surfaced explicitly", () => {
  const e = normalizeApiError(httpError(503, { error: "MODEL_NOT_LOADED", message: "The prediction model is not loaded." }));
  expect(e.kind).toBe("model_unavailable");
  expect(e.title).toBe("Model Not Loaded");
});

test("FastAPI validation list is joined into one message", () => {
  const e = normalizeApiError(httpError(422, [{ loc: ["body", "age"], msg: "must be >= 0" }, { loc: ["body", "name"], msg: "required" }]));
  expect(e.message).toBe("age: must be >= 0; name: required");
});

test("string details and auth / size errors", () => {
  expect(normalizeApiError(httpError(413, "File too large. Maximum size is 10 MB.")).title).toBe("File Too Large");
  expect(normalizeApiError(httpError(401, "Missing or invalid API key.")).kind).toBe("auth");
  expect(normalizeApiError(httpError(500, "Model inference failed.")).message).toBe("Model inference failed.");
});

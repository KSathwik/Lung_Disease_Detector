# Frontend

React 18 (Create React App), in `frontend/`. In development, `npm start` proxies `/api` to `localhost:8000` through the `package.json` `proxy`. In Docker, the backend serves the build from the same origin.

| Page | Data source | Notes |
|---|---|---|
| Analyze (`pages/AnalyzePage.js`) | `POST /predict` | Shows the predicted class or "Inconclusive", the top score and six score bars, and Grad-CAM. Shows an error banner for 4xx/5xx/network failures, and a warning when `checkpoint_verified` is false |
| History (`pages/HistoryPage.js`) | `/predictions`, `/predictions/{id}` | Shows the stored six-class scores, model and checkpoint hash |
| Model Performance (`pages/MetricsPage.js`) | `/model-metrics` | Labelled as *recorded, not reproduced*. External sensitivity is always shown with specificity |
| Patients (`pages/PatientsPage.js`) | `/patients` | — |

## Rules the code enforces

- **Class names and order come from the API response** (`class_order`, `probabilities`) via `services/prediction.js::classScores`. The frontend has no class list of its own.
- **A malformed response is an error, not a result.** If a class score is missing, out of range or not a number, or the predicted class is not in the order, a `ResponseShapeError` is raised and shown as "Unexpected Server Response". A missing value is never displayed as 0%.
- **Scores are formatted for display only** (`formatScore`) and are labelled "uncalibrated softmax", not probability of disease.
- **Errors are never replaced with mock results.** `services/errors.js` maps 401, 413, 422 (CXR gate), 503 (model not loaded), network and proxy failures to specific messages.
- **No ambiguous model names.** The model name, version and ID shown are the ones the API returns.

## Tests

`src/services/{errors,prediction,metrics}.test.js` (Jest): 18 tests. React Testing Library is not installed, so there are no component-render tests. The page logic that matters for correctness lives in the tested service modules.

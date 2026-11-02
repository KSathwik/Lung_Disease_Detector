# Limitations

## Evidence

- **No checkpoint, no images.** The trained weights and the dataset images are not in the repository. As a result:
  - no metric can be reproduced here;
  - no prediction from the real model has been observed on this branch.
- **No per-image predictions** were saved by the historical runs, so ROC-AUC, PR-AUC and calibration cannot be verified.
- **Single runs.** Each training variant was run once (seed 42, 4 epochs). Differences of a few points between variants may be noise.

## Data

- **Patient leakage:** 154 of 1,570 internal-test images (9.8%) share a patient with training images. This inflates the internal results, especially for Pneumonia and Normal.
- **Unknown patient identity:** identity cannot be recovered for COVID-QU, Shenzhen and TBX11K images, so further leakage there cannot be ruled out.
- **Proxy labels:** half of the Pneumonia class is TBX11K "sick but not TB" (abnormal, cause unspecified).
- **Source confounding:** COVID-19 comes from a single source, and 88% of Pleural Effusion comes from VinDr. The model can separate several classes by source characteristics alone.
- **Population:** the Kermany images are paediatric. No demographic data is available.
- **Metadata:** the size and intensity columns of 10,090 manifest rows are placeholders.

## Model

- **Single-label assumption:** the six classes are treated as mutually exclusive. Real radiographs can show several findings at once.
- **External failure:** on Montgomery, 0 of 80 normal scans were called Normal, and every image was predicted Nodule/Mass or Pleural Effusion. Expect similar behaviour on any new scanner or hospital.
- **Resolution-dependent preprocessing:** the blur, CLAHE and low-pass steps run before resizing. The same anatomy scanned at a different resolution receives a different effective filter. This is kept for parity with the recorded checkpoint, and fixing it requires retraining.
- **Uncalibrated scores:** the softmax output is not a probability of disease. The 0.5 inconclusive threshold is not validated.
- **Grad-CAM** shows which features influenced the score. It does not show radiological reasoning.

## Application

- **CXR gate:** the trained gate model is absent, so a heuristic is used (`cxr_gate_mode: heuristic_fallback` in `/health`). It accepts some synthetic shapes and rejects some bright or inverted radiographs.
- **Static guidance:** the "Typical features" and "Precautions" texts are generic text for each class, not findings read from the image.
- **Authentication:** a single shared API key, or loopback-only access when no key is set. There are no user accounts and no audit log, and uploads are kept indefinitely.

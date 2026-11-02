# Data and Code Availability Statement

**Manuscript Title:** LungAI: Deep Learning-Based Multi-Class Disease Detection and Clinical Triage from Chest X-Rays with Frequency-Aware Preprocessing and Cross-Source Generalization Analysis  
**Target Journal:** *Biomedical Signal Processing and Control* (Elsevier)

### 1. Data Availability
The primary imaging collections analyzed in this study are drawn from publicly available, de-identified medical imaging repositories:
* **NIH ChestX-ray14:** National Institutes of Health Clinical Center (https://nihcc.app.box.com/v/ChestXray-NIHCC)
* **VinDr-CXR:** VinUni-Illinois Smart Health Center / PhysioNet (https://physionet.org/content/vindr-cxr/)
* **TBX11K Benchmark:** Large-scale Tuberculosis Chest X-ray Dataset (https://github.com/Daisy-ycguo/TBX11K)
* **JSRT Database:** Japanese Society of Radiological Technology (http://imgcom.jsrt.or.jp/minijsrtdb/)
* **Montgomery County CXR Dataset:** National Library of Medicine (https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-chest-x-ray-image-data-sets)
* **COVID-19 Radiography Database:** Kaggle / University of Dhaka (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

The curated manifest files, perceptual hash deduplication indices, and patient-level partition split splits generated in this research will be archived and released publicly upon publication on Zenodo under a CC-BY 4.0 license.

### 2. Code Availability
The complete source code for dataset auditing, perceptual hashing, biophysical image preprocessing, DenseNet-121 model training, Grad-CAM visualization, and statistical evaluation is implemented in Python/TensorFlow and hosted in the project repository:
* **Repository:** https://github.com/KSathwik/Lung_Disease_Detector
* All scripts required to replicate the experimental results, confusion matrices, and figures are included in the repository.

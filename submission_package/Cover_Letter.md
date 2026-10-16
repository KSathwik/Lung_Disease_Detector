# COVER LETTER

**To:**  
The Editor-in-Chief  
*Biomedical Signal Processing and Control (BSPC)*  
Elsevier B.V.

**Subject:** Submission of Original Research Article — *"LungAI: Deep Learning-Based Multi-Class Disease Detection and Clinical Triage from Chest X-Rays with Frequency-Aware Preprocessing and Cross-Source Generalization Analysis"*

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript entitled **"LungAI: Deep Learning-Based Multi-Class Disease Detection and Clinical Triage from Chest X-Rays with Frequency-Aware Preprocessing and Cross-Source Generalization Analysis"** for consideration for publication as a regular research paper in *Biomedical Signal Processing and Control*.

While deep convolutional neural networks demonstrate impressive diagnostic metrics on single-institution medical imaging collections, their real-world clinical translation is frequently hindered by patient-identity data leakage, non-radiographic artifact contamination, and acquisition-site shortcut learning. In this work, we present a comprehensive investigation addressing these fundamental translational failure modes:

1. **Curated Multi-Source Benchmark with Verified Planar Integrity:** We established an audited multi-source benchmark comprising 10,547 verified frontal chest radiographs across 10,270 unique patients compiled from eight independent institutional archives. Radial intensity profiling and aspect-ratio bounds were applied to purge non-planar slices and remove 83 near-duplicate image pairs.
2. **Leak-Free Patient-Level Partitioning:** Training (70%), validation (15%), and internal test (15%) subsets were allocated strictly by unique patient identifier, guaranteeing zero cross-split patient overlap.
3. **Five-Paradigm Domain Generalization Benchmark:** Under an identical DenseNet-121 backbone, we systematically benchmarked five domain generalization paradigms: Empirical Risk Minimization (ERM), Deep Correlation Alignment (Deep CORAL), Domain-Adversarial Neural Networks (DANN), Frequency-Aware Gaussian Spatial Filtering ($\sigma=1.0$), and a Hybrid feature-frequency model.
4. **Biophysical Shortcut Suppression Outperforms Latent Alignment:** We show that attenuating high-frequency scanner noise in image space before feature extraction significantly outperforms latent-space domain alignment objectives, lifting internal test accuracy from 76.18% to 82.93% (+6.75%), macro F1-score to 78.35%, and macro ROC-AUC to 0.9755 across six clinical categories (COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule/Mass, Tuberculosis).
5. **Quarantined External Evaluation & Digitizer Shift Analysis:** Evaluating the system zero-shot on the Montgomery County cohort ($N=138$) revealed a critical diagnostic dissociation: the architecture preserved 100% binary abnormality screening sensitivity (routing all 58 active TB cases to non-normal categories), but suffered exact fine-grained triage breakdown due to an approximate four-fold discrepancy in high-frequency Laplacian edge variance ($1,580$ vs. $373$) inherent to digitized analog film.

This manuscript has not been published previously, is not under consideration for publication elsewhere, and its submission has been approved by all contributing authors. 

We have selected the **Standard Subscription Publishing Model (No Open Access / $0 APC)**.

Thank you very much for your time, consideration, and management of the peer-review process.

Sincerely,

**Sathwik Katkam**  
Department of Computer Science and Engineering  
Jawaharlal Nehru Technological University Hyderabad (JNTUH)  
Kukatpally, Hyderabad, Telangana 500085, India  
Email: skatkam@jntuh.ac.in

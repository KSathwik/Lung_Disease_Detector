# Literature check for C1 and C2 (frequency-band source-versus-disease map; band-stop suppression)

Search date: 2026-11-03. Method: WebSearch (standard, then extended) and WebFetch. Only papers whose page (abstract page, HTML or PDF header) was actually opened are in the main table. Statements about what a paper did come from the opened page; where the opened page did not show something, it says "not shown". This is a search by one agent, not a systematic review: Scopus, Google Scholar, IEEE Xplore and the Elsevier journals were not queried directly (WebSearch is general web only), so a gap here is not proof of absence.

## 1. Closest papers (all opened)

| # | Full citation | What they did | Closeness |
|---|---|---|---|
| 1 | Halimi S, Themyr L, Abreu A. "Revealing Hidden Failure Modes in Chest X-ray Classification via Spectral Domain Analysis." arXiv:2610.29830v1, 21 Oct 2026 (marked under review, MIDL 2026). https://arxiv.org/html/2610.29830 | Chest X-rays go through a 2-D DFT; a radially averaged power spectrum (RAPS) gives a 1-D profile. Profiles are clustered without labels, and DenseNet-121 (trained on CheXpert, tested on CheXpert, MIMIC-CXR, PadChest, NIH) fails mostly in particular spectral clusters. Post-hoc diagnostic only. It does NOT probe source or disease information per band and does NOT suppress bands. | Partial to very close on the "radial spectrum of chest X-rays" side. Not C1 or C2, but the newest and most similar spectral chest X-ray paper. Must cite. |
| 2 | Lu Y, Juodelyte D, Victor JD, Cheplygina V. "Exploring connections of spectral analysis and transfer learning in medical imaging." arXiv:2407.11379 (v2, July 2024); journal reference stated on the arXiv page: Proc. SPIE 13406, Medical Imaging 2025: Image Processing. https://arxiv.org/abs/2407.11379 (full text read at https://arxiv.org/html/2407.11379v2) | Power spectral density of model gradients versus artificial frequency shortcuts (CT photon noise, MRI non-local-means denoising); ResNet-50 pre-trained on ImageNet or RadImageNet; datasets are LoDoPaB-CT (LIDC-IDRI subset) and knee MRI. No chest X-rays. Does not measure source, site or disease information per band. Modifies the pre-training source data (low-pass of ImageNet, spectral whitening of RadImageNet), not a band-stop on the target data. | Partial. Closest on "frequency shortcuts in medical imaging". The prior-art risk for C2 is moderate: they edit spectra of the pre-training data to change shortcut resistance. |
| 3 | Zhang R, Griner D, Garrett JW, Qi Z, Chen G-H. "Training certified detectives to track down the intrinsic shortcuts in COVID-19 chest x-ray data sets." Scientific Reports 13, 12690 (2023). DOI 10.1038/s41598-023-39855-3. Page opened: https://pmc.ncbi.nlm.nih.gov/articles/PMC10403557/ (preprint version: Research Square rs-3.rs-2818347/v1, https://pmc.ncbi.nlm.nih.gov/articles/PMC10168454/) | Treats global image contrast and sharpness as intrinsic acquisition shortcuts. Trains "shortcut detective" DenseNet classifiers on altered MIMIC-CXR; finds contrast/sharpness shortcuts in COVIDx (AUC 0.84 sharpness, 0.81 contrast), none detected in UW, BIMCV, MIDRC. No frequency-band decomposition. | Partial. This is the exact SciRep 2023 citation. It supports the idea that acquisition-dependent sharpness/contrast is a shortcut but works at image level, not per band. |
| 4 | Xu Q, Zhang R, Zhang Y, Wang Y, Tian Q. "A Fourier-based Framework for Domain Generalization." CVPR 2021 (oral). arXiv:2105.11120. https://arxiv.org/abs/2105.11120 | Amplitude Mix (interpolating amplitude spectra, keeping phase) plus a co-teacher consistency loss; natural-image benchmarks. | Partial for C2 (spectral manipulation for domain generalisation), but is amplitude/phase, not radial band-stop, and not chest X-ray. Background/must-cite as FACT. |
| 5 | He X, Hu J, Lin Q, Luo C, Xie W, Song S, Khan MH, Shen L. "Towards Combating Frequency Simplicity-biased Learning for Domain Generalization." NeurIPS 2024 (Advances in Neural Information Processing Systems 37). DOI 10.52202/079017-0979. https://proceedings.neurips.cc/paper_files/paper/2024/hash/377235d5cee7b104501407c0e5066c92-Abstract.html | Argues networks over-rely on certain frequency sets ("frequency shortcuts"); proposes two Fourier-domain augmentation modules that adaptively perturb those frequencies. Abstract page only opened; datasets not read. | Partial for C2 conceptually (suppress over-relied frequencies). General vision, not medical per the abstract page. |
| 6 | Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. "Confounding variables can degrade generalization performance of radiological deep learning models." arXiv:1807.00431 (July 2018); journal version PLoS Medicine 15(11), 2018 (the arXiv page I opened said 2019; check the final volume/year before citing). https://arxiv.org/abs/1807.00431 | CNNs identify the hospital system and department of a chest X-ray with very high accuracy and use it in predictions. No frequency analysis on the page. | Background (source identifiability). Must cite. |
| 7 | Compton R, Zhang L, Puli A, Ranganath R. "When More is Less: Incorporating Additional Datasets Can Hurt Performance By Introducing Spurious Correlations." MLHC 2023; arXiv:2308.04431. https://arxiv.org/abs/2308.04431 | Four open chest X-ray datasets, nine labels: adding a second hospital lowered worst-group accuracy in 43% of settings through hospital-specific artifacts and disease-hospital correlation. No frequency analysis. | Background. Supports the premise of C2 and the multi-source design. |
| 8 | Rafferty A, Rajan A. "Limitations of Public Chest Radiography Datasets for Artificial Intelligence: Label Quality, Domain Shift, Bias and Evaluation Challenges." arXiv:2509.15107v2 (2025). https://arxiv.org/pdf/2509.15107v2 (a journal listing with DOI 10.1098/rsta.2025.0129 appeared in search results; its page returned 403 and was not opened, so the journal details are unverified) | Review plus an empirical domain-shift section. Only the title, authors and outline were readable; I could not confirm the source-classifier accuracy or whether any frequency analysis exists. A search-result summary claimed near-perfect four-dataset source classification, which is unverified. | Background. Cite for dataset-source identifiability only after the PDF text is checked locally (a copy is saved at C:\Users\KATKAM SATHWIK\.claude\projects\D--LungAI\4aed576d-818d-4e3a-a9ca-004792d20464\tool-results\webfetch-1793725054273-k0brpe.pdf). |
| 9 | McIntosh D, Marques TP, Branzan Albu A. "Preservation of High Frequency Content for Deep Learning-Based Medical Image Classification." 18th Conference on Robots and Vision (CRV) 2021, pp. 41-48; arXiv:2205.03898. https://arxiv.org/abs/2205.03898 | DWT-based input that keeps high-frequency content lost in downsampling; improves CNNs on NIH Chest-8 and ImageNet. Disease performance only; no source analysis. | Background. Shows high bands carry disease information, which is a risk for C2's band-stop. |
| 10 | Chattopadhyay P, Sarangmath K, Vijaykumar V, Hoffman J. "PASTA: Proportional Amplitude Spectrum Training Augmentation for Syn-to-Real Domain Generalization." ICCV 2023; arXiv:2212.00979. https://arxiv.org/abs/2212.00979 | Perturbs amplitude spectra, more at high frequency; synthetic-to-real segmentation, detection, classification. No medical data. | Background for C2 (amplitude-spectrum augmentation). |
| 11 | Pan H, Wang B, Zhang Z, Zhu X, Jha D, Cetin AE, Spampinato C, Bagci U. "Domain Generalization with Fourier Transform and Soft Thresholding." arXiv:2309.09866v3 (revised 12 Dec 2023); venue not stated on the page (the repository name suggests ICASSP 2024: unverified). https://arxiv.org/html/2309.09866v3 | Fourier amplitude swapping plus soft thresholding of small amplitudes; retinal fundus optic-cup/disc segmentation, leave-one-domain-out over four sources. No chest X-ray. | Background for C2 (Fourier-domain DG in medical imaging, evaluated leave-one-domain-out). |

## 2. Leads that I could NOT open (do not cite until verified)

- DeGrave AJ, Janizek JD, Lee S-I. "AI for radiographic COVID-19 detection selects shortcuts over signal." Nature Machine Intelligence 3, 610-619 (2021). Appeared only in search-result summaries; no page of the paper itself was opened. Very likely correct but unverified here; it is the standard citation for COVID CXR data-source shortcuts.
- A 2026 source-free domain adaptation paper for multi-institutional chest X-rays using Laplacian-pyramid frequency-band statistics (search summary pointed to J Appl Clin Med Phys 2026, DOI 10.1002/acm2.70626, PMC13240487). Both pages were blocked (verification screen / 403). Authors and exact title unverified. It may be a "per-band chest X-ray" prior for C2-like ideas, so open it manually.
- "Homogenization of multi-institutional chest x-ray images in various data transformation schemes", J Med Imaging 10(6):061103 (2023), SPIE. Page returned empty; unverified; per the search snippet it uses Laplacian-pyramid decomposition to standardise images across institutions, which would be a close C2-style precedent.
- Ben Ahmed et al., IEEE Access 2022 (shortcut mitigation for multisite generalisation; doaj page 403). Unverified.
- Wang and Xia, BSPC 81 (2023) DELCOM; Teixeira et al., Sensors 2021; Bassi and Attux arXiv:2104.06176; npj Digital Medicine 2024 s41746-024-01118-4; Wang et al. ICCV 2023 (frequency shortcuts). These are already in docs/publication/archive/NOVELTY_PLAN.md; I did not re-open them in this search.

## 3. Verdict

**C1 (per-band source-versus-disease probing of frozen CNN features, with a separability index).** In what I could open, no paper does this for chest X-rays. The nearest are Halimi et al. 2026 (radial spectrum of chest X-rays, but unsupervised clustering to find model failure slices, no source or disease probing per band) and Lu et al. 2024 (frequency shortcuts, but injected artifacts on CT and knee MRI, PSD of gradients, no chest X-ray). So C1 appears novel in the narrow form stated, but two caveats apply: (a) the Halimi paper is two weeks old and shows this space is moving; (b) the unopened Laplacian-pyramid chest X-ray papers (section 2) may contain per-band analysis. Open those before writing the claim.

**C2 (band-stop of source-heavy bands before training, judged on held-out source).** Weaker. Fourier and frequency-selective augmentation for domain generalisation is established (Xu 2021, PASTA, He 2024, Pan 2023) and Lu 2024 edits spectra of pre-training data. What I did not find is a data-driven band selection, from per-band source-versus-disease probes, applied as a band-stop on chest X-rays and evaluated on a held-out source. The novelty is the selection rule and the chest X-ray evaluation, not frequency manipulation itself. Your own preprocessing already includes a Gaussian low-pass, so a reviewer will see C2 as an extension of filtering.

**Safe wording (draft):**
- C1: "To our knowledge, and within the literature we could access, we are not aware of prior work that quantifies, band by band, how much source and how much disease information frozen CNN features retain for chest radiographs on a patient-grouped multi-source benchmark."
- C2: "We evaluate whether band-selective suppression, with the bands chosen in advance from the per-band probe on validation data, reduces source information and changes held-out-source and external performance." Do not write "first", "novel filtering" or "state of the art". Report a null result honestly if suppression does not help.

## 4. What to cite and how to position against the closest three

Must cite: Zech 2018 (and the PLoS Med version), Compton 2023, Zhang 2023 (SciRep, exact citation above), Lu 2024, Halimi 2026, Xu 2021, McIntosh 2021; and after verification DeGrave 2021 and the Laplacian-pyramid chest X-ray papers.

Positioning against the closest three:
1. Halimi et al. (2026): they use the radial spectrum as a post-hoc descriptor to find failure clusters in single-source-trained models, with no labels of source or disease per band. You supervise per-band probes for source and disease on frozen features and use the result to decide an intervention. Say this plainly and cite them as the nearest spectral chest X-ray work.
2. Lu et al. (2024): they inject artificial frequency shortcuts and study gradient PSD in CT/MRI; you study naturally occurring source fingerprints in chest X-rays and measure information per band. Do not claim they missed frequency shortcuts; they did not look at real source information.
3. Zhang et al. (2023): they show contrast and sharpness (acquisition properties) act as shortcuts in COVID CXR at image level; C1 localises such acquisition information in frequency bands and C2 tests removing it. Treat their result as motivation and your 93.8% / 97.2% source probe as a complementary measurement.

Because high bands carry pathology (McIntosh 2021), state in the manuscript that band-stop may cost disease signal and that the separability index exists to quantify that trade-off.

## 5. Cheap ways to raise novelty

- Add a within-Normal-class source probe per band (you already have 97.2% overall), so the map is free of disease confounding.
- Add a phase-versus-amplitude split per band (keep amplitude, randomise phase, and the reverse). This tests whether the source fingerprint sits in amplitude (as FDA/AmpMix assume) or in phase. It costs only extra forward passes.
- Add an amplitude-mixing baseline (Xu 2021) beside the band-stop in C2 so you compare against the established method rather than only against your own pipeline.
- Use a matched-energy control (stop a random band of equal energy) to show source-selective bands matter, not just removing any content.


## Lead resolution (2026-11-03)

Method: Crossref API (curl), Europe PMC REST search and full-text XML, arXiv abstract pages, WebFetch. Publisher pages (Nature, PMC, Wiley) were not readable (verification screens, 303 to login); Semantic Scholar and Crossref via WebFetch returned 429. Only facts from pages that returned content are recorded.

### Lead 1: J Appl Clin Med Phys 2026, DOI 10.1002/acm2.70626 (RESOLVED, correction to earlier lead)
- Confirmed: Kim H, Lee S, Cho S. "Source-free domain adaptation for multi-institutional chest X-ray images." J Appl Clin Med Phys 2026; 27(5): e70626 (Europe PMC record; PMC13240487).
- What it does: tuberculosis detection; source-free adaptation with SHOT and SFDA-DE, ranking-based pseudo-labelling and selective fine-tuning; six unseen target domains; improved F1 and AUROC. The abstract and keywords do not mention Laplacian pyramids or frequency, so the earlier "Laplacian-pyramid bands" description was wrong for the adaptation method itself. Europe PMC full text (first 100000 of about 268000 characters read) shows Laplacian-pyramid "multi-frequency standardization" only as a preprocessing baseline used to train the source models (taken from the authors' earlier work, Lead 2). No site classifier.
- Impact: C1 none. C2 none to weak (no probe-driven band selection).

### Lead 2: J Med Imaging 10(6):061103 (2023) (RESOLVED)
- Confirmed: Kim H, Lee S, Shim WJ, Choi M-S, Cho S. "Homogenization of multi-institutional chest x-ray images in various data transformation schemes." J Med Imaging 10(6):061103, 2023. DOI 10.1117/1.JMI.10.6.061103 (Crossref via WebFetch; Europe PMC record; PMC10132786, full text read via Europe PMC to character 100000).
- What it does: compares normalisation, standardisation (with and without lung mask) and multi-frequency-based (MFB) standardisation (with and without lung mask) for TB detection, one reference site plus six external sites. MFB uses a 5-level Laplacian pyramid; each level is shifted and scaled to reference mean and standard deviation, iterated 50 times. Best external F1, recall and attention score with MFB plus lung mask. Homogenisation judged by Mahalanobis distances on 93 radiomic features and on 512-d deep features. No site-classification model and no per-band source-versus-disease probe.
- Impact: C1 none (a measurement, not a probe; applies the same per-band alignment to all bands). C2 overlaps/weakens: per-band Laplacian-pyramid harmonisation of chest X-rays is published. C2 must be framed as probe-selected band suppression (not uniform band matching) and must cite this paper. C3 none.

### Lead 3: DeGrave, Janizek, Lee 2021 (RESOLVED bibliographically; paper body not read)
- Confirmed by Crossref API (curl): "AI for radiographic COVID-19 detection selects shortcuts over signal", authors DeGrave AJ, Janizek JD, Lee S-I; Nature Machine Intelligence 3, 610-619; published 2021-05-31; DOI 10.1038/s42256-021-00338-7. Preprint confirmed in Europe PMC: medRxiv 2020, DOI 10.1101/2020.09.13.20193565 (PPR213715).
- What it does (abstract level, from Europe PMC preprint abstract and search-result text; the Nature page redirected to login): explainable AI shows COVID-19 chest X-ray models rely on confounders instead of pathology and would likely fail at new hospitals. Not frequency-based, no source probe.
- Impact: C1/C2/C3 none. Cite as background for source shortcuts.

### Lead 4: INLP/ComBat site-removal lead (RESOLVED)
- Identified: Rahbar S. arXiv:2608.10295. v1 title "Frozen Brain-MRI Foundation Models Are Site Fingerprints" (submitted 2026-08-10); v2 (revised 2026-10-03) retitled "Site Is Decodable Before Pretraining: Negative Controls for Probing Frozen Brain-MRI Foundation Models", and the v2 abstract no longer mentions INLP, ComBat or segmentation. The v1 abstract page (opened via curl) states: site linearly decodable at about 0.9 from frozen 3-D encoders (ABIDE-I/II), also about 0.9 for random-init encoders and about 0.95 from the raw downsampled image; the site subspace is removable by iterative null-space projection or ComBat (site decodability 0.94 to 0.07/0.00), but for dense segmentation removal is destructive (site and anatomy entangled; matched-rank random projection is Dice-neutral).
- Domain: brain MRI, single-author preprint, no chest X-ray, no frequency bands, no disease-head retrain evaluated on classification.
- Impact: C1 none. C3 weakens modestly: INLP/ComBat site erasure with an entanglement test on frozen features exists in brain MRI. The chest X-ray, six-source, disease-head-retrain claim still stands but must cite this as the nearest medical-imaging precedent, citing v1 specifically (v2 dropped the result). Preprint only.

### Lead 5: extra targeted searches for the C1 gap (4 searches run, plus pages opened as noted)
- Dack E, Dai C. "Understanding Dataset Bias in Medical Imaging: A Case Study on Chest X-rays." arXiv 2507.07722 (v4 opened, WebFetch summary): "Name That Dataset" on NIH, CheXpert, MIMIC-CXR, PadChest; ResNet50 F1 about 99; pixel shuffling lowers F1 to about 58-63; smoothing/sharpening only as augmentation; no Fourier, high-pass or low-pass analysis. Confirms source identifiability (background) but not band resolution. Cite.
- Halimi S, Themyr L, Abreu A. arXiv 2610.29830, MIDL 2026 (abstract page opened): per-image spectral-energy profiles, clustering to find failure "blind spots". No source or site identification, no probe. Already the listed nearest neighbour; the gap stands.
- A natural-image study (arXiv 2412.01876, "Understanding Bias in Large-Scale Visual Datasets") reportedly runs dataset classification on high-pass and low-pass filtered images. Seen only as a search-result summary, page not opened: unverified. If confirmed it is a non-medical precedent for filtered dataset classification, so C1 wording must not present frequency-filtered source classification as new in general, only the per-band chest X-ray source-versus-disease comparison in frozen CNN features.
- Queries combining frequency band or spectral with source, site or hospital identification and chest X-ray returned nothing else. The search is not exhaustive (US-only search engine, no Google Scholar access, Europe PMC used only for title lookups).

### BibTeX entries to cite (not added to references.bib)
```bibtex
@article{kim2023homogenization,
  author  = {Kim, Hyeongseok and Lee, Seoyoung and Shim, Woo Jung and Choi, Min-Seong and Cho, Seungryong},
  title   = {Homogenization of multi-institutional chest x-ray images in various data transformation schemes},
  journal = {Journal of Medical Imaging},
  year    = {2023},
  volume  = {10},
  number  = {6},
  pages   = {061103},
  doi     = {10.1117/1.JMI.10.6.061103}
}
@article{kim2026sfda,
  author  = {Kim, Hyoyi and Lee, Seoyoung and Cho, Seungryong},
  title   = {Source-free domain adaptation for multi-institutional chest X-ray images},
  journal = {Journal of Applied Clinical Medical Physics},
  year    = {2026},
  volume  = {27},
  number  = {5},
  pages   = {e70626},
  doi     = {10.1002/acm2.70626}
}
@article{degrave2021shortcuts,
  author  = {DeGrave, Alex J. and Janizek, Joseph D. and Lee, Su-In},
  title   = {{AI} for radiographic {COVID-19} detection selects shortcuts over signal},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  volume  = {3},
  pages   = {610--619},
  doi     = {10.1038/s42256-021-00338-7}
}
@misc{rahbar2026site,
  author        = {Rahbar, Saman},
  title         = {Frozen Brain-{MRI} Foundation Models Are Site Fingerprints},
  year          = {2026},
  eprint        = {2608.10295v1},
  archiveprefix = {arXiv}
}
@misc{dack2025bias,
  author        = {Dack, Ethan and Dai, Chengliang},
  title         = {Understanding Dataset Bias in Medical Imaging: A Case Study on Chest X-rays},
  year          = {2025},
  eprint        = {2507.07722},
  archiveprefix = {arXiv}
}
```
(Dack and Dai: year inferred from the arXiv ID 2507; confirm before submission. The first name "Hyoyi" is as shown in the Europe PMC record.)

### Final novelty verdict (as of 2026-11-03)
- C1 stands, with the scoped wording already in docs/publication/archive/NOVELTY_ROADMAP.md ("to our knowledge, within the searches described"). No opened paper does a per-band source-versus-disease probe on chest X-ray CNN features. Both Laplacian-pyramid chest X-ray leads are harmonisation/adaptation papers, not probes. Caveats: do not claim frequency-filtered dataset classification as new in general (an unverified natural-image precedent exists), and cite Dack and Dai for source identifiability.
- C2 is weakened: Kim 2023 (MFB) is a published per-band chest X-ray harmonisation. C2 is novel only as probe-driven band suppression; keep it optional and cite it.
- C3 is modestly weakened: INLP/ComBat removal with a segmentation entanglement test exists in brain MRI (Rahbar, v1). The chest X-ray, six-source, disease-head-retrain version stands as a diagnostic, not a method.
- Still unverified: DeGrave full text (abstract level only), arXiv 2412.01876, Ben Ahmed IEEE Access 2022, Rafferty and Rajan journal details.

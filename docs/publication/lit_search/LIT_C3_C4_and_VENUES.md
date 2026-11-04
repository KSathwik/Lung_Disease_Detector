# Literature check for C3 / C4 and Elsevier venue comparison

Checked on 2026-11-03. Only pages opened in this session are listed (arXiv/ACL/NeurIPS abstract pages and PLOS/PMC/Microsoft Research pages via WebFetch; ScienceDirect pages via the browser pane, because scripted fetches get HTTP 403). The searches were not exhaustive: 2 "standard" and 3 "extended" web searches for C3, and 2 for C4. Absence of a hit below means "not found in this search", not "does not exist". Run one Scopus/Google Scholar query before writing any novelty sentence.

## 1. Closest papers

### C3 (source/site removal from CNN features by INLP or similar linear erasure, then a linear disease head)

| # | Paper | Authors | Venue, year | URL / DOI | What it does | Closeness to C3 |
|---|---|---|---|---|---|---|
| 1 | Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (INLP) | Ravfogel, Elazar, Gonen, Twiton, Goldberg | ACL 2020 | https://aclanthology.org/2020.acl-main.647 (cited in NOVELTY_PLAN; not re-opened here) | Method source. Text/NLP only | Method source (not medical) |
| 2 | Linear Adversarial Concept Erasure (RLACE) | Ravfogel, Twiton, Goldberg, Cotterell | ICML 2022 (arXiv v1 Jan 2022) | https://arxiv.org/abs/2201.12091 (opened) | Linear subspace removal as a maximin game; gender removal in text | Method source (not medical) |
| 3 | LEACE: Perfect linear concept erasure in closed form | Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman | NeurIPS 2023 | https://arxiv.org/abs/2306.03819 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract.html (arXiv ID and authors from search-result pages, not opened by me) | Closed-form erasure of a concept for all linear classifiers | Method source (not medical) |
| 4 | Feature-level Site Leakage Reduction for Cross-Hospital Chest X-ray Transfer via Self-Supervised Learning | Bouaziz, Chebouba | arXiv 2604.00263 (31 Mar 2026); accepted at 7th Int. Conf. on Computing Systems and Applications, Algiers 2026 | https://arxiv.org/abs/2604.00263 (abstract opened) | ResNet-18 SSL on NIH+CheXpert, frozen encoder, linear pneumonia head, linear probe measuring site leakage, adversarial site confusion on features; transfer to RSNA. Abstract does not mention nullspace/erasure | HIGH (same idea: frozen features + linear site probe + linear disease head, CXR). Differs: adversarial not INLP/LEACE, 3 sites, binary pneumonia, no leakage-controlled six-class benchmark. Preprint/small conference |
| 5 | Deep Learning Models for COVID-19 Chest X-ray Classification: Preventing Shortcut Learning Using Feature Disentanglement | Robinson, Trivedi, Blazes, Ortiz, Desbiens, Gupta, Dodhia, Bhatraju, Liles, Lee, Kalpathy-Cramer, Lavista Ferres | PLOS ONE, Oct 2022 | doi:10.1371/journal.pone.0274098 ; https://www.microsoft.com/en-us/research/?p=776422 (opened) | Training-time penalty on features that identify the source dataset (pooled COVID CXR); beat lung masking and histogram equalisation (AUC gain up to 0.13) | HIGH conceptually (source removal from CXR features, multi-source). Differs: training-time disentanglement, not post-hoc linear erasure + linear head retrain |
| 6 | Investigation of ComBat Harmonization on Radiomic and Deep Features from Multi-Center Abdominal MRI Data | authors not captured | PMC11950493 (journal/year not captured) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11950493/ (seen only as a search result, not opened) | ComBat on Swin-Transformer deep features, MRI | MEDIUM-LOW (ComBat on deep features, but MRI; unverified beyond search snippet) |
| 7 | Robust Multicentre Detection and Classification of Colorectal Liver Metastases on CT: Application of Foundation Models | authors not captured | arXiv 2601.07585 | https://arxiv.org/pdf/2601.07585 (search result only, not opened) | ComBat on foundation-model features lowered AUC (0.72 vs 0.81) per search snippet | LOW (CT; snippet only) |
| 8 | From Measurement to Mitigation: Quantifying and Reducing Identity Leakage in Image Representation Encoders with Linear Subspace Removal | George, Yeh, Lee, Zhang | arXiv 2604.05296 (7 Apr 2026) | https://arxiv.org/abs/2604.05296 (opened) | One-shot linear projector removes identity subspace from CLIP/DINO embeddings; faces, not medical | LOW (shows linear removal on image embeddings; not medical) |

Unverified lead (do not cite): one search snippet described a medical-imaging study where a site subspace was removed by INLP or ComBat (site decodability about 0.94 to near 0) and dense segmentation was harmed. I could not identify or open the source. Search again by title before claiming C3 is novel.

### C4 (leave-one-source/domain-out matrices, multi-source CXR)

| # | Paper | Authors | Venue, year | URL / DOI | What it does | Closeness to C4 |
|---|---|---|---|---|---|---|
| 1 | A Strong Baseline for Domain Adaptation and Generalization in Medical Imaging | Yao, Prosky, Covington, Lyman | arXiv 1904.01638, 2019 | https://arxiv.org/abs/1904.01638 ; https://ar5iv.labs.arxiv.org/html/1904.01638 (opened) | DenseNet-121, normal vs abnormal; 5 training sources, leave one source out each run, evaluated on all 10 datasets (AUC, shown as figure) | HIGH (leave-one-domain-out on CXR). Differs: binary task, AUC, no source probe, no leakage control, no six-class setup |
| 2 | When More is Less: Incorporating Additional Datasets Can Hurt Performance By Introducing Spurious Correlations | Compton, Zhang, Puli, Ranganath | MLHC 2023 (arXiv Aug 2023, 2308.04431) | https://arxiv.org/abs/2308.04431 (opened) | Four open CXR datasets, nine labels; two-hospital training worse in worst-group accuracy in 43% of settings | MEDIUM (multi-source CXR shortcut; train/test source combinations, not a full LOSO matrix as far as the abstract says) |
| 3 | Feature-level Site Leakage Reduction ... (see C3 #4) | Bouaziz, Chebouba | arXiv 2604.00263, 2026 | https://arxiv.org/abs/2604.00263 | NIH to RSNA transfer; abstract does not state leave-one-site-out (a search-result snippet said it does; I could not confirm from the abstract) | MEDIUM |
| 4 | How intra-source imbalanced datasets impact the performance of deep learning for COVID-19 diagnosis using chest X-ray images | Zhang, Zhang, Ichiji, Bukovsky, Homma | Scientific Reports 13:19049, 2023 | doi:10.1038/s41598-023-45368-w ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10624834/ (opened) | Train on one dataset, test on the other (2 datasets); source-specific marker learned | MEDIUM-LOW (cross-dataset, not LOSO) |
| 5 | The pitfalls of using open data to develop deep learning solutions for COVID-19 detection in chest X-rays | Harkness, Hall, Frangi, Ravikumar, Zucker | arXiv 2109.08020, MedInfo 2021 | https://arxiv.org/abs/2109.08020 (opened) | Train on open set, test on external and hospital data; COVIDx inflated | LOW-MEDIUM (external-set evaluation, no LOSO) |
| 6 | Trivedi et al. (first author per Crossref, corrected 2026-11-04), PLOS ONE 2022 (see C3 #5) | | | | Page did not mention leave-one-source-out | LOW for C4 |

Search-result mentions (not opened, do not cite): Zech et al. 2018 and the npj Digital Medicine 2024 paper are in NOVELTY_PLAN.md as already-read items.

## 2. Verdict and safe wording

**C3.** Not novel as a technique (INLP/RLACE/LEACE are published; ComBat on deep features exists in MRI/CT). I found no paper that applies INLP/LEACE/RLACE to chest X-ray CNN features and then retrains a linear disease head, but the closest work (Bouaziz and Chebouba 2026, Trivedi et al. 2022) addresses the same goal with adversarial or disentanglement methods, and one unidentified lead remains. Claim only an application and evaluation contribution.
Safe wording: "To our knowledge, and within the searches described in Section X, we did not find prior work applying post-hoc linear concept erasure (INLP) to remove source information from CNN features in multi-source chest radiograph classification. We evaluate it as a diagnostic of source-disease entanglement, not as a new method." Avoid "first", "novel method", "state of the art". Cite Bouaziz and Chebouba 2026 and Trivedi et al. 2022 as related work; cite Ravfogel 2020/2022 and Belrose 2023 as the methods used.

**C4.** Not novel as a protocol: Yao et al. 2019 already ran leave-one-domain-out on CXR (DenseNet-121, AUC). The defensible gap is narrower: a full source-by-source matrix on a patient-grouped, leakage-controlled six-class benchmark, paired with a source-identifiability probe and (if C3 runs) before/after erasure.
Safe wording: "Following the leave-one-domain-out design of Yao et al. (2019), we report a held-out-source matrix on a leakage-controlled six-class benchmark and relate it to source identifiability (93.8% balanced accuracy in our probe)." Do not claim the protocol as a contribution.

Combined C3+C4 as "source information is linearly erasable but held-out-source performance still drops" is a reasonable, honest framing if the results support it; report negative results as found.

## 3. Venue comparison (Elsevier, all pages opened 2026-11-03)

Insights pages: https://www.sciencedirect.com/journal/<slug>/about/insights. Times are the publisher-stated "submission to first decision / decision after review / acceptance". APC is "excluding taxes".

| Journal | Scope fit to source shortcut / leakage-controlled CXR evaluation | Scopus (page lists) | Access and APC | First decision / after review / acceptance | Article type and length |
|---|---|---|---|---|---|
| Biomedical Signal Processing and Control (BSPC), CiteScore 13.0, IF 5.7 | Good: image analysis, robustness, benchmark papers (scope text from earlier session, see JOURNALS.md); frequency analysis suits "signal processing" | Yes (plus SCIE, Ei Compendex) | Hybrid; OA USD 3,470; subscription: no fee | 10 / 66 / 203 days | Full paper about 5,000 words, short communication about 2,000 (from JOURNALS.md; guide not re-opened today) |
| Computers in Biology and Medicine (CBM), CiteScore 13.1 | Good scope, but per JOURNALS.md Web of Science coverage was discontinued Nov 2025 and the guide rejects unclear splits and weak SOTA comparison | Yes (page lists Scopus, Medline) | Hybrid; OA USD 3,080; subscription free | 5 / 56 / 167 days; acceptance rate 12% (page) | Not re-read; not recommended |
| Informatics in Medicine Unlocked (IMU), CiteScore 10.5 | Good: medical informatics incl. imaging; accepts original research, tutorials, perspectives | Yes (plus DOAJ) | Gold OA; USD 2,710 with geographical pricing (India amount not shown) | 2 / 36 / 114 days | No word limit found; abstract up to 250 words, 1 to 7 keywords |
| Intelligent Systems with Applications (ISWA), CiteScore 13.1, IF 5.5 | Moderate: AI journal with an "Intelligent Systems for Medical Treatment and Health" topic; shortcut analysis fits as AI reliability | Yes (plus DOAJ) | Gold OA; USD 2,830 with geographical pricing | 2 / 50 / 133 days | No word limit found; abstract up to 250 words |
| Array, CiteScore 10.1, IF 5.3 | Weak-moderate: general computer science incl. computer vision and pattern recognition; medical is not a focus | Yes (plus DOAJ, ESCI) | Gold OA; USD 2,770 with geographical pricing | 9 / 64 / 130 days | Not read |
| Healthcare Analytics, CiteScore 20.0 | Weak: decision-making and operations in healthcare organisations; image classification is off-topic | Yes (plus DOAJ) | Gold OA; discounted USD 1,260 for submissions by 31 Dec 2026, full USD 2,520 | 1 / 63 / 155 days | Not read |
| Computer Methods and Programs in Biomedicine (CMPB), CiteScore 11.9, IF 6.4 | Good topical fit (computing methods in medical practice); original research is expected to carry a method or software contribution | Yes (plus Medline, SCIE) | Hybrid; OA USD 3,320; subscription free | first decision not stated / 98 / 163 days | Original research normally up to 3,500 words excluding abstract (350 words, structured), references up to 50 (guide) |

Other Elsevier title already assessed in JOURNALS.md (2026-11-03): Computerized Medical Imaging and Graphics (CiteScore 8.8, IF 5.5, Scopus, hybrid OA USD 2,920, 3 / 70 / 145 days, expects method novelty). Not re-opened today.

Scopus note: "Scopus" above means the journal's own Insights page lists it as an abstracting and indexing service; I did not query the Scopus Sources list.

## 4. Ranked recommendation (fast, realistic acceptance, little time)

1. **Informatics in Medicine Unlocked** - fastest stated process (2 / 36 / 114 days), imaging explicitly in scope, original research and tutorials accepted, Scopus listed, no method-novelty emphasis. Cost: gold OA only, USD 2,710 (geographical pricing may lower it; ask the institution about waivers). Best for a leakage-controlled evaluation plus shortcut analysis.
2. **Biomedical Signal Processing and Control** - best subscription route (no APC) and better scope for frequency analysis, but acceptance takes about 203 days, and reviewers will ask for seeds, ablations and more cohorts (JOURNALS.md). Choose it if the APC is the binding constraint and the extra experiments are done.
3. **Intelligent Systems with Applications** - quick first decision, Scopus listed, OA USD 2,830; weaker medical focus.
4. **Computer Methods and Programs in Biomedicine** - strong indexing, free subscription route, but slower (98 days after review) and expects a methods/software contribution; a 3,500-word limit forces a tight paper.
5. **Array** - fast enough, indexed, weakest scope fit among OA options (ESCI, not SCIE).
6. **Healthcare Analytics** - scope mismatch; only if the framing shifts to evaluation practice for health AI.
- Skip **CBM** (Web of Science delisting per JOURNALS.md).

Caveats: publisher-stated timelines are averages, not guarantees; APCs change; first decision of 1 to 2 days usually means desk screening, not full review. Check the Scopus Sources list by ISSN before submission (IMU 2352-9148, BSPC 1746-8094, ISWA 2667-3053, CMPB 0169-2607, Array 2590-0056, Healthcare Analytics 2772-4425).


## Lead resolution (2026-11-03)

The unidentified INLP/ComBat lead above is RESOLVED: it is Rahbar S, arXiv:2608.10295 v1, "Frozen Brain-MRI Foundation Models Are Site Fingerprints" (v1 abstract page opened; site decodability 0.94 to 0.07 with INLP and 0.00 with ComBat; dense segmentation harmed by site-subspace removal, matched-rank random projection Dice-neutral). Brain MRI (ABIDE-I/II), single-author preprint. v2 (2026-10-03) is retitled "Site Is Decodable Before Pretraining: Negative Controls for Probing Frozen Brain-MRI Foundation Models" and its abstract no longer reports the INLP/ComBat/segmentation result, so cite v1 explicitly. Impact on C3: weakens modestly (nearest medical-imaging precedent for site erasure with an entanglement test); the chest X-ray six-source disease-head-retrain claim stands. Full details, the other lead resolutions and the BibTeX entry are in the "Lead resolution (2026-11-03)" section at the end of LIT_C1_C2_frequency.md. The "do not cite" status can be lifted for v1 only.

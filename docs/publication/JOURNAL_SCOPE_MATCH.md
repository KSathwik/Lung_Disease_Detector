# Journal match per scope option (S1, S2)

> Decision (2026-11-04, student): target journal is Informatics in Medicine Unlocked (IMU); Biomedical Signal Processing and Control (BSPC) is the fallback. Reason: diagnostic study after the C2 NO-GO. EMBC 2027 is unchanged.

Prepared 2026-11-03. All pages below were opened on 2026-11-03 (access date) in the browser pane (scripted fetches of ScienceDirect return HTTP 403). Only what those pages show is recorded; "not stated" means the page text I read did not contain it. No impact-factor claims are made except where a page shows them (CiteScore/Impact Factor figures are copied from each Insights page header). The earlier table in `lit_search/LIT_C3_C4_and_VENUES.md` was re-checked, not copied blindly: every APC, timeline and indexing figure in it that I re-opened matched today's pages (BSPC, IMU, CBM, CMPB, ISWA, Array, CMIG).

Scopus: "Scopus" below means the journal's own ScienceDirect Insights page lists it under "Abstracting and indexing". I did not query the Scopus Sources list; check each ISSN there before submission.

URL pattern: `https://www.sciencedirect.com/journal/<slug>/about/insights` (Insights) and `.../publish/guide-for-authors` (Guide for Authors, which also carries the Aims and scope text).

## 0. What is being placed

- Manuscript size: `docs/manuscripts/journal_bspc/sections/*.tex` total 13,171 whitespace-separated tokens (LaTeX commands and macros included, so true prose is somewhat lower; abstract 246, intro 842, related work 772, materials 1,997, methods 1,835, experimental design 1,366, results 3,386, discussion 2,409, conclusion 318). Adding C1/C3/C4 subsections will push it higher before any cutting (`SCOPE_MERGE_REVIEW.md`: compress or move the per-class table and run comparison to supplementary).
- S1: measurement and evaluation study (C1 band map, C3 linear erasure, C4 leave-one-source-out) on a leakage-controlled benchmark. No new architecture.
- S2: S1 plus C2 probe-driven band-stop with one exploratory retraining seed (weak evidence).

## 1. Journal records

### 1.1 Biomedical Signal Processing and Control (BSPC)
- Pages: https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors ; https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/about/insights
- Scope quotes: "Medical image processing and analysis"; "Signal quality assessment, artifact detection, denoising, and robustness analysis"; "Time-frequency, adaptive, nonlinear, and data-driven signal processing methods"; "benchmark or dataset papers". The same page says the journal "welcomes contributions that develop novel methods", and that its identity includes how data "drive decisions, feedback mechanisms, and closed-loop control".
- Article types and limits: Full papers "describing original work", defined as "the technical development and the clinical / experimental study to prove the concept / value of a novel technique", "normally about 5,000 words"; Short communications "typically 2,000 words" for "preliminary" observations; Technical Notes 5,000 words, maximum 5 illustrations, "would not normally look at data from a large cohort"; Reviews 10,000 words. Whether references, tables and captions count toward the 5,000 is not stated. Abstract under 250 words, 1-7 keywords, highlights 3-5 bullets of at most 85 characters (separate file), graphical abstract encouraged (min 1328 px wide), single-anonymised review, generative-AI declaration section if AI tools used, preprints allowed.
- A ~13k-word paper does not fit: it is about 2.6x the "normally about 5,000" figure. Wording is "normally", not a hard cap, and the counting rule is not stated, but expect an editor query or desk rejection. An S1 paper has no "novel technique", which is the stated definition of a full paper.
- Scopus: listed (also SCIE, SJR, SNIP, Ei Compendex). CiteScore 13.0, Impact Factor 5.7 (page header).
- Access: gold OA APC USD 3,470 (excluding taxes; "Actual APCs may differ"); subscription route "No publication fee charged to authors".
- Insights timeline: 10 days to first decision, 66 days to decision after review, 203 days to acceptance, 12 days to online. Acceptance rate: not stated.
- Similar published paper (verified on its ScienceDirect page): "Analysis of the Clever Hans effect in COVID-19 detection using Chest X-Ray images and Bayesian Deep Learning", BSPC, April 2024, DOI https://doi.org/10.1016/j.bspc.2023.105831 (page: https://www.sciencedirect.com/science/article/pii/S1746809423012648). Its highlights describe explainability used to identify spurious correlations and artefact effects. Also seen in the BSPC search list (title and abstract opening only, DOI not captured): "CausalCompNet: Causal intervention meets vision-language priors for robust CXR diagnosis" (PII S1746809426011080), which is a new-method paper on artefact-induced spurious correlations, i.e. BSPC does take shortcut-themed CXR work but both examples add a method.

### 1.2 Informatics in Medicine Unlocked (IMU)
- Pages: https://www.sciencedirect.com/journal/informatics-in-medicine-unlocked/publish/guide-for-authors ; https://www.sciencedirect.com/journal/informatics-in-medicine-unlocked/about/insights
- Scope quotes: "covering a broad spectrum of topics within medical informatics"; "papers focusing on imaging, pathology, teledermatology, public health". Accepted types: "original research papers, reviews, perspectives, letters to the editor and editorials, tutorials, and papers on innovative and topical new technology". "Only manuscripts of sufficient quality with scientific relevance will be considered" (no method-novelty requirement stated).
- Limits: no word or page limit found on the page; abstract up to 250 words; 1-7 keywords; highlights 3-5 bullets of at most 85 characters. Graphical abstract: heading present, requirement wording not captured. Ethics statement required at submission: if none needed, authors "must state why". A 13k-word paper is not excluded by anything stated, but trimming is still sensible.
- Scopus: listed (also DOAJ, SJR, SNIP). CiteScore 10.5 (no Impact Factor shown).
- Access: gold open access only. APC USD 2,710 (excluding taxes), with Geographical Pricing (price set by author-country income level; the amount for the student's country is not shown). No subscription route.
- Timeline: 2 days / 36 days / 114 days / 4 days to online. The guide says average evaluation time "less than 4 weeks" and 2-4 months to publication (editor's statement, not a guarantee).
- Similar published paper: "Factorial evaluation of DICOM harmonization and clinically constrained augmentation for multi-vendor TB triage", IMU, September 2026, DOI https://doi.org/10.1016/j.imu.2026.101808 (page: https://www.sciencedirect.com/science/article/pii/S235291482600078X). A factorial evaluation of preprocessing effects on multi-vendor chest radiograph triage with no new architecture; same journal accepts evaluation-style CXR work. It appeared for the query "shortcut learning X-ray" but its highlights mention harmonization and stability, not shortcut learning by name.

### 1.3 Computers in Biology and Medicine (CBM)
- Pages: https://www.sciencedirect.com/journal/computers-in-biology-and-medicine/publish/guide-for-authors ; .../about/insights
- Scope: application of the computer to bioscience and medicine; the page lists "Medical Informatics" and "Biomedical Engineering". It states it does not consider submissions "with unclear dataset division into training/validation/test sets" or "with minor architecture/model modifications and only slight increase in performance".
- Journal announcement on the same page: placed "On Hold" in April 2024; Web of Science Core Collection coverage "discontinued on November 17, 2025"; content after Volume 172 "not currently indexed" (there); integrity investigations running.
- Scopus: listed (also Medline, SJR, SNIP). CiteScore 13.1. APC USD 3,080; subscription no fee. Timeline 5 / 56 / 167 / 11 days. Acceptance rate 12% (page). Word limit: not stated in what I read.
- Similar published paper: a review only, "A systematic review of generalization research in medical image classification", DOI https://doi.org/10.1016/j.compbiomed.2024.109256 (page: https://www.sciencedirect.com/science/article/pii/S0010482524013416).
- Not recommended: delisting and integrity announcement.

### 1.4 Computer Methods and Programs in Biomedicine (CMPB)
- Pages: https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine/publish/guide-for-authors ; .../about/insights
- Scope quotes: "to report new computer methodologies applied in biomedical areas"; "computing methodology and software systems". Emphasis on methods and software.
- Limits: "Should not normally exceed 3500 words excluding abstract of 350 words"; "References up to 50"; structured abstract (Background and Objective, Methods, ...). 13k words would need to lose about three quarters. Highlights 3-5 bullets, 85 characters; keywords 3-6.
- Scopus: listed (also Medline, SCIE). CiteScore 11.9, Impact Factor 6.4. APC USD 3,320; subscription no fee. Timeline: first decision not stated; 98 days after review; 163 days to acceptance; 5 days to online.
- Similar published paper: none verified. The search for chest X-ray dataset bias in CMPB returned CXR classification papers and "AI explainability and bias propagation in medical decision support" (PII S0169260724004589, not chest X-ray, DOI not captured); no chest X-ray shortcut or leakage paper confirmed.

### 1.5 Computer Methods and Programs in Biomedicine Update (CMPB Update, partner journal)
- Pages: https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine-update/publish/guide-for-authors ; .../about/insights
- Scope quotes: "innovative computing methodologies and software systems research"; article types "Original Research, Systematic Reviews and or Meta-analysis, Methodology, Research Letters, Viewpoint". Word limit: not stated in what I read. Abstract 250 words; keywords 1-7.
- Scopus: listed (also DOAJ). CiteScore 12.6. Gold OA only, APC USD 2,990 (Geographical Pricing). Timeline: first decision and review time not stated; 151 days to acceptance; 4 days to online.
- Similar published paper: none verified.

### 1.6 Intelligent Systems with Applications (ISWA)
- Pages: https://www.sciencedirect.com/journal/intelligent-systems-with-applications/publish/guide-for-authors ; .../about/insights
- Scope quotes: "Submissions must be novel, technically sound, and clearly presented"; topics include "Machine Learning (ML) and Neural Computing" and application area "Medical Treatment and Health"; "Pattern Recognition (PR) and Computer Vision (CV)".
- Types: "regular papers and survey articles". Word/page limit: not stated. Abstract 250 words; keywords 1-7; highlights 3-5.
- Scopus: listed (also DOAJ). CiteScore 13.1, Impact Factor 5.5. Gold OA only, APC USD 2,830 (Geographical Pricing). Timeline 2 / 50 / 133 / 8 days.
- Similar published paper: none verified (the CXR bias search for ISWA returned a blank page; not retried successfully).

### 1.7 Array
- Pages: https://www.sciencedirect.com/journal/array/publish/guide-for-authors ; .../about/insights
- Scope quotes: topics include "Medical Informatics and Biomedical Engineering", "Computer Vision, Speech and Pattern Recognition", "Control & Signal Processing"; "Submissions must be novel, technically sound, and clearly presented".
- Types: "technical notes (... maximum of 10 pages in the standard Elsevier format) and regular papers" (regular papers have no stated limit).
- Scopus: listed (also DOAJ, ESCI). CiteScore 10.1, Impact Factor 5.3. Gold OA only, APC USD 2,770 (Geographical Pricing). Timeline 9 / 64 / 130 / 6 days.
- Similar published paper: none verified (search returned blank).

### 1.8 Medical Engineering & Physics (MEP)
- ScienceDirect journal pages `medical-engineering-and-physics/about/insights` and `/publish/guide-for-authors` returned "Page not found" on 2026-11-03. The IOP Publishing page https://publishingsupport.iopscience.iop.org/journals/medical-engineering-physics/about-medical-engineering-physics/ states it is published by IOP Publishing on behalf of IPEM (hybrid OA; standard APC USD 3,670; original research up to 10,000 words; Scopus listed). So on what I saw it is not an Elsevier journal at present; excluded from the Elsevier shortlist. Scope is devices, biomechanics, biomaterials, rehabilitation; chest X-ray evaluation is off-scope. (The Elsevier shop page appeared in search results but I did not open it.)

### 1.9 Pattern Recognition Letters (PRL)
- Pages: https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors ; .../about/insights
- Scope quotes: "rapid publication of concise articles of a broad interest in pattern recognition"; "Biometrics, biomedical pattern analysis and information systems"; research reports are "concise summaries of methodological inventions and findings" or "significant and novel applications of an established technique".
- Limit: "double column format and should not exceed 7 pages", including "narrative, figures, tables, references". 13k words cannot be reduced to this without writing a different paper.
- Scopus: listed (also SCIE). CiteScore 7.8, Impact Factor 3.5. APC USD 2,320; subscription no fee. Timeline 17 / 95 / 196 / 10 days.
- Similar published paper: none verified (search returned blank).

### 1.10 Computerized Medical Imaging and Graphics (CMIG)
- Pages: https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/publish/guide-for-authors ; .../about/insights
- Scope quotes: "publishes pioneering research"; "emphasizes algorithmic advances, novel methodologies and clinically oriented applications"; X-ray is a listed modality. Word limit: not stated in what I read.
- Scopus: listed (also Medline, SCIE). CiteScore 8.8, Impact Factor 5.5. APC USD 2,920; subscription no fee. Timeline 3 / 70 / 145 / 10 days.
- Similar published paper: none verified (the search returned reviews and an unrelated breast paper; later searches blank).

### 1.11 Other Elsevier titles checked
- Artificial Intelligence in Medicine (https://www.sciencedirect.com/journal/artificial-intelligence-in-medicine/publish/guide-for-authors ; .../about/insights): requires manuscripts with "Strong novelty of method and theory related to AI and computer science techniques". CiteScore 15.0, Impact Factor 7.8, Scopus listed, APC USD 3,440, subscription no fee, timeline 12 / 147 / 303 / 7 days. Excluded for S1 and S2: the page asks for method novelty, so an analysis paper is a poor fit, and acceptance takes 303 days.
- Intelligence-Based Medicine (https://www.sciencedirect.com/journal/intelligence-based-medicine/publish/guide-for-authors ; .../about/insights): scope text names "ethics and bias" and "medical imaging"; "The primary focus of the journal is on the clinical perspective and translation". Accepts "original research articles, review articles". Word limit not stated. CiteScore 4.7 (shown), Scopus listed (also DOAJ). Gold OA, APC USD 2,500 (Geographical Pricing). Timeline 5 / 28 / 88 / 4 days. Only journal whose scope text names bias explicitly; clinical-translation focus and the low CiteScore are the cautions.
- No Elsevier title whose aims and scope name "shortcut learning" or "dataset bias" in imaging AI was found in the pages above; Intelligence-Based Medicine is closest ("ethics and bias"). Medical Image Analysis and Journal of Biomedical Informatics were not opened.

## 2. Fit ratings (reviewer view: analysis/measurement contribution, no new architecture)

| Journal | S1 | S2 | One-line reason |
|---|---|---|---|
| BSPC | Medium | Medium | Scope names robustness, benchmark papers, frequency methods and images; but a full paper is defined around "a novel technique" at about 5,000 words, so S1 lacks the technique and S2's is weak (one seed) |
| IMU | High | High | Imaging informatics in scope, evaluation-style CXR paper already published there, no method-novelty line, no stated length cap; cost is the gold-OA APC |
| CBM | Low | Low | Web of Science delisting and integrity notice on its own page; guide rejects weak method papers |
| CMPB | Low | Low | Asks for "new computer methodologies" and about 3,500 words; S1/S2 would need a three-quarter cut |
| CMPB Update | Medium | Medium | Scope generic ("Original Research", "Methodology"); limit and review speed not stated; one gold-OA price |
| ISWA | Medium | Medium | Medical health listed under intelligent systems; "must be novel" is a risk; no length limit stated |
| Array | Medium | Medium | Lists medical informatics, vision and signal processing; generic CS venue, ESCI; novelty line as ISWA |
| PRL | Low | Low | 7-page cap including references; wants methodological inventions |
| CMIG | Low | Low | "Algorithmic advances, novel methodologies" is its stated emphasis |
| AIIM | Low | Low | Explicit demand for method novelty; 303 days to acceptance |
| Intelligence-Based Medicine | Medium | Low | Names "ethics and bias", but clinical-translation emphasis and lowest CiteScore shown; S2's band-stop method is off its focus |
| MEP | not applicable | not applicable | Not on ScienceDirect (IOP-published) per pages seen |

## 3. Recommendations

Judgement note: the ratings and the two recommendations below are my reading of the pages, not publisher statements. The one stated fact that decides BSPC for S1 is the definition of a full paper (a novel technique with a proving study).

### S1 (no training): Informatics in Medicine Unlocked; fallback Biomedical Signal Processing and Control
- Why IMU: no stated length cap, imaging is named, an evaluation-style multi-vendor CXR paper without a new network is already in the journal (DOI above), stated 36 days after review and 114 days to acceptance. Realistic acceptance is highest here.
- Cost: gold OA only, USD 2,710 with Geographical Pricing; the student's actual price is not shown. Decide this before submitting (ask the department about institutional agreements; the page says institutions may have them).
- Fallback BSPC: no fee on the subscription route and a better fit for the Fourier-band angle, but the paper must become "normally about 5,000 words", and the missing novel technique is the likely reviewer objection. Technical Note (5,000 words, maximum 5 illustrations, "would not normally look at data from a large cohort") is not suitable.
- Manuscript changes for IMU: keep a single full paper; cut about 20-30% (merge related work into introduction; move per-class table, run comparison and hyperparameter tables to supplementary; keep C1/C3/C4 results in the main text); abstract under 250 words; 1-7 keywords; highlights file of 3-5 bullets at most 85 characters; ethics statement saying why none is needed (public data); keep the patient-identity caveat and data/code statement; graphical abstract optional until its requirement wording is read.
- Manuscript changes for BSPC fallback: cut to roughly 5,000-6,500 words of main text (drop sections 3 and 5 detail to supplementary, shorten results and discussion, single-run comparison table only); reframe the abstract and highlights around robustness analysis and benchmark; prepare a graphical abstract; CRediT and AI declaration (already in `declarations.tex`); ask the editor-in-chief about length before full submission if possible.

### S2 (S1 + C2): Biomedical Signal Processing and Control; fallback Informatics in Medicine Unlocked
- Why BSPC: C2 supplies a method component, which is what the full-paper definition asks for, and band-stop filtering sits inside the signal-processing scope. Risk: one exploratory seed will draw a major-revision or reject; present C2 as exploratory in the title, abstract and highlights, and lead with the S1 measurements.
- Fallback IMU: if the C2 evidence is weak or the BSPC length cut is not acceptable to the student, the same paper fits IMU with fewer constraints, at the APC.
- Manuscript changes: as for BSPC above, plus a matched-energy random band-stop control and the seed spread of runs 2-4 stated beside the single C2 seed (already planned in `docs/publication/archive/NOVELTY_ROADMAP.md`). Do not claim an improvement from one run.

### Further fallbacks if both fail
CMPB Update (gold OA USD 2,990) for S1 or S2; ISWA or Array (gold OA, novelty line) as last resorts. Do not use CBM, AIIM, PRL, CMPB or CMIG for these scopes.

## 4. Open items
- Check each ISSN in the Scopus Sources list and record the date: BSPC 1746-8094, IMU 2352-9148, CMPB Update 2666-9900, ISWA 2667-3053, Array 2590-0056, CMPB 0169-2607, CMIG 0895-6111, PRL 0167-8655, CBM 0010-4825.
- Re-open the Insights pages on submission day; figures change.
- Confirm the student's APC under Geographical Pricing for IMU/CMPB Update/ISWA/Array (not shown on pages).
- Similar-paper searches for CMPB, CMPB Update, ISWA, Array, PRL and CMIG were limited (ScienceDirect search returned blank pages after repeated queries); the absence of a verified example there does not show absence.
- Per the thesis-first rule in `docs/publication/archive/NOVELTY_ROADMAP.md`, nothing is submitted until the thesis is final; the EMBC overlap must be declared.

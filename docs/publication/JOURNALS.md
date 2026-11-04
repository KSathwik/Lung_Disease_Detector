# Track A: journal venues

> Decision (2026-11-04, student): target journal is Informatics in Medicine Unlocked (IMU); Biomedical Signal Processing and Control (BSPC) is the fallback. Reason: diagnostic study after the C2 NO-GO. EMBC 2027 is unchanged.

> Note (2026-11-03 night): the current venue analysis per scope is `docs/publication/JOURNAL_SCOPE_MATCH.md`; the ranking below is the earlier one and is kept for reference.

Prepared 2026-11-03. All "accessed" dates are 2026-11-03. Figures are copied from the pages named; they change, so re-check on the submission day.
Work to be published: leakage-controlled, multi-source six-class chest X-ray (CXR) classifier (DenseNet-121), patient-grouped splits, once-only test and external (Montgomery) evaluation, source-shortcut analysis, audit of an earlier leaking pipeline. Single seed (42), CPU, time-boxed. Final numbers are pending (model chosen after run 2).

## 1. Primary target: Biomedical Signal Processing and Control (BSPC), Elsevier

Sources (accessed 2026-11-03):
- Guide for authors: https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors
- Journal insights: https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/about/insights
(Scripted fetches get HTTP 403; the pages were read in a browser pane.)

| Item | What the official pages say |
|---|---|
| Owner | Official journal of the European Alliance of Medical and Biological Engineering and Science (EAMBES) |
| Metrics shown | CiteScore 13.0; Impact Factor 5.7 |
| Scope | Original research on the "theory, methodology, and application of signal and data processing techniques" for health. Listed topics include "Medical image processing and analysis", "Machine learning, deep learning, and statistical methods for biomedical signal and image analysis", "Signal quality assessment, artifact detection, denoising, and robustness analysis", and "benchmark or dataset papers" are welcomed. Emphasis on interpretability, robustness and translational relevance. |
| Article types | Full paper (normally about 5,000 words); Short communication (about 2,000 words); Technical Note (5,000 words, max 5 illustrations, no large validation cohort); Review (10,000 words); Perspective (invited). Abstract under 250 words. |
| Peer review | Single anonymised; editors screen first; normally at least two reviewers. |
| Required at submission | Highlights (3 to 5 bullets, max 85 characters each, separate editable file with "highlights" in the name); abstract max 250 words; 1 to 7 keywords; CRediT author statement (required); declaration of competing interests via the Elsevier declarations tool (Word file uploaded, no signatures); funding statement; data statement. Research data "Option C": deposit data in a repository, cite and link it, or explain why it cannot be shared. |
| Graphical abstract | "Encouraged", not mandatory; at least 1328 px wide, readable at 13 cm; TIFF/EPS/PDF/MS Office. |
| Generative-AI declaration | Required if AI tools were used: a new section before the reference list titled "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", using the statement format on the page. Not needed for basic grammar/spelling/reference tools. AI cannot be an author. AI must not create or alter primary/patient images. |
| Files | .doc/.docx or .tex (PDF not acceptable as source). Word: single column; double column only for LaTeX. Figures as separate files; photographs min 300 dpi; tables as editable text. |
| References | Any consistent style at submission; journal style applied at proof: numbered in order of appearance in square brackets, journal names abbreviated by LTWA, DOIs where available, datasets/software citable. Every reference must be a real source. |
| Preprints / prior work | Preprints allowed (not prior publication). Submission declaration: work "has not been published previously except in the form of a preprint, abstract, published lecture, academic thesis, or registered report" and is not under consideration elsewhere; see Elsevier's policy on multiple, redundant or concurrent publication (linked in the guide, not read in full). |
| Open access | Gold OA APC USD 3,470 (excluding taxes; "actual APCs may differ"). Subscription route: no fee to authors. |
| Publisher-stated timeline (Insights) | 10 days submission to first decision; 66 days submission to decision after review; 203 days submission to acceptance; 12 days acceptance to online. Acceptance rate: not stated. |
| Indexing (Insights) | Scopus, Science Citation Index Expanded, SCImago (SJR), SNIP, Ei Compendex |
| Scopus status | Listed as covered on the journal's ScienceDirect Insights page. Not checked in Scopus itself (see section 3). |

### Scope fit (honest assessment)
- Fits: medical image analysis with deep learning, robustness analysis, benchmark papers, interpretability (Grad-CAM).
- Weaker: the journal's identity is signal processing and decision/control loops; CXR classification is in scope but not central. Our contribution is an auditable benchmark, leakage audit and shortcut analysis, not a new network. The defensible angle is robust, reproducible evaluation, which matches "robustness analysis" and "benchmark" wording.
- No architectural novelty claim, no clinical-validity claim.

### What BSPC reviewers are likely to require that the current study lacks
This is the agent's judgement from field practice. The BSPC guide does not list it. The Computers in Biology and Medicine guide (section 2.8) states several of the same expectations.
1. Multiple seeds: mean and SD over 3 to 5 seeds, or an explicit seed-variation study. One CPU run invites "lucky seed".
2. Ablations: preprocessing (blur/CLAHE/low-pass), two-phase fine-tuning, class caps and loss weighting, backbone choice (at least one more backbone), CXR gate.
3. Statistical tests: paired comparisons (McNemar or bootstrap difference with CIs) between models and runs; calibration and per-class CIs.
4. Baselines: at least one other architecture on the same splits, and the V5-style leaky split re-run on the same data to measure the inflation, not only narrate it.
5. More external cohorts: Montgomery alone is small. Add Shenzhen or another held-out source; always report external sensitivity with specificity.
6. Deeper shortcut analysis: source-only probes, border/crop tests, leave-one-source-out training.
7. Reporting: CLAIM and TRIPOD+AI items; the patient-identity limitation (TBX11K and COVID-19 Radiography grouped per image); data and code deposit (Option C).
8. Literature comparison only with clearly different test sets noted.
Practical point: 3 to 5 CPU seeds of DenseNet-121 is the main cost. If infeasible, prefer a venue with a lower novelty bar and state claims narrowly.

## 2. Other candidate journals

### 2.1 Journal of Imaging Informatics in Medicine (JIIM), Springer Nature / SIIM
- Pages: https://link.springer.com/journal/10278 ; https://link.springer.com/journal/10278/how-to-publish-with-us ; https://link.springer.com/journal/10278/submission-guidelines
- Metrics: Impact Factor 3.1 (2025); 5-year 3.1; median submission to first decision 5 days.
- Model/APC: hybrid. OA APC USD 4,890 (GBP 3,090 / EUR 3,890). Subscription route has no APC; both offered after acceptance.
- Scope: imaging informatics in all medical imaging environments, including analytics and decision support. Recent articles include deep-learning classification and segmentation, so the fit is reasonable; reproducible pipelines suit "informatics".
- Structure: Abstract (150 to 250 words), Introduction, Materials and Methods, Results, Discussion, Conclusions, Acknowledgements; Declarations section (ethics, consent, funding, competing interests, data/code availability, author contributions) before references.
- Scopus: SCOPUS listed on the Springer page (with SCIE, Medline, PubMed Central, INSPEC). Timelines after first decision: not stated.

### 2.2 Physical and Engineering Sciences in Medicine (PESM), Springer Nature
- Pages: https://link.springer.com/journal/13246 ; https://link.springer.com/journal/13246/how-to-publish-with-us
- Metrics: Impact Factor 2.4 (2025); 5-year 2.7; median submission to first decision 13 days.
- Model/APC: hybrid. OA APC USD 3,790 (GBP 2,490 / EUR 2,890). Page: "There are no publication charges except for special services" for the traditional route.
- Scope: physics and engineering in medicine and biology; article types reviews, scientific papers, notes, letters. Latest articles include a CXR lung-segmentation paper, so imaging AI is accepted. The Apostolopoulos and Mpesiana COVID CXR paper in our reference list is from this journal.
- Scopus: SCOPUS listed on the Springer page (also SCIE, Medline, EI Compendex). Length and structure rules: not read.

### 2.3 Medical & Biological Engineering & Computing (MBEC), Springer Nature / IFMBE
- Pages: https://link.springer.com/journal/11517 ; https://link.springer.com/journal/11517/how-to-publish-with-us
- Metrics: Impact Factor 3.1 (2025); 5-year 3.0; median first decision 6 days.
- Model/APC: hybrid. OA APC USD 4,090 (GBP 2,790 / EUR 3,190). Subscription route: no APC.
- Scope: biomedical and clinical engineering, computer-based methodologies. The page links a "Notice for Authors - Desk Rejection Policy from the Journal Editors" (not read; read before submitting).
- Scopus: SCOPUS listed on the Springer page (also SCIE, INSPEC, DBLP).

### 2.4 Informatics in Medicine Unlocked (IMU), Elsevier
- Pages: https://www.sciencedirect.com/journal/informatics-in-medicine-unlocked/about/insights ; .../publish/guide-for-authors
- Metrics: CiteScore 10.5 (no Impact Factor shown). Timeline: 2 days first decision; 36 days to decision after review; 114 days to acceptance; 4 days to online.
- Model/APC: gold OA only, APC USD 2,710 excluding taxes; Geographical Pricing for OA applies (price set by author-country income level; the India figure is not shown, so the actual APC is not confirmed).
- Scope: medical informatics incl. imaging; original research, reviews, tutorials. Guide: 2 to 5 referees, average evaluation under 4 weeks, ethics statement required (state why none needed for public data). Data "Option B" (encouraged).
- Scopus: Scopus and DOAJ listed on the Insights page.

### 2.5 Computerized Medical Imaging and Graphics (CMIG), Elsevier
- Pages: https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/about/insights ; .../publish/guide-for-authors
- Metrics: CiteScore 8.8; Impact Factor 5.5. Timeline: 3 days first decision; 70 days after review; 145 days to acceptance; 10 days to online.
- Model/APC: hybrid; OA APC USD 2,920; subscription no fee. Data "Option C".
- Scope: stresses "algorithmic advances, novel methodologies and clinically oriented applications"; X-ray is a named modality. High method-novelty expectation, so a benchmark/audit paper is a harder fit.
- Scopus: Scopus, Medline, SCIE listed on the Insights page.

### 2.6 Scientific Reports, Nature Portfolio
- Pages: https://www.nature.com/srep/about ; https://www.nature.com/srep/open-access
- Model/APC: gold OA only; APC GBP 2,290 / USD 2,850 / EUR 2,490 (plus taxes where applicable).
- Scope: all natural sciences, medicine and engineering. Page says it is indexed in "Web of Science, Scopus, Dimensions, Google Scholar, DOAJ and SAO/NASA ADS". Impact factor, review timeline and length limits: not confirmed (metrics page returned 404; submission guidelines not read).
- Expectation (from general Nature Portfolio practice, not read here): technical soundness over novelty; data and code availability statements.

### 2.7 Diagnostics, MDPI
- Pages: https://www.mdpi.com/journal/diagnostics/about ; .../apc ; .../indexing
- Page metrics: Impact Factor 3.8; CiteScore 6.9; 21 days to first decision; 3 days acceptance to publication.
- Model/APC: gold OA; APC CHF 2,600.
- Scope: medical diagnosis, including "machine learning and artificial intelligence in diagnostics" and medical imaging; no length limit.
- Indexing per MDPI: Scopus, Web of Science (SCIE), PubMed/PMC, DOAJ.
- Caution: some institutions are sceptical of MDPI; check department acceptance first.

### 2.8 Computers in Biology and Medicine (CBM), Elsevier: NOT RECOMMENDED
- Pages: https://www.sciencedirect.com/journal/computers-in-biology-and-medicine/publish/guide-for-authors ; .../about/insights
- The journal's own announcement: placed "On Hold" in April 2024; Web of Science Core Collection coverage discontinued on 17 November 2025; content after Volume 172 (2024) not indexed there; Elsevier is investigating. A secondary report (https://phdtalks.org/2026/01/computers-in-biology-and-medicine-journal-elsevier.html, not authoritative) says Scopus coverage continues. Scopus status: not confirmed.
- Its guide says it does not consider submissions "with unclear dataset division into training/validation/test sets", "minor architecture/model modifications and only slight increase in performance", or "lack of comparison with real state-of-the-art methods".
- Insights: CiteScore 13.1; APC USD 3,080; acceptance rate 12%.

## 3. Scopus verification note
The requested authoritative source (Scopus Sources list) is an Excel download and was not downloaded. SCImago (built on Scopus data) was blocked by a bot-detection challenge, which was not bypassed. So for every journal above Scopus status means "listed as covered on the publisher's own page on 2026-11-03", not "verified in Scopus". Before submission, search each ISSN at https://www.scopus.com/sources and note the date: BSPC 1746-8094; JIIM eISSN 2948-2933; PESM 2662-4729; MBEC 0140-0118; IMU 2352-9148; CMIG 0895-6111; Sci Rep 2045-2322; Diagnostics 2075-4418.

## 4. Ranked recommendation

| Rank | Journal | Reason | Main risk |
|---|---|---|---|
| 1 | BSPC | Requested target; scope names robustness analysis, benchmark papers, medical image ML; Scopus + SCIE; first decision stated 10 days; subscription route free. | Higher novelty bar; reviewers will ask for section 1 items; desk-rejection risk. |
| 2 | JIIM | Imaging informatics and reproducible pipelines fit; median first decision 5 days. | OA APC USD 4,890 (subscription avoids it). |
| 3 | PESM | Accepts imaging AI; lower bar; subscription free. | Lower impact; physics-leaning. |
| 4 | MBEC | Computer-based methods; IFMBE. | Read desk-rejection notice; novelty. |
| 5 | Scientific Reports | Soundness-focused venue suits an honest evaluation study. | APC USD 2,850 only (no subscription route). |
| 6 | IMU | Fast, gold OA, Scopus. | Reputation; APC unconfirmed for India. |
| 7 | CMIG | Best imaging scope wording. | Needs method novelty we lack. |
| 8 | Diagnostics | Fast, indexed. | Institutional acceptance; CHF 2,600. |
| excluded | CBM | WoS delisting, integrity investigation. | |

Top recommendation: BSPC, after adding multi-seed runs, one more baseline and ablations. Fallback if these cannot be done: JIIM or Scientific Reports, with claims stated narrowly.

## 5. Open items
- Authors to confirm author order and corresponding author (Elsevier discourages changes after submission and forbids them after acceptance).
- OA versus subscription decision comes after acceptance; APC support or waivers were not researched.
- Prepare highlights, CRediT, declaration of interest, AI statement, data statement with repository link, optional graphical abstract.
- JNTUH R22 regulations (a different university, see REPORT_TEMPLATE_FINDINGS.md) require a published paper for dissertation submission; KU's rule for this is not found, so ask the department whether a journal publication is required.

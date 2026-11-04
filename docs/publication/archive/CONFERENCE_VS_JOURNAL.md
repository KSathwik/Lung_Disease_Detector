> **Archived 2026-11-04 (superseded):** superseded by [JOURNAL_SCOPE_MATCH.md](../JOURNAL_SCOPE_MATCH.md) and [SUBMISSION_TRACKER.md](../SUBMISSION_TRACKER.md) (IMU target decision of 2026-11-04). Kept for traceability; the EMBC/journal differentiation points still apply to the manuscripts.

# Conference paper versus journal article: differentiation note

Prepared 2026-11-03. Purpose: keep the IEEE EMBC 2027 paper and the Biomedical Signal Processing and Control (BSPC) article distinct, so that the journal article is a substantially extended work and not duplicate publication. Venue facts come from [CONFERENCES.md](../CONFERENCES.md), [JOURNALS.md](../JOURNALS.md) and [SUBMISSION_TRACKER.md](../SUBMISSION_TRACKER.md); where those files do not say something, this note does not either. Author order is confirmed by the project owner; the corresponding author is [TO BE CONFIRMED]; contributions must be verified and all authors must approve before any submission [AUTHORS TO CONFIRM].

**Status of the extensions.** The journal items marked "planned" below are not yet in the manuscript drafts or have not been run. As of this note the draft `docs/manuscripts/journal_bspc/` contains the run 1 versus run 2 comparison, calibration and the input-gate discussion, but `journal_bspc/REVIEWER_RISK.md` still lists single seed, no preprocessing ablation and only indirect shortcut evidence as open risks. Do not describe a planned analysis as done until its results exist.

## 1. What each paper contains

| Aspect | IEEE EMBC 2027 paper (`docs/manuscripts/conference/`) | BSPC journal article (`docs/manuscripts/journal_bspc/`) |
|---|---|---|
| Format and length | IEEE template; 4 pages of technical content plus a 5th page for references only; at most 20 references; up to 35 authors (CONFERENCES.md 1.1) | Elsevier elsarticle; full paper about 5,000 words, abstract under 250 words, highlights file, CRediT, generative-AI declaration, data statement (JOURNALS.md section 1) |
| Core message | Internal accuracy versus external generalisation of a leakage-controlled six-class CXR classifier | Same theme as an auditable benchmark paper, with the evidence needed to support it |
| Benchmark rebuild | Sources, label rules, caps, patient-grouped split and leakage checks, in short form | Full construction: acquisition, labelling rules, hashing and near-duplicate detection, caps, grouping, both dataset builds, external cohort |
| Model runs | The single selected run (seed 42, CPU, time-boxed) | Run 1 and run 2 compared on a common validation set, plus planned seed variability over 3 seeds |
| Results | Internal test metrics, external Montgomery result with specificity, source effects; numbers from the shared results macros | The same kinds of results with per-class, per-source and calibration detail, and the run comparison |
| Internal versus external finding | Yes, the central finding | Yes, discussed in depth with the audit and benchmark-design implications |
| Preprocessing ablation | No | Planned: none / blur / CLAHE / low-pass variants (REVIEWER_RISK.md items 5, 16) |
| Source-identifiability probe, per-source analysis | Short "source effects" subsection only | Planned probe of how well the source can be identified, with a fuller per-source analysis |
| CXR-gate calibration | No | Gate discussion and calibration of its thresholds (`lungai/configs/lungai_densenet121.json`, `inference.cxr_gate`) |
| V5 audit | At most a brief mention | The audit of the earlier V5 pipeline (patient leakage, proxy labels, external specificity 0/80, lost checkpoint) as its own discussion subsection |
| Discussion | Short | Extended: internal versus external behaviour, benchmark design, deployment lesson, limitations, future work |

Both papers share the project facts and the numbers macros (`docs/manuscripts/shared/results_macros.tex`); sharing numbers is expected, sharing text is not (section 4).

## 2. Disclosure wording if the conference paper is published first

Venue rules (SUBMISSION_TRACKER.md, publication-ethics items 1 to 4): BSPC's submission declaration requires that the work "has not been published previously except in the form of a preprint, abstract, published lecture, academic thesis, or registered report" and is not under consideration elsewhere. Elsevier's policy on multiple, redundant or concurrent publication applies and was not read in full. The tracker advises citing the conference paper, stating what is new and uploading it for the editors. The text below is a draft for the authors to edit; the bracketed items cannot be filled in yet.

Cover letter (replaces the bracketed originality item in `journal_bspc/cover_letter.md`):

> A shorter conference version of this work, "[CONFERENCE PAPER TITLE]", was [published in / accepted at / submitted to] the Proceedings of the 49th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC 2027) [DOI or status: TO BE PROVIDED]. The present manuscript is a substantially extended version. It adds [list only what was actually completed: seed variability over three seeds; a preprocessing ablation; a source-identifiability probe and per-source analysis; a comparison of two training runs on a common validation set; calibration of the input gate; the audit of the earlier pipeline; and a fuller discussion]. The conference paper is cited as reference [N] and a copy is uploaded for the editors. No figure or text has been reused without permission.

Manuscript (footnote on the title page or a sentence in the Introduction, whichever BSPC accepts):

> A preliminary version of this work was presented at IEEE EMBC 2027 [N]. This article extends it with [the same list].

If the journal is submitted while the conference paper is still under review, say so in the cover letter and note that the two have different scope; see item 1 of section 3.

## 3. Checklist to avoid duplicate publication

1. Do not have both manuscripts under review at once with substantially the same content. The tracker's suggested order is: submit EMBC by 24 January 2027 (11:59 PM AoE, no extension; accept/reject 16 April 2027), and submit the journal paper after the conference decision or earlier only if the extension work is complete (SUBMISSION_TRACKER.md, order note and item 1). ISBI, noted as a precedent, forbids a "substantially similar" paper under review elsewhere.
2. Extension is real. Complete the planned analyses (3 seeds, ablation, probe and per-source analysis, run comparison, gate calibration, V5 audit) and put their results in the journal paper. The tracker says common practice is substantial new content, often about 30 percent or more, but the exact threshold is not stated in the sources read; confirm with the BSPC editor.
3. Different text. Write the journal Introduction, Methods and Discussion afresh; do not paste conference paragraphs. Re-draw conference figures or request permission.
4. Copyright. IEEE copyright transfer limits reuse of text and figures (tracker item 3): read the EMBC copyright form at acceptance and, if needed, request permission or rewrite.
5. Check the EMBC post-publication policy (listed on its author-guidelines page, not read) before relying on any reuse plan.
6. Cite the conference paper in the journal manuscript, state in the cover letter what is new, and upload the conference PDF for the editors.
7. Preprints are allowed by BSPC and are not prior publication; if a preprint is posted, cite it consistently.
8. The M.Tech dissertation is allowed prior material at Elsevier ("academic thesis"); cite it and keep the similarity report (tracker item 5).
9. Same facts everywhere. Numbers come from the shared macros and the final model chosen after run 2; the same single-seed, no-clinical-validity, external-sensitivity-with-specificity and unknown-patient-identity statements apply to both papers.
10. Authors. Same four authors in the confirmed order in both papers; the corresponding author is [TO BE CONFIRMED]; contributions are verified and all authors approve each submission [AUTHORS TO CONFIRM]. BSPC does not allow author changes after acceptance (tracker item 6). One author must be a qualified reviewer for EMBC (reciprocal review).
11. AI-use disclosure for each venue as in tracker item 8; the BSPC declaration section and the IEEE-required disclosure are written separately and checked by the authors.
12. Keep the venue facts current: all dates and rules above were read on 2026-11-03 and must be re-checked on the submission day.

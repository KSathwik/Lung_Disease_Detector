# Submission tracker

> Status pointer (2026-11-04): `docs/TRACKER.md` is the single status page (see `PROJECT_EXECUTION_STATUS.md` for the handoff). This file is the venue-specific submission table only.

> Decision (2026-11-04, student): target journal is Informatics in Medicine Unlocked (IMU); Biomedical Signal Processing and Control (BSPC) is the fallback. Reason: diagnostic study after the C2 NO-GO. EMBC 2027 is unchanged.

> Note (2026-11-03 night): the current venue analysis per scope is `docs/publication/JOURNAL_SCOPE_MATCH.md` (the fallback order in the table below is the earlier one).

Updated 2026-11-03. Statuses: not started / drafting / ready / submitted. Owner is a placeholder: [AUTHORS TO CONFIRM]. Sources for rows are in JOURNALS.md and CONFERENCES.md.

| Venue | Track | Deadline | Template | Page limit | Required items | Status | Owner | Notes |
|---|---|---|---|---|---|---|---|---|
| IEEE EMBC 2027 (Singapore, 11 to 15 Jul 2027) | B conference | 24 Jan 2027 (11:59 PM AoE, no extension); portal opens Dec 2026 | IEEE template from the EMBC paper management system | 4 pages technical + 5th page references only (max 20 refs) | IEEE-format PDF; named reviewer among authors; AI-use disclosure; ethics/data statement; author list (max 35) | drafting (`docs/manuscripts/conference/`) | [AUTHORS TO CONFIRM] | Final model selected after run 2 (2026-11-03 about 17:00); results macros regenerate. Registration fee unpublished. |
| IEEE CONECCT 2027 (Bangalore, 29 to 31 Jul 2027) | B conference (backup) | 15 Dec 2026 | Not retrieved | Not retrieved | Not retrieved | not started | [AUTHORS TO CONFIRM] | Weak topical fit; re-check when CFP opens in Dec 2026. |
| Biomedical Signal Processing and Control (Elsevier) | A journal | Rolling | Elsevier elsarticle (LaTeX) or Word single column | About 5,000 words (full paper); abstract under 250 words | Highlights file (3 to 5 bullets, 85 characters max); CRediT; declaration of interests (Word file); generative-AI declaration section; funding and data statements (Option C: repository link or reason); optional graphical abstract; figures as separate files (300 dpi+) | drafting (`docs/manuscripts/journal_bspc/`) | [AUTHORS TO CONFIRM] | Needs multi-seed, ablations, tests, baseline, more external cohorts before submission (JOURNALS.md section 1). APC USD 3,470 only if OA chosen after acceptance. |
| JIIM (Springer) | A journal (fallback 1) | Rolling | Springer submission guidelines | Not stated in pages read | Declarations section; abstract 150 to 250 words | not started | [AUTHORS TO CONFIRM] | OA APC USD 4,890 optional. |
| Scientific Reports (fallback 2) | A journal | Rolling | Nature Portfolio | Not confirmed | Data/code availability | not started | [AUTHORS TO CONFIRM] | APC USD 2,850 (no subscription route). |

Suggested order: submit EMBC 2027 by 24 Jan 2027; submit the extended journal paper after the conference decision (16 Apr 2027) or earlier if the extension work is complete, never both with the same content under review at once.

## Publication-ethics section

1. No simultaneous submission of substantially identical manuscripts. ISBI states that no paper "substantially similar in content" may be submitted to another conference or workshop during its review period, and Elsevier's submission declaration says the article is "not under consideration for publication elsewhere" (BSPC guide). Treat these as standing rules for all venues.
2. Conference first, journal extended. The journal version must add substantial new work, not only text. Planned extensions: (a) 3 to 5 seeds with mean and SD; (b) ablations (preprocessing, fine-tuning schedule, class caps, backbone); (c) statistical tests and calibration; (d) at least one additional external cohort; (e) deeper shortcut analysis and the quantified leaky-split re-run; (f) full methods, per-class and per-source analysis that the 4-page limit omits. Common practice is substantial new content (often about 30 percent or more); the exact threshold is not stated in the sources read, so confirm with the target journal's editor.
3. Disclose the conference paper. Cite it in the journal manuscript and in the cover letter, state what is new, and upload it for the editors. Elsevier policy on multiple, redundant or concurrent publication applies (linked from the BSPC guide; read it before submission). Check copyright: IEEE copyright transfer limits reuse of text and figures; request permission or rewrite and redraw.
4. Preprints are allowed by BSPC and ISBI; a preprint is not prior publication there. Check EMBC's post-publication policy (listed on its author-guidelines page, not read).
5. The M.Tech report (thesis) is allowed as prior material by Elsevier ("academic thesis"); cite it. Similarity limits in the thesis (see REPORT_TEMPLATE_FINDINGS.md) apply to overlap with the papers, so keep the similarity report.
6. Authorship: order confirmed by the project owner (1. Katkam Sathwik, 2. Prof. Nagavelli Ramana, 3. Dr. P. Tirupathi, 4. Dr. K. Srinivas); corresponding author [TO BE CONFIRMED]; CRediT contributions must be verified and all authors must approve before submission [AUTHORS TO CONFIRM]; BSPC does not allow changes after acceptance. Every author must meet the substantial-contribution criteria in the BSPC guide and approve the final version.
7. Data integrity: no invented results; report single-seed status honestly until multi-seed runs exist; external sensitivity only with specificity; state that patient identity is unknown for TBX11K and COVID-19 Radiography rows; no clinical-validity claim; no comparison with other papers' numbers as if on the same test set.
8. AI-assistance disclosure per venue:
   - BSPC / Elsevier: required statement at first submission in a section before the reference list, titled "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process"; wording: "During the preparation of this work the author(s) used [NAME OF TOOL/SERVICE] to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article." Grammar/spell checkers exempt. Authors, not the tool, fill in the tool name and reason.
   - IEEE EMBC 2027: disclosure per IEEE policy; unacknowledged generative AI prohibited (https://embc.embs.org/2027/author-guidelines/). Use the IEEE-required acknowledgements-section disclosure; exact wording not read.
   - Springer (JIIM, PESM, MBEC), Nature (Sci Rep), MDPI: policies not read; check at submission.
9. Reviewer duty: EMBC reciprocal review means one author must be qualified and willing to review.

## IMU submission checklist (added 2026-11-04)

- [ ] APC: check Informatics in Medicine Unlocked gold OA fee under geographical pricing and any institutional agreement before choosing OA
- [ ] Highlights file (3 to 5 bullets)
- [ ] Ethics statement (public datasets only; no new patient data)
- [ ] Generative-AI declaration section
- [ ] Data and code availability statement
- [ ] CRediT author statement
- [ ] Cover letter (state overlap with the EMBC paper)
- [ ] Scopus / Google Scholar novelty check immediately before submission
- [ ] All author approvals of the final manuscript
- [ ] Author contacts and corresponding author confirmed [AUTHORS TO CONFIRM]

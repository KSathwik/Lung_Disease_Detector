# Review of the novelty protocol, for the student

Advice only. The decision to approve is yours. Sources: `NOVELTY_PROTOCOL.md`, `artifacts/novelty/access_log.jsonl`, `c2_decision.json`, the two 2026-11-04 audits in `docs/AUDIT_FINDINGS.md`, and `git log`.

## 1. What this protocol is

It is a list of rules written down before any novelty result existed. The rules say which measurements to make (which frequency bands carry source information and disease information; what happens when source information is removed; how a model does on a source it never saw) and exactly how to decide whether an extra retraining (C2) is worth doing. Fixing the rules first stops anyone, including us, from choosing the analysis that looks best afterwards. It describes measurements only and makes no clinical claim.

## 2. Main rules

| Rule | What it means | Why it matters |
|---|---|---|
| Data and samples | Fixed build, fixed random sample (seed 0, caps 150/100/100 per source-class cell); small cells (under 20 images) are not scored. | Same images for every comparison, so differences are not sampling luck. |
| Frequency bands | Six octave bands A-F from coarse to fine detail, smooth edges, average brightness kept. | Defined before looking, so no band can be picked after the fact. |
| Scores | Source accuracy, disease score inside each source, and a source-minus-disease index; 95% bootstrap intervals, no p-values. | Disease is scored inside each source so source shortcuts cannot inflate it. |
| C1 | Keep or remove one band at a time and test how well a simple classifier still finds source and disease. | Gives the band-by-band map, the main novelty claim. |
| C3 | Repeatedly delete the directions in the features that identify the source (up to 20 rounds), then recheck disease accuracy. | Shows whether source and disease information can be separated linearly. |
| C4 | Train on five sources, test on the sixth. | Shows how much performance drops on an unseen source. |
| C2-lite | Cheap version of band removal using only a linear classifier on frozen features. | A low-cost check before spending a night of CPU on retraining. |
| C2 go/no-go | GO only if one band removes at least 5 points of source accuracy while disease score stays within 2 points, a random control does worse, and C2-lite agrees. | Thresholds were fixed before any result, so the outcome is not negotiable. |
| Test and Montgomery once only | Each analysis may score test and Montgomery once, in a set order, with a logged marker. | Prevents repeated peeking at the final-exam data. |
| Change control | After approval, any change must be logged before the affected result, with your approval. | Keeps the record honest. |

## 3. The seven clarifications (section 13)

The protocol was silent on these points, so the code made a choice. All were logged and committed (`c60e0c7`, 04:01 UTC) before the first test access (04:40 UTC).

1. Feature files are stored one per variant. Storage only; cannot change a result.
2. Disease score counts only images in qualifying cells. This follows the 20-image rule already in the protocol; it changes nothing the protocol had not already implied.
3. Ties in the choice of C go to the smaller C. Only matters on an exact tie, which is rare.
4. The reference head in C4 picks C using validation images from all six sources. Affects only the reference column, not the held-out source's own data.
5. Stop with an error if a validation class has no training images. Cannot occur with the stated counts.
6. A probability of exactly 0.5 counts as abnormal. Exact ties essentially never happen.
7. Criterion 3 uses point estimates because no interval rule was written. Evidence: C2-lite source drop was 0.025 (val), below 0.05, and its CI upper bound (0.042) is also below 0.05, so criterion 3 fails under either reading. Criterion 1 also fails for every band (largest source drop 0.028, band F), independent of this choice.

Other small notes: the matched-energy control details never ran (control "not run"), and the INLP stopping check uses the same fit whose directions are removed (INLP hit the 20-round cap in both extractors anyway). Honest limit: for items 1, 3, 4 and 6 the evidence is only the timing (logged before results) and the audit's statement that no result was affected; I did not re-run anything.

## 4. Compliance check

| Rule | Evidence | Verdict |
|---|---|---|
| Rules committed before first result | Approval commit `f427c04` 07:44 +0530 (02:14 UTC). Code `a3827d7` 08:15 +0530. Clarifications `c60e0c7` 09:31 +0530 (04:01 UTC). First metric marker 04:40:19 UTC. | Met. |
| Each test/Montgomery access once | Exactly 8 markers: c1-main, c1-served, c1-nofilter test; c3 test and Montgomery; c4 main and Montgomery; c2lite test. None forced, none repeated, no uncommitted novelty code. | Met. |
| C2 decision by the fixed rule | `c2_decision.json`: no band satisfies criterion 1, control not run per section 9, criterion 3 fails, so NO-GO. Matches the protocol text. | Met. |
| No setting changed after results | No change-log entry after a result; code at `d3a6788` for steps 2-7 (docs-only change from `c60e0c7`). | Met, as far as the record shows. |

Gaps, stated honestly:
- Markers exist for test and Montgomery only. Validation reads are not logged (the protocol allows repeated validation use), so "val first, then test" rests on the log text, not a marker.
- The protocol file was edited after `f427c04` (hash entry, acknowledgement, log rows, a corrected code-commit column for steps 0-1, found by the audit). These are log entries, not rule changes, but the file's own last-commit hash differs from the approval commit, so a reader should be told that.
- Step 0 timing ran at 02:46 UTC, after code `a3827d7` but with no metric; fine.
- The approval record is thin: your approval was "Proceed with your findings" and the seven clarifications were acknowledged with "Proceed". The audits (N7, W9) flag this.

## 5. Cautions before approving

- The manuscript and thesis already say the protocol was "approved by the student in chat on 4 November 2026". Approval makes that sentence true; keep that exact wording and never "pre-registered" (use "fixed in advance and recorded in the repository").
- Limits you should know: C2 did not run (NO-GO, listed as future work); INLP did not reach its stopping target in 20 rounds; Montgomery is not untouched (used in runs 1-5); the C4 patient-identity caveat applies; the trained-backbone numbers are not "held out"; the NIH LOSO gap is a metric artefact.
- Reasons NOT to approve: you do not understand a rule well enough to defend it to a reviewer; you believe a rule or number was changed after results (I found no evidence of this); or you want the approval recorded properly first (below).

## 6. Recommendation (advice only)

Approve with notes. The record shows rules committed before any result, one access per test and Montgomery split, and a mechanical NO-GO. The remaining weakness is the one-word approval trail. Write a short confirmation in your own words that you read and approve the protocol and the seven clarifications, and have it entered in section 13 before submission. Also ask that the manuscripts keep the phrase "approved by the student in chat" until then. The decision is yours.

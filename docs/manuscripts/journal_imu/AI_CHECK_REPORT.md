# AI-check report: IMU manuscript (humanize + ai-check pass, 2026-11-04)

Scope: `sections/*.tex`, `highlights.tex`, `supplementary.tex` (prose only), `cover_letter.md`. Method: the humanize rewrite protocol as far as it applies to formal academic text (hard rules 1, 4, 5, 6, 7; levers 2, 3 filler substitutions only, 4, 7, 9; semicolons allowed in the formal register, LaTeX quotes unchanged), then the nine-signal ai-check procedure. The pass does not try to defeat any detector. It does not add personal experience, deliberate errors or plausible-specificity frames, and the AI-use declaration (`declarations.tex`) is untouched. Hedges that carry scientific uncertainty ("consistent with", "indirect", "rough", "cannot be ruled out") stay, and so do technical terms and all `\R...` macros, numbers, citations, labels, table cells and confirmation markers.

Scores are 0 to 3 per signal (total out of 27). They are the auditor's judgement from the quoted evidence, not the output of a detector. The skill's verdict bands (0-4 Human, 5-8 Likely Human, 9-13 Uncertain) are calibrated on general prose; the calibration notes say that academic prose legitimately carries hedges and semicolons, and that a text which was drafted with AI assistance (as declared here) is expected to show a residual score.

## Hard-rule gate (counted on the finished files)

| Check | Result |
|---|---|
| Em dashes (U+2014) in all 11 files | 0 |
| Curly quotes or apostrophes (U+201C/D, U+2018/9) | 0 (LaTeX ``...'' quotes unchanged) |
| Banned vocabulary (delve, leverage, utilize, robust, comprehensive, furthermore, moreover, streamline, pivotal, crucial, notably, "it is important to note") | 0 hits in sections and cover letter ("Highlights" is the journal's required label) |
| Negation pivots ("not just", "not X but Y", "not X, it is Y") | 0 after edits (before: "not as a gap" in the introduction; "is not claimed as a method of ours: it is applied as" in the methods) |
| "more X than Y" comparatives | 0 |
| Semicolons | present, formal register (raw count of ";" in the files: 2, 2, 7, 20, 5, 39, 22, 1, 0, 10, 9 in file order of the table below, including a few LaTeX and table uses) |

## Sentence-length counts (hard rule 7), before and after

Counted by script on the prose only (tables, equations, captions, comments and LaTeX commands removed; each `\R` macro counted as one word).

| File | Sentences | Min | Max | SD | Share in 10-20 band | Verdict |
|---|---|---|---|---|---|---|
| 00_abstract before | 13 | 9 | 31 | 6.7 | 0.69 | fails (range 22 ok, mid band over half) |
| 00_abstract after | 12 | 9 | 35 | 8.0 | 0.42 | passes (range 26) |
| 01_introduction before / after | 39 / 39 | 5 / 5 | 68 / 68 | 15.0 / 14.9 | 0.28 / 0.28 | passes |
| 03_materials (unchanged) | 38 | 4 | 60 | 12.4 | 0.39 | passes |
| 04_methods before / after | 73 / 73 | 6 / 6 | 72 / 72 | 13.8 / 13.9 | 0.45 / 0.45 | passes |
| 05_experimental_design before / after | 25 / 24 | 8 / 8 | 54 / 54 | 11.6 / 12.3 | 0.36 / 0.33 | passes |
| 06_results before / after | 95 / 94 | 4 / 4 | 60 / 60 | 11.8 / 12.0 | 0.23 / 0.22 | passes |
| 07_discussion before / after | 60 / 60 | 2 / 2 | 116 / 112 | 20.3 / 18.1 | 0.25 / 0.22 | passes |
| 08_conclusion before / after | 10 / 10 | 7 / 7 | 52 / 63 | 11.9 / 14.2 | 0.00 / 0.00 | passes |
| cover_letter before / after | 19 / 19 | 4 / 4 | 183 / 184 | 39.9 / 40.1 | 0.32 / 0.32 | passes |

Remaining note: formal technical prose has few sentences of six words or fewer (the skill asks for one per 150 words). The abstract has none, because its 250-word limit and the need to keep "indirect" in a full sentence leave no room; the Methods have three in 1,650 words. Padding fragments into methods text would reduce precision, so this is left as a normal property of the register.

## Per-section ai-check

Legend: A perplexity, B burstiness, C hedge density, D structural, E specificity, F transitions, G punctuation, H voice, I rhetorical scaffolding.

### Abstract (`00_abstract.tex`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| After | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 4 |

Evidence before. B: sentence counts "19, 15, 15, 30, 11, 15, 31, 19, 16, 15, 28, 17, 9", nine of 13 in the 10-20 band, three consecutive within 5 words ("19, 16, 15"). C: "are often trained" (reflexive but accurate). H: opens with a framing sentence ("Chest radiograph classifiers are often trained on pooled public collections..."). I: "neither band removal nor linear erasure is a cheap remedy" (weak negation framing).
Edits. Joined the two background sentences ("..., so we asked where source information resides..."), lengthened two mid-band sentences so the share in the band is 0.42, and replaced "Evidence on model use of the source is indirect" by "Evidence that the model uses the source is indirect". Word count 250 (limit 250).
Remaining, normal for the register: structured Background/Methods/Results/Conclusions headings; "often" is accurate; the "neither ... nor" clause states a conclusion of the null result.

### Introduction (`01_introduction.tex`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 2 | 6 |
| After | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 5 |

Evidence before. B: one window of three consecutive sentences within 5 words (the counts otherwise range from 5 to 68 words, with 28% of sentences in the 10-20 band). C: "usually trained on images pooled"; double hedge "To our knowledge, within the searches described below, we are not aware of prior work that quantifies". D: five-item bold-headed contribution list (genuinely enumerated, so weak). H: opens with a general statement ("The chest radiograph is the most frequently acquired imaging examination"). I: "We regard the rule and its outcome as part of the design, not as a gap." (negation pivot, moderate); the five items open with the same article ("A leakage-controlled...", "A band-by-band map...", "A diagnostic application...", "A leave-one-source-out matrix...", "A recorded negative decision") (weak anaphora).
Edits. Removed ", not as a gap". Replaced "we are not aware of prior work that quantifies" by "no prior work quantifies" (the "To our knowledge, within the searches described below" qualifier stays).
Remaining, normal: the numbered contributions list and the closing roadmap paragraph are expected by the venue; the generic opener carries no checkable claim beyond what the cited literature supports and was left as written.

### Data and benchmark construction (`03_materials.tex`), unchanged

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 |
| After | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 |

Evidence. B: three windows of three consecutive sentences within 5 words (script count); the range is 4 to 60 words. H: impersonal, no first person, "polished neutral tone" (the register for a data section). No flagged vocabulary, transitions or scaffolding; every paragraph carries counts, hashes or thresholds. No edit was warranted. The only negated clauses are factual exclusions ("were not used", "never part of a development split").

### Methods (`04_methods.tex`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 2 | 5 |
| After | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 4 |

Evidence before. B: six three-sentence windows within 5 words, all long definitional sentences. G: 20 semicolons and several colon introductions, permitted in the formal register. H: impersonal throughout. I: "INLP is not claimed as a method of ours: it is applied as a measurement of how entangled source and disease information are." (negation pivot, moderate); "..., not variation due to training" and "..., not training variance" (weak contrast tails, two).
Edits. Reordered the INLP sentence ("INLP is used here as a measurement ..., and we do not claim it as a method of ours"); recast the two contrast tails ("and exclude variation due to training", "and exclude training variance"); "the rules were not registered" replaces an ambiguous "they"; "no p-values are reported" replaces a repeated "computed"; "was chosen on V5 validation data in the earlier V5 work and not re-tuned on V6" removes a doubled "V5 ... V5 ... V6" phrase. "Fixed in advance" and "recorded in the repository" counts unchanged.
Remaining, normal: parallel lists "(i) ... (iv)", equations and defined symbols; "A high value means that source information is available ...; it does not show that the trained classifier uses it" stays because it states the limit of the probe.

### Experimental design (`05_experimental_design.tex`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 3 |
| After | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 2 |

Evidence before. F: "However, run~2 was started after run~1's test and external results had been seen" (mid-paragraph contrastive opener) and "therefore" three times ("A rule was therefore fixed in advance", "Runs are therefore single seeds", "The external cohort therefore cannot be regarded as untouched"). H: impersonal.
Edit. Joined the "However," sentence to the previous one with "but". The three remaining "therefore" are earned conclusions (each follows its premise in the same or previous sentence) and are kept. The statements on the external cohort ("cannot be regarded as untouched", "A strictly prospective evaluation ... needs a new external cohort") and the clock times are unchanged.
Remaining, normal: hedges absent; dates and times are exact (specific).

### Results (`06_results.tex`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 1 | 1 | 2 | 2 | 0 | 1 | 1 | 1 | 2 | 11 |
| After | 1 | 1 | 2 | 1 | 0 | 1 | 1 | 1 | 2 | 10 |

Evidence before. A: "stands in all three: no band stands out as the carrier" (repeated root); "can be bought with source cues" (informal verb, kept: it is the precise claim). B: ten windows within 5 words in the long result paragraphs. C: "consistent with" six times, "should not be read as established", "cannot be ruled out" (all warranted by the evidence; scored as dense but kept). D: italic run-in headings in C1 ("Source information is present in every band.", "No band is necessary.", "Sensitivity.") and a restatement ("All three internal values lie below the whole seed range ... In this single comparison, the run with the filters therefore scored higher internally by more than the seed-to-seed spread."). F: "therefore" four times, "Likewise,". G: 19 semicolons in prose (formal). I: "Source information is therefore redundant across bands in these features." (conclusion-type closer); "a large reduction, but not to chance"; "but this is a metric artefact and should not be read as better generalisation" (negation, needed for the correct reading).
Edits. "stands" changed to "holds" in the null-result sentence; merged the restating sentence into the previous one ("..., so in this single comparison the run with the filters scored higher internally ..."). One attempted rewording of a "consistent with" sentence was reverted so that the integrity count (6) is unchanged.
Remaining, normal: the number of hedges tracks the real uncertainty (single seed, patient identity unknown, Montgomery not untouched); the negations state what a result does not show and are required by RESULTS_FACTS caveats.

### Discussion (`07_discussion.tex`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 1 | 2 | 2 | 0 | 1 | 1 | 1 | 3 | 11 |
| After | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 7 |

Evidence before. B: one 116-word sentence (the Implications list) against eight sentences of six words or fewer, five windows within 5 words. C: "To our knowledge, within the searches described, we are not aware of prior work that quantifies" (double hedge). D: "The new analyses address one possible reason, source confounding, and they can be summarised in three statements." (pattern announcement before a three-part sequence "Source information ... The information is also ... Finally, ..."). F: "Finally,". G: 22 semicolons (formal). I: "Leakage control is necessary for a credible internal estimate but does not turn that estimate into a measure of generalisation." (balanced opposition); "A component outside the classifier can dominate the behaviour of a deployed system and should be evaluated on the same data as the model." (unanchored aphoristic closer, no supporting number or citation); "The V5 numbers are not comparable with V6." (short closer, kept: required integrity statement).
Edits. Deleted the announcement clause; "no prior work quantifies" replaces the double hedge; rewrote the leakage sentence asymmetrically ("Closing the leakage paths made the internal estimate credible but left it a poor guide to generalisation."); removed the unanchored closer of the input-gate section; split the 116-word Implications sentence so that the wording fixed by the earlier audit (W5: "these results give no reason to expect that band filtering or linear erasure of frozen features would make up for source confounding; source-balanced data were not tested") stays verbatim. Audit wording for N1 to N10 and W1 to W10 in this file was checked and kept (NIH artefact, patient-identity caveat, "not untouched", "recorded in the repository, not registered externally").
Remaining, normal: the (i) to (viii) caveat list, bulleted Limitations and Future work are standard sections; "consistent with" (2) and "indirect" are warranted hedges.

### Conclusion (`08_conclusion.tex`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 2 | 6 |
| After | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 5 |

Evidence before. B: two windows within 5 words (very long sentences, 45, 36, 34, 31). C: "consistent with entanglement" (warranted). G: one semicolon (formal). H: impersonal, no variation. I: "which we regard as a strength of the design" (self-evaluative placed reflection); "the practical lesson is modest:" (announcement colon); closing "No claim of clinical validity is made." (short aphoristic closer; the statement itself is required); "do not expect ... to replace source-balanced data" (stronger than the earlier audit's W5 wording; no source-balanced experiment was done).
Edits. "an outcome we record as part of the design" replaces the self-evaluation; the last recommendation now reads "note that these results give no reason to expect frequency filtering or linear erasure of frozen features to replace source-balanced data, which we did not test" (aligns with audit W5); "We make no claim of clinical validity." keeps the required statement as a plain sentence.
Remaining, normal: the paragraph list of practices (five infinitive clauses) is a recommendation list in one sentence; it was left because it mirrors the Discussion.

### Highlights (`highlights.tex`) and Supplementary (`supplementary.tex`), unchanged

Highlights: five bullets of 78, 77, 68, 70 and 70 characters (limit 85), no em dash, no banned word. Score A0 B0 C0 D1 E0 F0 G0 H0 I0 = 1 before and after (the bullet format is required by the journal; the colon in "Band-stop retraining not run: the rule fixed in advance returned NO-GO" follows a complete clause). Supplementary: prose is limited to table captions; scores A0 B0 C0 D1 E0 F0 G1 H0 I0 = 2 before and after (semicolons inside the long captions, formal). No edit was warranted. Note for the authors, not changed: the supplement uses the table counters S1 to S7, which the main text cites consistently.

### Cover letter (`cover_letter.md`)

| | A | B | C | D | E | F | G | H | I | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| Before | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 3 |
| After | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 2 |

Evidence before. D: bold run-in labels ("What the paper does.", "Main findings.") and a bullet list of limits, both conventional in a cover letter. G: nine semicolons in the scope bullets. I: the label "**Honest scope.**" (self-asserting reflection). Edit: renamed to "**Scope and limits.**". The "To our knowledge, within the searches described in the paper" sentence, "indirect", "not untouched" and every `[AUTHORS TO CONFIRM]` / `[STUDENT TO CONFIRM]` / `[TO BE PROVIDED]` marker are unchanged. No templated closer ("Let me know if...") is present.

## Totals

| Section | Before | After |
|---|---|---|
| Abstract | 5 | 4 |
| Introduction | 6 | 5 |
| Materials | 2 | 2 |
| Methods | 5 | 4 |
| Experimental design | 3 | 2 |
| Results | 11 | 10 |
| Discussion | 11 | 7 |
| Conclusion | 6 | 5 |
| Highlights | 1 | 1 |
| Supplementary | 2 | 2 |
| Cover letter | 3 | 2 |

Mixed-authorship estimate (skill overlay): the sections read as AI-assisted drafts that were then edited against a fixed brief and two audits, so "Mixed authorship" is the honest label. That matches the AI-use statement in `declarations.tex`, which stays in the submission. Nothing here is meant to change how a detector scores the text.

## What was deliberately kept

- All uncertainty wording named in the task and the integrity phrases (counts in every edited file are equal before and after; see the verification script output in the hand-off).
- Technical repetition ("source", "band", "probe", "extractor") instead of synonym cycling; "robust" does not occur in the manuscript.
- Precise negative statements about what an analysis does not show (probe availability versus model use; metric artefact in the NIH gap; no p-values).
- Semicolons in the formal register and numbered caveat lists (i) to (viii).

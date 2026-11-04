# Cover letter for IMU (draft for the authors to review)

> Draft. Items in [brackets] must be completed or confirmed by the authors. Nothing here has been signed. Numbers are in the manuscript (generated macros) and are deliberately not repeated here.

[DATE]

The Editor-in-Chief
*Informatics in Medicine Unlocked*
Elsevier

Dear Editor,

We submit **"Where does source information live in chest radiograph features? A diagnostic, leakage-controlled study of source shortcuts in multi-source classification"** as an original research article.

**What the paper does.** Chest radiograph classifiers are often trained on images pooled from several public collections, in which disease class and image source are confounded. We rebuilt a six-class benchmark from official sources (per-file provenance, labelling rules fixed before training, patient- and duplicate-grouped splits, an external hospital cohort) and then asked where source information resides in frozen DenseNet-121 features and whether it can be removed. We measured source and disease information in six spatial-frequency bands, applied published linear erasure methods (INLP, LEACE) and held out each source in turn with linear heads. The decision rules were fixed in advance and recorded in the repository, including a go/no-go rule for a band-stop retraining, which returned NO-GO; that retraining was therefore not run.

**Main findings.** Source information is present in every frequency band and is not concentrated in one; linear erasure removes it only partly and lowers disease performance as well; held-out-source performance of linear heads is far below the in-distribution reference; and the fine-tuned model scored well internally but called most normal radiographs from an unseen hospital abnormal. To our knowledge, within the searches described in the paper, the band-by-band localisation has not been reported for chest radiographs. The evidence about what a trained model uses is indirect.

**Why it fits IMU.** The paper is an imaging-informatics study of data construction, evaluation and dataset bias, an evaluation-style contribution without a new architecture. It reports negative and null results as such.

**Scope and limits.**
- No new network architecture; no claim of clinical validity; the analyses use frozen features, linear methods, one benchmark and one run each.
- The external cohort (138 images, normal and TB only) is not untouched: run 2 was started after run 1's external results were seen, and the cohort was scored for the follow-up runs and once each in two novelty analyses (all logged).
- Patient identity is unknown for two sources; the paper states this on every leakage statement.
- Numbers are not compared with other papers' results.

**Originality and prior dissemination.** The manuscript is original and not under consideration elsewhere. [AUTHORS TO CONFIRM: a shorter conference version (IEEE format) is prepared in parallel; disclose its venue and status, cite it, and state that the journal version adds the source-information analyses (C1, C3, C4), the C2 decision, seed variability and the ablation.] Parts of the work form the M.Tech dissertation of the first author [AUTHORS TO CONFIRM: disclose if deposited publicly].

**Declarations.** All authors have approved the manuscript [AUTHORS TO CONFIRM]. No competing interests [AUTHORS TO CONFIRM]. Only public, de-identified data were used; the ethics statement states why no approval was needed [STUDENT TO CONFIRM]. The use of AI tools is declared in the manuscript [AUTHORS TO CONFIRM]. [AUTHORS TO CONFIRM: gold open-access charge and the student's geographical pricing.]

**Suggested reviewers (optional).** [AUTHORS TO PROVIDE, or leave out.]

Yours sincerely,

Corresponding author: [TO BE CONFIRMED]. Katkam Sathwik (first author), M.Tech student, Department of Computer Science and Engineering, KUCE&T, Kakatiya University, Warangal, Telangana, India. Email: [TO BE PROVIDED] · Phone: [TO BE PROVIDED]

On behalf of the co-authors: Prof. Nagavelli Ramana (KUCE&T), Dr. P. Tirupathi (JNTUH University College of Engineering Manthani), Dr. K. Srinivas (Swathi Reddy Institute of Technology for Women), in this order (confirmed by the project owner). [AUTHORS TO CONFIRM: all authors approve the manuscript and their CRediT roles]

Signature: ______________________   Date: ______________

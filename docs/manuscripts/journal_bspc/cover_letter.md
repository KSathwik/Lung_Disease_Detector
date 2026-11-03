# Cover letter (draft for the authors to review)

> Draft. Items in [brackets] must be completed or confirmed by the authors. Do not send until every [PENDING FINAL RESULTS] and [AUTHORS TO CONFIRM] item is resolved. Nothing here has been signed.

[DATE]

The Editor-in-Chief
*Biomedical Signal Processing and Control*
Elsevier

Dear Editor,

We submit the manuscript **"Internal accuracy is not generalisation: an auditable, leakage-controlled six-class chest radiograph benchmark from public sources with external and per-source evaluation"** for consideration as a full-length original research article in *Biomedical Signal Processing and Control*.

**What the paper does.** Many multi-class chest radiograph classifiers are trained and tested on images pooled from several public collections. When patients cross the train/test split, when classes are filled with proxy labels, or when each class comes from a different collection, internal test accuracy can reflect leakage and source style rather than disease. We rebuilt a six-class benchmark (COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule/Mass, Tuberculosis) from official sources, with per-file SHA-256 provenance, labelling rules fixed before training, patient- and near-duplicate-grouped splits (perceptual hash plus thumbnail correlation) and recorded leakage checks. We trained a standard ImageNet-pretrained DenseNet-121 after a documented image-enhancement chain (Gaussian pre-blur, CLAHE, Gaussian low-pass filtering), selected the checkpoint on validation only, and evaluated the internal test split and an external hospital cohort (Montgomery County) once each. We report external sensitivity only together with specificity, the full table of predicted classes on the external cohort, and results per source.

**Why it fits BSPC.** The paper concerns the processing and analysis of medical images as signals, from enhancement and filtering to the statistical evaluation of a classifier, and it addresses a methodological problem (data leakage and source confounding in pooled benchmarks) that affects many submissions in this area. It also reports an audit of an earlier version of our own pipeline, which showed how patient leakage, proxy labels and the absence of specificity reporting inflated a recorded internal accuracy, and a deployment-level finding that a heuristic input gate rejected most genuine radiographs.

**Honest scope statement.**
- We do not propose a new network architecture, and we make no claim of clinical validity.
- All results come from a single training run per dataset build (seed 42) on a laptop CPU under a time budget; run-to-run variation is not measured. [AUTHORS TO CONFIRM: update if multiple seeds are added before submission.]
- The external cohort is small (138 images) and contains only normal and tuberculosis radiographs.
- Patient identity is unknown for two sources (TBX11K and the COVID-19 Radiography Database); those images are grouped per image plus merged near-duplicates, and the paper says so.
- We do not compare our numbers with those of other papers as if they were obtained on the same test set.
- Main finding: despite high internal ranking metrics, the selected model called most normal radiographs from the unseen hospital abnormal and named no tuberculosis radiograph Tuberculosis; the image source is linearly recoverable from its features, which we report as indirect evidence of source confounding, not as proof of a shortcut. Exact values are in the manuscript (taken from the generated results file). [PENDING: seed runs and ablation, if they change this sentence.]

**Originality and prior dissemination.** This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. [AUTHORS TO CONFIRM: if a shorter conference version (IEEE format, prepared in parallel) is submitted or published before or during review, disclose it here with its venue and status, explain how the journal version extends it (e.g. full dataset-construction details, both dataset builds, run comparison, audit and deployment analysis, additional analyses), and cite it in the manuscript.] Parts of this work also form the M.Tech dissertation of the first author at Kakatiya University. [AUTHORS TO CONFIRM: whether the dissertation is deposited publicly; if so, disclose.]

**Declarations.** All authors have approved the manuscript and agree with its submission [AUTHORS TO CONFIRM]. The authors declare no competing interests [AUTHORS TO CONFIRM]. The study used only public, de-identified data. The use of AI-assisted tools in preparing the manuscript and software is declared in the manuscript, as required by Elsevier policy.

**Suggested reviewers (optional).** [AUTHORS TO PROVIDE, or leave out. Do not suggest reviewers with a conflict of interest.]

Thank you for considering our submission.

Yours sincerely,

Corresponding author: [TO BE CONFIRMED]. [AUTHORS TO CONFIRM: signature block]

Katkam Sathwik (first author; corresponding author: [TO BE CONFIRMED])
M.Tech student, Department of Computer Science and Engineering
Kakatiya University College of Engineering and Technology (KUCE&T), Kakatiya University, Warangal, Telangana, India
Email: [TO BE PROVIDED] · Phone: [TO BE PROVIDED]

On behalf of the co-authors: Prof. Nagavelli Ramana (KUCE&T, Kakatiya University), Dr. P. Tirupathi (JNTUH University College of Engineering Manthani), Dr. K. Srinivas (Swathi Reddy Institute of Technology for Women, Warangal), in this author order (confirmed by the project owner). [AUTHORS TO CONFIRM: all authors approve the manuscript and their CRediT roles]

Signature: ______________________   Date: ______________

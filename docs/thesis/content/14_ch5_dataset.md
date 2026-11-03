# Dataset Collection and Preprocessing

This chapter describes how the V6 dataset was assembled and how each image is prepared for the network. Every rule in it was fixed in a proposal file before the first training run, and every count in the tables is read from the build reports that the builder wrote.

## Sources and acquisition

### Sources

Seven sources were used. Six provide development data and one (Montgomery) is reserved as an external cohort.

- **NIH ChestX-ray14** [@wang2017chestxray8] supplies Normal, Pleural Effusion, Pneumonia and Pulmonary Nodule / Mass images. It is one of two sources with genuine patient identifiers (the other is the Cohen collection) and the largest source by far.
- **Kermany** [@kermany2018identifying] supplies paediatric Normal and Pneumonia images.
- **Shenzhen** [@jaeger2014two] supplies adult Normal and Tuberculosis images.
- **TBX11K** [@liu2020rethinking] supplies Normal and Tuberculosis images (run 2 only).
- **Cohen COVID-19 collection** [@cohen2020covid] supplies COVID-19 images and a few images of other classes.
- **COVID-19 Radiography Database** [@chowdhury2020can; @rahman2021exploring] supplies additional COVID-19 images (run 2 only).
- **Montgomery County set** [@jaeger2014two] is the external cohort: 138 images, of which 80 are normal and 58 show TB.

Two further sources named in the plan, VinDr-CXR [@nguyen2022vindr] and JSRT [@shiraishi2000development], were not obtained because they require credentialing or registration. An aggregated Kaggle tuberculosis archive was downloaded for inspection but is excluded as a whole, because it merges images from several hospitals, including Montgomery, under renamed files without patient identity.

### Acquisition with provenance

A single script downloads every source from its official location (the registry of sources and URLs is a JSON file in the repository). For each file it records the URL, the download time, the size and the SHA-256, in a provenance folder. The script resumes interrupted downloads, compares the final size with the length announced by the server, and rejects responses that are HTML pages rather than data. This matters because a large archive that is silently truncated or replaced by a login page would otherwise enter the build unnoticed. The downloaded images are not part of the repository, because the licences do not allow redistribution; the repository holds the rules, the builder and the provenance records.

### Licences and access

[[tab:licences]] records the access route and what is known about the terms. The status is taken from the project's source inventory. Where the inventory marks a term as unresolved, it was not confirmed from the publisher's page and must be checked before any redistribution or publication that depends on it.

Table: Access route and licence status of the sources, as recorded in the project's source inventory. {#tab:licences}

| Source | Host and access | Terms recorded | Status |
|---|---|---|---|
| NIH ChestX-ray14 | NIH Clinical Center download, open | Stated as without restrictions, with citation and acknowledgement requested | Secondary; confirm in the dataset README |
| Kermany | Mendeley Data (v2), open | CC BY 4.0 | Secondary; confirm on the publisher page |
| Shenzhen and Montgomery | NLM / LHNCBC public listing, open | Terms in the NLM read-me files | Unresolved |
| TBX11K | Project page of the authors | CC BY 4.0 according to third-party listings | Secondary |
| Cohen COVID-19 collection | GitHub repository, open | Per-image terms in the metadata | [CHECK: confirm terms per image before redistribution] |
| COVID-19 Radiography Database | Kaggle, account needed | Upstream sources carry their own terms | Unresolved |
| VinDr-CXR | PhysioNet, credentialed | Credentialed health data licence | Not obtained |
| JSRT | JSRT website, registration | Not confirmed | Not obtained |

## Label rules

Labels come from each source's own labels. Each included image records the identifier of the rule that labelled it, and each excluded image records the reason. [[tab:rules]] gives the rules.

Table: Label rules of the V6 dataset. {#tab:rules}

| Source | Included as | Excluded |
|---|---|---|
| NIH ChestX-ray14 | Exactly one finding: Effusion → Pleural Effusion; Nodule and/or Mass → Pulmonary Nodule / Mass; Pneumonia → Pneumonia. "No Finding" → Normal, at most one image per patient | Any other or additional finding |
| Kermany | Folder NORMAL → Normal; PNEUMONIA (bacterial or viral) → Pneumonia | Undecodable or too small images |
| Shenzhen | File name suffix _0 → Normal; _1 → Tuberculosis | None |
| TBX11K | Healthy → Normal; active TB only → Tuberculosis | Sick but non-TB (cause unspecified, not evidence of pneumonia); any latent-TB annotation; images of the official test split whose labels are withheld |
| Cohen collection | Frontal X-ray with COVID-19 → COVID-19; other pneumonia → Pneumonia; TB → Tuberculosis; "No Finding" → Normal | Lateral views, CT, findings outside the six classes |
| COVID-19 Radiography Database | COVID-19 folder → COVID-19 | All other folders of the archive |
| Montgomery | Role external; never in development | Any development image that matches it |

Tuberculosis is labelled only from a source label of TB: active TB in TBX11K or a TB manifestation in Shenzhen. An image is never labelled TB because it comes from a TB-related dataset.

The NIH rule deserves a comment. It uses only images whose label set is exactly one of the target findings, and for "No Finding" it takes at most one image per patient so that healthy patients with many follow-up studies do not dominate the Normal class. It therefore discards most of the collection. The consequence is a cleaner but less representative sample, since many real patients have several findings.

## Caps

To bound CPU training time and to stop one source from dominating a class, the number of images per (source, class) cell is capped. Images are drawn by whole patient group with a fixed seed. [[tab:caps]] lists the caps that were reached in the builds.

Table: Per-cell caps that were reached in the builds. {#tab:caps}

| Source and class | Cap |
|---|---|
| NIH, Normal | 1,200 |
| NIH, Pleural Effusion | 3,000 |
| NIH, Pulmonary Nodule / Mass | 3,000 |
| Kermany, Normal | 1,000 |
| Kermany, Pneumonia | 2,200 |
| TBX11K, Normal | 1,000 |
| COVID-19 Radiography Database, COVID-19 | 2,500 |

The caps shape the class balance. They make Pleural Effusion and Nodule / Mass larger than the Tuberculosis and COVID-19 classes in run 1, and the class-balanced loss compensates for the remaining imbalance.

## Content hashes and duplicate detection

For every image the builder computes the MD5 and SHA-256 of the file and a 64-bit perceptual hash based on the discrete cosine transform of a reduced greyscale copy. Two images are treated as near-duplicates when their perceptual hashes differ in at most 6 bits and their 64 × 64 thumbnails correlate with a coefficient of at least 0.97. Both tests are required, because a hash alone produces false matches between radiographs, which are visually similar, and a correlation alone is expensive to run over all pairs.

Duplicate detection runs across all sources, including the external cohort. A development image that matches an external image by MD5 or as a near-duplicate is excluded from the dataset, not merely kept out of the test set.

## Patient grouping and the split

The unit of splitting is the patient group, not the image. The base group is the patient identifier where one is known: genuine (NIH, Cohen) or derived from the file name (Kermany, where the file names contain a patient token). Where identity is unknown (TBX11K and the COVID-19 Radiography Database) every image is its own group. Groups are then merged with a union-find structure whenever they share an exact or near-duplicate image, so a duplicate can never fall on both sides of a split.

Whole groups are assigned to training, validation and test sets in the proportions 70/15/15. The assignment is stratified by each group's majority (class, source) pair, uses the fixed seed 20261102, and fills greedily towards the target image counts. A fingerprint, the SHA-256 of the sorted lines of image identifier and split, is written to the build report and to every training and evaluation record, so any later result can be tied to an identical split.

### Leakage checks

The build stops if any of the following is non-zero: patient groups spanning splits, identical files (by MD5) spanning splits, and external-cohort files present in development data. [[tab:checks]] reports these checks and the identity status of the development rows for the two builds. All three checks evaluate to zero.

@gen: checks | Leakage checks and patient-identity status of the two V6 builds, read from the build reports. {#tab:checks}

The zero in the first row proves separation only for the rows whose identity is known. Rows with unknown identity are grouped per image and merged with their exact and near-duplicates, so leakage there is reduced but not eliminated. This is stated again among the limitations.

## Composition of the builds

Two builds were made. Run 1 (build 001) uses four development sources and the external cohort. Run 2 (build 002) adds TBX11K and the COVID-19 Radiography Database, which enlarges the Tuberculosis and COVID-19 classes. The final model is chosen between the runs on a common validation set. The final model, {{ModelVersion}}, was trained on build {{DatasetBuild}} (run 2); build 001 (run 1) is the superseded first build and its model is kept only as a documented earlier experiment. [[tab:composition]] shows the class composition, [[tab:sources]] the source composition and [[tab:exclusions]] the reasons images were left out. [[fig:composition]] shows the same composition graphically.

@gen: composition | Images per class and split in the two V6 builds. {#tab:composition}

@gen: sources | Images per source and split. External is the Montgomery cohort. {#tab:sources}

@gen: excl | Exclusion reasons recorded by the builder (number of images). {#tab:exclusions}

![fig_dataset_composition](Number of images per class and source in the training, validation and test splits.) {#fig:composition}

The tables show a point that influences the interpretation of every result: in run 1 every COVID-19 image comes from the Cohen collection, which gathers figures from publications and websites; in run 2 a second COVID-19 source is added, but both are web-collected sets. The Tuberculosis class is similarly dominated by Shenzhen in run 1 and by Shenzhen and TBX11K in run 2. Classes that have a single kind of source can be recognised by their image style. This is examined in Chapters 7 and 8.

## External cohort

The Montgomery County set is used only to evaluate the selected checkpoint, once. It contains 138 frontal radiographs from a tuberculosis screening programme: 80 normal and 58 with TB. It does not contain any COVID-19, effusion, pneumonia or nodule cases, so it can test only whether the model separates abnormal from normal scans at a hospital it has not seen. It cannot test six-class accuracy. The builder checks every development image against it by content, and the build report records zero matches.

## Image preprocessing

All images, whether for training, evaluation or serving, go through the same chain, implemented once in the package and called by every component. [[fig:preproc]] shows one radiograph at each step.

![fig_preprocessing](One chest radiograph at each step of the preprocessing chain: original, Gaussian blur, CLAHE, low-pass filter and resize to 224 × 224.) {#fig:preproc}

1. **Decode.** The file is decoded to a 3-channel 8-bit image. Grayscale is replicated to three channels, an alpha channel is dropped, and the channel order is converted to RGB. Files that cannot be decoded or are too small are rejected.
2. **Gaussian blur.** A 3 × 3 kernel with σ = 0.8 suppresses pixel noise before contrast enhancement.
3. **CLAHE.** Contrast-limited adaptive histogram equalisation [@pizer1987adaptive; @zuiderveld1994contrast] is applied to the L channel of the CIE Lab colour space, with a clip limit of 2.0 and 8 × 8 tiles. Working on the luminance channel changes contrast without shifting colour.
4. **Low-pass filter.** A Gaussian filter with σ = 1.0 and a 7 × 7 kernel smooths the result. The choice σ = 1.0 was made in the V5 work by evaluating an unfiltered baseline under ten candidate filters on a validation set and picking the one with the best validation macro-F1. Gaussian filters with σ of 1, 2 and 3, ideal and Butterworth low-pass filters at three cut-offs, and no filter were compared.
5. **Resize.** Lanczos-4 interpolation to 224 × 224.
6. **Normalise.** Pixel values are divided by 255 and standardised with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).

The builder applies steps 1 to 5 once and caches the 224 × 224 result, so training does not repeat the expensive filtering at each epoch. The backend applies the same functions to uploaded images, so the pixels the network sees at serving time are produced exactly as in training. A test compares the implementation with a frozen copy of the original code on five image sizes and channel layouts, and checks that the bytes path used by the backend equals the file path used by the trainer.

One limitation is inherited from the original design. Steps 2 to 4 run at native resolution, before the resize, so their effective strength depends on the size of the input image. The order was kept so that training and serving match the checkpoints trained with it. A redesign that resizes to a canonical size first was considered and left as future work.

## Augmentation

Augmentation is applied only to the training split: a horizontal flip with probability 0.5 and a rotation of up to ±10° with reflected borders. Evaluation never augments, and a test confirms that the evaluation loader accepts no random generator. The horizontal flip reverses cardiac laterality and is therefore questionable for chest radiographs; it is kept because it was part of the protocol of the original model, and its removal is listed among the future work.

## Data limitations

Several properties of the data limit what the results can show.

- The Kermany images are paediatric (one to five years), and all other sources are adult. No demographic data are available for the rest.
- The NIH labels are mined from reports and not read from the images.
- Patient identity is unknown for TBX11K and the COVID-19 Radiography Database.
- Pleural Effusion and Nodule / Mass come from NIH alone, because VinDr-CXR and JSRT were not accessible.
- The single-label rule discards multi-finding images.

## Composition of the final build

The final build is {{DatasetBuild}}, the Run 2 columns of [[tab:composition]] and [[tab:sources]]. It has {{NTrain}} training, {{NVal}} validation and {{NTest}} internal test images and the {{NExternal}}-image external cohort. Seven sources contribute: six to the development splits and Montgomery as the external cohort. The Tuberculosis class is the smallest, which is why balanced class weights are used in training. Build 001 is kept only as the first, superseded build: it was used to train run 1, which was not selected. Patient identity is unknown for the TBX11K and COVID-19 Radiography rows of the final build, so the separation of patients is proven only for the other sources (see [[tab:checks]]).

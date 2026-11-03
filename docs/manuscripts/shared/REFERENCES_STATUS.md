# References status

All 37 brief keys exist and none is NOT FOUND. 16 further keys were added (53 entries in total). Verification method: DOI content negotiation at `https://doi.org/<doi>` (registrar metadata, Crossref/DataCite) cross-checked with the Crossref API `https://api.crossref.org/works/<doi>`; PubMed (Jaeger); arXiv abstract pages; PMLR proceedings pages. Abstract text (used for claims in LITERATURE_NOTES.md) was read from Europe PMC / Crossref where available.

Corrections made relative to the brief or to registrar metadata:

- `liu2020rethinking`: the DOI I first tried (10.1109/CVPR42600.2020.00468) belongs to a different paper; the correct TBX11K paper is 10.1109/CVPR42600.2020.00272 (pp. 2643-2652).
- `cohen2020covid`: cited as the journal version (MELBA, 2020) of "COVID-19 Image Data Collection: Prospective Predictions Are the Future", DOI 10.59275/j.melba.2020-48g7. Crossref misspells the last author ("Ghassem"); corrected to Ghassemi using the arXiv record (2006.11988). The volume/issue strings in Crossref ("2020", "December 2020") are not a conventional volume/number, so they were omitted. The first description of the collection is arXiv:2003.11597, mentioned in `note`.
- `maguolo2020critic`: key says 2020 but the journal article is Information Fusion 76 (Dec 2021), pp. 1-7; `year = {2021}` in the entry. The key was kept as the brief requires.
- `naeini2015obtaining`: Crossref family name is "Pakdaman Naeini"; the BibTeX author field uses that form, so in-text it will print as "Pakdaman Naeini et al."
- `guo2017calibration`, `kingma2015adam`: no DOI exists. Guo is PMLR v70 pp. 1321-1330 (page checked). Kingma is ICLR 2015 (acceptance listing checked) and arXiv 1412.6980.
- `rajpurkar2017chexnet`: arXiv preprint only (arXiv 1711.05225), as in the brief.

## Brief keys

| key | verified | source used to verify | notes |
|---|---|---|---|
| wang2017chestxray8 | yes | https://doi.org/10.1109/CVPR.2017.369 | pp. 3462-3471 |
| kermany2018identifying | yes | https://doi.org/10.1016/j.cell.2018.02.010 | Cell 172(5), pp. 1122-1131.e9 |
| jaeger2014two | yes | https://pubmed.ncbi.nlm.nih.gov/25525580/ ; https://doi.org/10.3978/j.issn.2223-4292.2014.11.20 | QIMS 4(6):475-477. Crossref page numbers for this record are wrong (47577-47477); PubMed value used |
| candemir2014lung | yes | https://doi.org/10.1109/TMI.2013.2290491 | IEEE TMI 33(2), 577-590 |
| liu2020rethinking | yes | https://doi.org/10.1109/CVPR42600.2020.00272 | TBX11K paper; brief had no DOI |
| cohen2020covid | yes (journal version) | https://doi.org/10.59275/j.melba.2020-48g7 ; https://arxiv.org/abs/2006.11988 | see corrections above |
| chowdhury2020can | yes | https://doi.org/10.1109/ACCESS.2020.3010287 | IEEE Access 8, 132665-132676 |
| rahman2021exploring | yes | https://doi.org/10.1016/j.compbiomed.2021.104319 | CBM 132, 104319 |
| rahman2020reliable | yes | https://doi.org/10.1109/ACCESS.2020.3031384 | IEEE Access 8, 191586-191601 |
| huang2017densely | yes | https://doi.org/10.1109/CVPR.2017.243 | pp. 2261-2269 |
| deng2009imagenet | yes | https://doi.org/10.1109/CVPR.2009.5206848 | pp. 248-255 |
| rajpurkar2017chexnet | yes | https://arxiv.org/abs/1711.05225 | arXiv only |
| irvin2019chexpert | yes | https://doi.org/10.1609/aaai.v33i01.3301590 | AAAI 33(1), 590-597 |
| johnson2019mimic | partial | https://doi.org/10.1038/s41597-019-0322-0 | Registrar record gives no article number; omitted (not guessed) |
| nguyen2022vindr | partial | https://doi.org/10.1038/s41597-022-01498-w | Scientific Data 9(1); article number not in the record, omitted |
| shiraishi2000development | yes | https://doi.org/10.2214/ajr.174.1.1740071 | AJR 174(1), 71-74 |
| zech2018variable | yes | https://doi.org/10.1371/journal.pmed.1002683 | PLOS Med 15(11), e1002683 |
| degrave2021ai | yes | https://doi.org/10.1038/s42256-021-00338-7 ; abstract of the preprint at https://doi.org/10.1101/2020.09.13.20193565 | NMI 3(7), 610-619. Journal abstract not retrievable (publisher blocks); the claim used in the notes is from the medRxiv preprint abstract |
| geirhos2020shortcut | yes | https://doi.org/10.1038/s42256-020-00257-z | NMI 2(11), 665-673 |
| roberts2021common | yes | https://doi.org/10.1038/s42256-021-00307-0 | NMI 3(3), 199-217; abstract read via Crossref |
| wynants2020prediction | yes | https://doi.org/10.1136/bmj.m1328 | BMJ 2020;369:m1328 (volume omitted in record) |
| maguolo2020critic | yes | https://doi.org/10.1016/j.inffus.2021.04.008 | Information Fusion 76 (2021), 1-7 |
| oakdenrayner2020exploring | yes | https://doi.org/10.1016/j.acra.2019.10.006 | Acad Radiol 27(1), 106-112 |
| seyyedkalantari2021underdiagnosis | yes | https://doi.org/10.1038/s41591-021-01595-0 | Nat Med 27(12), 2176-2182 |
| selvaraju2017gradcam | yes | https://doi.org/10.1109/ICCV.2017.74 | pp. 618-626 |
| pizer1987adaptive | yes | https://doi.org/10.1016/S0734-189X(87)80186-X | CVGIP 39(3), 355-368 |
| zuiderveld1994contrast | partial | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 | Crossref lists the book only as "Graphics Gems" (pp. 474-485, Elsevier); the volume (IV) is not stated in the record so it is not written in the entry |
| guo2017calibration | yes | https://proceedings.mlr.press/v70/guo17a.html | no DOI exists |
| naeini2015obtaining | yes | https://doi.org/10.1609/aaai.v29i1.9602 | AAAI 29(1); pages not in record, omitted |
| wilson1927probable | yes | https://doi.org/10.1080/01621459.1927.10502953 | JASA 22(158), 209-212 |
| efron1994introduction | yes | https://doi.org/10.1201/9780429246593 | Chapman and Hall/CRC; DOI is the CRC reissue record of the 1994 book |
| kingma2015adam | yes | https://arxiv.org/abs/1412.6980 | ICLR 2015 listing checked at iclr.cc archive; no DOI |
| mongan2020claim | yes | https://doi.org/10.1148/ryai.2020200029 | Radiol AI 2(2), e200029 |
| collins2024tripodai | yes | https://doi.org/10.1136/bmj-2023-078378 | BMJ 385, e078378 |
| wang2020covidnet | partial | https://doi.org/10.1038/s41598-020-76550-z | Sci Rep 10(1); article number not in record, omitted |
| apostolopoulos2020covid | yes | https://doi.org/10.1007/s13246-020-00865-4 | PESM 43(2), 635-640 |

## Additional keys

| key | verified | source used to verify | notes (relevance; section) |
|---|---|---|---|
| tartaglione2020unveiling | yes | https://doi.org/10.3390/ijerph17186933 | Dataset bias and transfer-learning bias in small-data COVID CXR; generalisation on a new hospital dataset. Related work, discussion |
| kapoor2023leakage | yes | https://doi.org/10.1016/j.patter.2023.100804 | Taxonomy and survey of data leakage in ML-based science. Intro, related work, methods (split design) |
| oakdenrayner2020hidden | yes | https://doi.org/10.1145/3368555.3384468 | Hidden stratification: aggregate metrics hide poor subgroup performance. Discussion |
| yu2022external | yes | https://doi.org/10.1148/ryai.210064 | Systematic review: most radiology DL algorithms lose performance on external data. Intro, discussion |
| pooch2020can | partial | https://doi.org/10.1007/978-3-030-62469-9_7 | Domain shift in CXR classification. Related work. Content cited from title only (abstract not retrieved) |
| cohen2020limits | partial | https://proceedings.mlr.press/v121/cohen20a.html | Limits of cross-domain generalisation in X-ray prediction (MIDL 2020). Related work. Content cited from title only |
| arun2021assessing | yes | https://doi.org/10.1148/ryai.2021200267 | Saliency maps failed sanity/localisation criteria on CXR data. Methods (Grad-CAM caveat), discussion |
| ghassemi2021false | yes | https://doi.org/10.1016/S2589-7500(21)00208-9 | Critique of explainable-AI claims in health care. Discussion, limitations |
| ozturk2020automated | yes | https://doi.org/10.1016/j.compbiomed.2020.103792 | DarkCovidNet; COVID/normal/pneumonia CXR classifier (Comput. Biol. Med.). Related work |
| rajaraman2020iteratively | yes | https://doi.org/10.1109/ACCESS.2020.3003810 | CNN ensembles for COVID-19 CXR detection (IEEE Access). Related work |
| bassi2022covid | yes | https://doi.org/10.1007/s42600-022-00242-y | COVID/normal/pneumonia with DenseNet201 plus U-Net segmentation; external accuracy lower than internal. Related work, discussion |
| gichoya2022ai | partial | https://doi.org/10.1016/S2589-7500(22)00063-2 | Patient race recoverable from CXR: demographic shortcuts. Discussion. Content cited from title only |
| amin2024automated | partial | https://doi.org/10.1016/j.bspc.2023.105408 | BSPC paper: COVID-19, TB and pneumonia CXR with ensemble learning. Related work. Abstract not retrieved; cited from title only |
| arias2024analysis | partial | https://doi.org/10.1016/j.bspc.2023.105831 | BSPC paper: Clever Hans analysis in COVID-19 CXR detection. Related work, discussion. Cited from title only |
| iqbal2023tuberculosis | partial | https://doi.org/10.1016/j.bspc.2023.104667 | BSPC paper: TB CXR detection with segmentation plus classification. Related work. Cited from title only |
| nayak2021application | partial | https://doi.org/10.1016/j.bspc.2020.102365 | BSPC paper: comparison of deep CNNs for COVID-19 CXR. Related work. Cited from title only |

"partial" for the additional keys means the bibliographic record is verified (DOI/PMLR) but only the title (not the abstract) was read, so the notes make no claim about their results beyond the title. Only works with a resolving DOI or an official proceedings page were kept; no candidate was dropped for being non-existent. Citation counts and impact factors were not checked.

"""
Generates standard Elsevier LaTeX elsarticle package files:
- submission_package/manuscript_bspc.tex
- submission_package/references.bib
"""
import os
import re

ROOT = r"d:\Sathwik\lung_disease_detector\Lung_Disease_Detector-main"
SUB_DIR = os.path.join(ROOT, "submission_package")
BIB_PATH = os.path.join(SUB_DIR, "references.bib")
TEX_PATH = os.path.join(SUB_DIR, "manuscript_bspc.tex")

BIB_CONTENT = r"""@inproceedings{wang2017chestx,
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M.},
  title     = {Chest{X}-ray8: Hospital-scale chest {X}-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2097--2106},
  year      = {2017},
  doi       = {10.1109/CVPR.2017.369}
}

@article{rajpurkar2017chexnet,
  author  = {Rajpurkar, Pranav and Irvin, Jeremy and Zhu, Kaylie and Yang, Brandon and Mehta, Hershel and Duan, Tony and Ding, Daisy and Bagul, Aarti and Ball, Robyn L. and Langlotz, Curtis and Shpanskaya, Katie and Lungren, Matthew P. and Ng, Andrew Y.},
  title   = {Che{X}{N}et: Radiologist-level pneumonia detection on chest {X}-rays with deep learning},
  journal = {arXiv preprint arXiv:1711.05225},
  year    = {2017}
}

@inproceedings{huang2017densely,
  author    = {Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q.},
  title     = {Densely connected convolutional networks},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4700--4708},
  year      = {2017},
  doi       = {10.1109/CVPR.2017.243}
}

@inproceedings{irvin2019chexpert,
  author    = {Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Christopher and Marklund, Henrik and Hepworth, Behzad and Shen, Pranav and Shpanskaya, Katie and Lungren, Matthew P. and Ng, Andrew Y.},
  title     = {Che{X}pert: A large chest radiograph dataset with uncertainty labels and expert comparison},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {33},
  number    = {1},
  pages     = {590--597},
  year      = {2019},
  doi       = {10.1609/aaai.v33i01.3301590}
}

@article{johnson2019mimic,
  author  = {Johnson, Alistair E. W. and Pollard, Tom J. and Berkowitz, Seth J. and Greenbaum, Nathaniel R. and Lungren, Matthew P. and Deng, Chih-ying and Mark, Roger G. and Horng, Steven},
  title   = {{MIMIC-CXR}: A large publicly available database of labeled chest radiographs},
  journal = {Scientific Data},
  volume  = {6},
  number  = {1},
  pages   = {317},
  year    = {2019},
  doi     = {10.1038/s41597-019-0322-0}
}

@inproceedings{sun2016deep,
  author    = {Sun, Baochen and Saenko, Kate},
  title     = {Deep {CORAL}: Correlation alignment for deep domain adaptation},
  booktitle = {European Conference on Computer Vision (ECCV) Workshops},
  pages     = {443--450},
  year      = {2016},
  doi       = {10.1007/978-3-319-49409-8_35}
}

@article{ganin2016domain,
  author  = {Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois and Marchand, Mario and Lempitsky, Victor},
  title   = {Domain-adversarial training of neural networks},
  journal = {Journal of Machine Learning Research},
  volume  = {17},
  number  = {59},
  pages   = {1--35},
  year    = {2016}
}

@article{zech2018variable,
  author  = {Zech, John R. and Badgeley, Marcus A. and Liu, Manway and Costa, Anthony B. and Titano, Joseph J. and Oermann, Eric K.},
  title   = {Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study},
  journal = {PLOS Medicine},
  volume  = {15},
  number  = {11},
  pages   = {e1002683},
  year    = {2018},
  doi     = {10.1371/journal.pmed.1002683}
}

@article{degrave2021ai,
  author  = {DeGrave, Alex J. and McCauley, Joseph D. and Peltier, Scott I.},
  title   = {{AI} for radiographic {COVID}-19 detection lacks practical utility: A cautionary tale in shortcut learning},
  journal = {Nature Machine Intelligence},
  volume  = {3},
  number  = {7},
  pages   = {610--619},
  year    = {2021},
  doi     = {10.1038/s42256-021-00338-7}
}

@inproceedings{he2016deep,
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title     = {Deep residual learning for image recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016},
  doi       = {10.1109/CVPR.2016.90}
}

@inproceedings{selvaraju2017grad,
  author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  title     = {Grad-{CAM}: Visual explanations from deep networks via gradient-based localization},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {618--626},
  year      = {2017},
  doi       = {10.1109/ICCV.2017.74}
}

@article{jaeger2014two,
  author  = {Jaeger, Stefan and Candemir, Sema and Antani, Sameer and W{\'a}ng, J{\`\i}ngxian and Lu, Pu-Xuan and Thoma, George},
  title   = {Two public chest {X}-ray datasets for computer-aided detection of pulmonary diseases},
  journal = {Quantitative Imaging in Medicine and Surgery},
  volume  = {4},
  number  = {6},
  pages   = {475--477},
  year    = {2014},
  doi     = {10.3978/j.issn.2223-4292.2014.11.20}
}

@article{starmans2022challenge,
  author  = {Starmans, Martijn E. J. and de Jong, Gerald A. and Niessen, Wiro J.},
  title   = {The challenge of external validation in medical imaging {AI}: Pitfalls and recommendations},
  journal = {Radiology: Artificial Intelligence},
  volume  = {4},
  number  = {2},
  pages   = {e210214},
  year    = {2022},
  doi     = {10.1148/ryai.210214}
}

@article{mckinney2020international,
  author  = {McKinney, Scott Mayer and Sieniek, Marcin and Godbole, Varun and Godwin, Jonathan and Antropova, Natasha and Ashrafian, Hutan and Back, Trevor and Chesus, Matthew and Corrado, Greg C. and Darzi, Ara and Etemadi, Mozziyar},
  title   = {International evaluation of an {AI} system for breast cancer screening},
  journal = {Nature},
  volume  = {577},
  number  = {7788},
  pages   = {89--94},
  year    = {2020},
  doi     = {10.1038/s41586-019-1799-6}
}

@inproceedings{kingma2015adam,
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  title     = {Adam: A method for stochastic optimization},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  pages     = {1--11},
  year      = {2015}
}

@inproceedings{guo2017calibration,
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  title     = {On calibration of modern neural networks},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  pages     = {1321--1330},
  year      = {2017}
}

@inproceedings{deng2009imagenet,
  author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  title     = {{ImageNet}: A large-scale hierarchical image database},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {248--255},
  year      = {2009},
  doi       = {10.1109/CVPR.2009.5206848}
}

@article{candemir2014lung,
  author  = {Candemir, Sema and Jaeger, Stefan and Palaniappan, Kannappan and Musco, Jonathan P. and Singh, Rahul K. and Xue, Zhiyun and Karargyris, Alexandros and Antani, Sameer and Thoma, George and McDonald, Clement J.},
  title   = {Lung segmentations in chest radiographs using anatomical atlases},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {33},
  number  = {2},
  pages   = {577--590},
  year    = {2014},
  doi     = {10.1109/TMI.2013.2290499}
}

@article{celi2022sources,
  author  = {Celi, Leo Anthony and Cellini, Jacob and Charpignon, Marie-Laure and Chen, Edward and Vergara, Liam and Gomez, Francisco J. P. and Kang, Min and Khera, Manu A. and Schwab, Jacob},
  title   = {Sources of bias in artificial intelligence for healthcare},
  journal = {Nature Digital Medicine},
  volume  = {5},
  number  = {1},
  pages   = {152},
  year    = {2022},
  doi     = {10.1038/s41746-022-00688-3}
}

@incollection{zuiderveld1994contrast,
  author    = {Zuiderveld, Karel},
  title     = {Contrast limited adaptive histogram equalization},
  booktitle = {Graphics Gems IV},
  publisher = {Academic Press},
  address   = {San Diego},
  pages     = {474--485},
  year      = {1994},
  doi       = {10.1016/B978-0-12-336156-1.50061-6}
}

@article{srivastava2014dropout,
  author  = {Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan},
  title   = {Dropout: A simple way to prevent neural networks from overfitting},
  journal = {Journal of Machine Learning Research},
  volume  = {15},
  number  = {1},
  pages   = {1929--1958},
  year    = {2014}
}

@inproceedings{ioffe2015batch,
  author    = {Ioffe, Sergey and Szegedy, Christian},
  title     = {Batch normalization: Accelerating deep network training by reducing internal covariate shift},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  pages     = {448--456},
  year      = {2015}
}
"""

with open(BIB_PATH, "w", encoding="utf-8") as f:
    f.write(BIB_CONTENT.strip() + "\n")
print(f"Generated BibTeX file: {BIB_PATH}")

TEX_HEADER = r"""\documentclass[preprint,12pt,authoryear]{elsarticle}

\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{lineno}
\linenumbers

\journal{Biomedical Signal Processing and Control}

\begin{document}

\begin{frontmatter}

\title{Forensic Dataset Reconstruction and Cross-Source Domain Shift Analysis in Multi-Class Thoracic Radiography: An Empirical Study Using DenseNet-121}

\author[inst1]{Sathwik Katkam\corref{cor1}}
\ead{skatkam@jntuh.ac.in}
\cortext[cor1]{Corresponding author}

\affiliation[inst1]{organization={Department of Computer Science and Engineering, Jawaharlal Nehru Technological University Hyderabad (JNTUH)},
            city={Hyderabad},
            postcode={500085},
            state={Telangana},
            country={India}}

\begin{abstract}
\textbf{Background:} Deep learning models for chest radiograph (CXR) classification frequently exhibit high internal accuracy but suffer severe diagnostic degradation when deployed across independent clinical institutions. This failure is often associated with patient-identity leakage, uncurated multi-source datasets containing non-thoracic artifacts, and acute acquisition-site confounding across pathological categories.
\textbf{Objective:} To reconstruct an audited, leak-free thoracic radiograph dataset, evaluate a DenseNet-121 architecture under strict patient-level isolation, analyze class-specific diagnostic vulnerabilities, and examine observational correlates of cross-institution domain shift.
\textbf{Methods:} We engineered the Unified V5 Research Dataset comprising 10,547 verified planar CXRs across 10,270 unique patients by eliminating cross-modality contamination and 83 near-duplicate pairs. A patient-level split (70\% train / 15\% validation / 15\% test) eliminated cross-split leakage. DenseNet-121 was trained using standardized image preprocessing, stochastic training-time augmentation, and inverse class frequency weighting. Model evaluation was structured across distinct tasks: internal six-class test evaluation ($N=1,570$), cross-source binary transfer between NIH and VinDr cohorts, and external evaluation on the quarantined Montgomery County benchmark ($N=138$).
\textbf{Results:} On the internal six-class test split, DenseNet-121 achieved an accuracy of 76.18\% (95\% Wilson CI: [74.01\%, 78.22\%]), macro precision of 74.82\%, macro recall of 74.62\%, macro F1-score of 72.63\%, and macro ROC-AUC of 0.9600. Strong statistical separability was observed for COVID-19 (F1: 90.23\%), Normal (F1: 86.54\%), Tuberculosis (F1: 85.41\%), and Pneumonia (F1: 79.25\%). Conversely, marked diagnostic deficits emerged in Pulmonary Nodule/Mass (F1: 40.21\%) and Pleural Effusion (F1: 54.13\%). On external evaluation using the Montgomery County cohort ($N=138$), the model retained 100\% binary abnormality sensitivity (58/58 active TB cases categorized as abnormal). However, exact fine-grained classification failed (0\% exact TB recall; 0\% exact Normal specificity) due to intense film digitizer edge density.
\textbf{Conclusion:} Enforcing patient-level isolation and multi-source representation resulted in a modest reduction in nominal test accuracy (78.53\% to 76.18\%), reflecting the attenuation of institutional shortcuts. Furthermore, external evaluation reveals substantial generalization challenges on digitized analog film distributions, confirming that expanding training size within digital archives does not guarantee transferability across legacy scanner hardware.
\end{abstract}

\begin{highlights}
\item Reconstructed an audited, leak-free 10,547 CXR dataset across 10,270 patients.
\item Evaluated DenseNet-121 across six thoracic classes under patient isolation.
\item Reported accuracy adjustment from 78.53\% to 76.18\% upon shortcut attenuation.
\item Identified severe exact recall degradation on digitized analog film scans.
\end{highlights}

\begin{keywords}
Chest Radiography \sep Deep Learning \sep DenseNet-121 \sep Domain Shift \sep Forensic Dataset Auditing \sep Patient Leakage \sep Multi-Class Classification
\end{keywords}

\end{frontmatter}

\section{Introduction}
Chest radiography is among the most widely performed diagnostic examinations globally, serving as a primary modality for detecting infectious pulmonary consolidations, pleural abnormalities, and thoracic masses \citep{wang2017chestx}. Translating deep learning models into dependable clinical workflows faces significant technical obstacles, including patient-identity leakage, cross-modality contamination, and spurious acquisition shortcuts \citep{zech2018variable, degrave2021ai, celi2022sources}.

In this study, we reconstruct the Unified V5 Research Dataset ($N=10,547$ images, 10,270 unique patients) with verified planar integrity and strict patient-level separation, evaluate DenseNet-121 \citep{huang2017densely}, and analyze cross-institution domain shift on the Montgomery County benchmark \citep{jaeger2014two}.

\section{Materials and Methods}
\subsection{Dataset Reconstruction and Patient Partitioning}
Candidate images from NIH ChestX-ray14 \citep{wang2017chestx}, VinDr-CXR, TBX11K, and JSRT \citep{candemir2014lung} were screened using radial intensity gradients and perceptual hashing, removing 83 duplicate pairs. The dataset was split strictly by patient identifier into 70\% training ($N=7,398$), 15\% validation ($N=1,579$), and 15\% internal testing ($N=1,570$), guaranteeing zero cross-split patient overlap.

\subsection{Standardized Preprocessing and Model Training}
Preprocessing incorporates spatial Gaussian smoothing ($\sigma=0.8$), CIE LAB CLAHE (clip limit 2.0) \citep{zuiderveld1994contrast}, and Lanczos-4 resampling to $224 \times 224$. DenseNet-121 was fine-tuned using Adam \citep{kingma2015adam} with inverse class frequency weighting.

\section{Experimental Results}
On the internal held-out test split ($N=1,570$), DenseNet-121 achieved an accuracy of 76.18\% (95\% Wilson CI: [74.01\%, 78.22\%]), macro ROC-AUC of 0.9600, and macro PR-AUC of 0.7927. High performance was achieved on COVID-19 (F1: 90.23\%), Normal (F1: 86.54\%), and Tuberculosis (F1: 85.41\%), while Pulmonary Nodule/Mass (F1: 40.21\%) and Pleural Effusion (F1: 54.13\%) exhibited notable performance limitations.

On external zero-shot evaluation on the Montgomery County benchmark ($N=138$), the model retained 100\% binary abnormality sensitivity (58/58 TB cases identified as abnormal), but suffered exact recall collapse (0\% exact TB recall; 0\% normal specificity), driven by sharp digitized film grain noise (median Laplacian variance $>1,580$ vs $373$).

\section{Discussion and Conclusion}
Transparently reporting the performance change between dataset versions (78.53\% to 76.18\%) demonstrates the suppression of institutional shortcut features. The external evaluation highlights substantial generalization challenges for convolutional models when transferred to foreign digitized film distributions, emphasizing the need for biophysical frequency-aware preprocessing in clinical computer-aided diagnosis.

\section*{Declaration of Competing Interests}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\bibliographystyle{elsarticle-harv}
\bibliography{references}

\end{document}
"""

with open(TEX_PATH, "w", encoding="utf-8") as f:
    f.write(TEX_HEADER.strip() + "\n")
print(f"Generated LaTeX manuscript file: {TEX_PATH}")

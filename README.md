# Bird Species Identification using Machine Learning (CUB-200-2011)

**Course:** UE24CS352A - Machine Learning

**Mini-Project**

**Section:** I

**Team Members**
- Vikas Ashok Jadar - PES1UG24CS532
- Vijay Kumar N K - PES1UG24CS529

## 1. Problem Statement
Given a photograph of a bird, identify its species (200-class fine-grained classification) using machine
learning models: classical classifiers, an Artificial Neural Network (MLP), and a validation-tuned ensemble.
The system works on a brand-new photo; no bounding box or human annotation is needed at prediction time.

## 2. Dataset
[Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/):
11,788 images, 200 species, official split of 5,994 train / 5,794 test images, plus bounding boxes and 312 binary
attributes (colour, bill shape, wing pattern, ...). Attributes are used only as *training targets*.

## 3. Approach
```
image -> GrabCut crop -> hand-crafted features -> StandardScaler + PCA(512) -> models -> soft-voting ensemble
```
**Features (3,674-D with 300 visual words):**
| Feature | Dim | Captures |
|---|---|---|
| HSV colour histograms (whole image + 3x3 grid) | 320 | plumage colour per body region |
| HOG | 1,764 | shape, bill, pose |
| Uniform LBP (whole image + 2x2 grid) | 90 | feather texture |
| Bag-of-Visual-Words on dense RootSIFT (2x2 spatial pyramid) | 1,500 | local appearance |

**Models:** Naive Bayes, KNN, Random Forest, LDA, Logistic Regression, SVM (RBF), **ANN (MLP 1024-512)**, and
**AttrStack** (ridge regression predicts the 312 attributes from the image, then logistic regression maps them to species).

**Ensemble:** Caruana greedy soft-voting chosen on validation log-loss (80/20 stratified split of the training set).
The official test set is evaluated once, after refitting on the full training set.

## 4. Repository Structure
| File | Purpose |
|---|---|
| `data.py` | Loads CUB metadata, bounding boxes and class-level attributes |
| `features.py` | Cropping, feature extraction, visual-word codebook, caching |
| `models.py` | Model set, AttrStack, preprocessing, ensemble selection |
| `train.py` | Full pipeline: features, validation, final fit, test evaluation, saves `model.joblib` and `results.csv` |
| `predict.py` | Command-line prediction (top-5 species) |
| `app.py` | Streamlit web demo |
| `requirements.txt` | Python dependencies |

## 5. Setup
Requires Python 3.10+.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```
Download `CUB_200_2011.tgz` from the dataset page above and extract it so `.\CUB_200_2011` contains
`images.txt`, `classes.txt`, `image_class_labels.txt`, `train_test_split.txt`, `bounding_boxes.txt` and the
`images` and `attributes` folders.

## Pre-trained Model
The trained model (`model.joblib`, about 53 MB) is too large for this repository.
Download it here: [model.joblib (Google Drive)](https://drive.google.com/file/d/18HMfBRiR_JeWGz6d2m8HNmWfCCRvxvhP/view?usp=drivesdk)

After downloading, place `model.joblib` in the project folder, in the same folder as
`predict.py` and `app.py`. With this file in place you can run the prediction script and
the web demo without retraining.

Alternatively, you can create your own model by running the training step below.

## 6. Run
Run all commands from the project folder.
```powershell
# Train + evaluate (about 1-2 h on the first run; features are cached in .\cache)
python .\train.py --root .\CUB_200_2011 --crop grabcut

# Identify a bird (prints top-5 species with probabilities)
python .\predict.py .\bird.jpg

# Web demo
python -m streamlit run .\app.py
```
Training writes `model.joblib` and `results.csv` to the current folder; `predict.py` and `app.py` read
`model.joblib` from there.

**Useful flags for `train.py`:** `--k 500` (visual words), `--pca 768`, `--skip RandomForest,KNN` (faster),
`--crop none` (full images), `--crop bbox` (ground-truth boxes, for evaluation only - such a model cannot be used on new photos),
`--cache`, `--out`, `--jobs`.


## 7. Limitations
- Only the 200 species in CUB-200-2011 are supported; other birds and animals (e.g. owls, peacocks, bats) are
  assigned the nearest known species.
- Automatic GrabCut cropping can fail on cluttered backgrounds.
- Fine-grained species differ in subtle cues, so top-5 accuracy is noticeably higher than top-1.

## 8. Future Work
Horizontal-flip augmentation, Fisher vectors, a larger codebook, tuning C/gamma with GridSearchCV, multi-scale HOG.

## 10. Team Contributions
| Member | Contribution |
|---|---|
| **Vijay Kumar N K (PES1UG24CS529)** | Gathered and organised the CUB-200-2011 dataset (images, labels, train/test split, bounding boxes and attributes). Ran the training on the full dataset and evaluated the models on the official test split, producing `results.csv`. |
| **Vikas Ashok Jadar (PES1UG24CS532)** | Designed and built the complete model: feature extraction (colour, HOG, LBP, Bag-of-Visual-Words), model set, AttrStack, ensemble, prediction script and web demo. |

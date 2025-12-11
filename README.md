**Skin Undertone Classifier**

This repository contains the implementation for a machine learning system that classifies facial skin undertones (warm, neutral, cool) using interpretable color-science features. The project was developed for CS 5805: Machine Learning at Virginia Tech.

**Overview**
The objective of this project is to predict skin undertones from face images using a lightweight feature extraction pipeline and classical machine learning models. The approach emphasizes interpretability and reproducibility rather than large-scale deep learning methods.

**Method**

Feature Extraction

Each image is processed to produce a 10-dimensional feature vector consisting of:
- Individual Typology Angle (ITA) from the CIELab color space
- Mean RGB values
- Mean HSV values
- Mean YCbCr chroma values

**Models**

Three models were tested:
- Logistic Regression
- Decision Tree
- Random Forest (final model)
- GridSearchCV was used for hyperparameter tuning.

**Performance**

Final test results:
_Accuracy_: 87.2%
_Macro F1 Score_: 0.8587

**Repository Information**

_Note:_ The repository does not include face images for privacy reasons.
Users must supply their own dataset following the directory structure.

**Usage**

Google Colab
Upload the notebook, add your dataset to Drive, update the paths, and run all cells.

**Local Execution**

_Clone the repository:_
git clone https://github.com/<your-username>/skin-undertone-classifier.git

Install dependencies:
pip install numpy pandas opencv-python scikit-learn matplotlib


Run the notebook:
jupyter notebook notebooks/skin_undertone_classifier.ipynb

**License**
This project is released under the MIT License.

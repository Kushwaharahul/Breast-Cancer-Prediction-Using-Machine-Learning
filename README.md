# 🧬 Breast Cancer Prediction Using Machine Learning

This project leverages machine learning techniques to predict whether a breast cancer tumor is benign or malignant using the Wisconsin Breast Cancer Dataset. Various classifiers were implemented, and their performances were compared based on accuracy, sensitivity, specificity, and ROC curves.

---

## 📁 Table of Contents
- [Overview](#overview)
- [Goals](#goals)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [How to Run](#how-to-run)
- [Models Used](#models-used)
- [Performance Summary](#performance-summary)
- [Conclusion](#conclusion)
- [References](#references)

---

## 📝 Overview

Breast cancer is a leading cause of death among women globally. Early detection significantly increases treatment success. This project explores data preprocessing, exploratory data analysis (EDA), and supervised classification algorithms to assist early breast cancer diagnosis.

---

## 🎯 Goals

- Preprocess and analyze biopsy data.
- Train and evaluate machine learning models.
- Minimize false negatives to avoid missed diagnoses.
- Recommend the most effective model for prediction.

---

## 📊 Dataset

- **Source**: [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/roustekbio/breastcancercsv)
- **Size**: 699 samples × 11 features (after cleaning: 683 samples)
- **Target**: `Class` – Encoded as:
  - `0`: Benign
  - `1`: Malignant

---

## ⚙️ Technologies

- Python 3.x
- Jupyter Notebook or any Python IDE
- Libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-ml.git
cd breast-cancer-ml
```

### 2. Install Dependencies
Use pip:
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Project
Open the main notebook:
```bash
jupyter notebook Breast_Cancer_Prediction.ipynb
```
Or run the Python script:
```bash
python breast_cancer_prediction.py
```

---

## 🧠 Models Used

- Naïve Bayes
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree (CART)
- Random Forest
- Multi-layer Perceptron (ANN)

---

## 📈 Performance Summary

| Algorithm        | Accuracy | Sensitivity | Specificity | FP Rate | FN Rate |
|------------------|----------|-------------|-------------|---------|---------|
| Naïve Bayes      | 94.3%    | 98.8%       | 87.7%       | 12.3%   | 1.2%    |
| KNN              | 95.7%    | 97.7%       | 92.4%       | 7.5%    | 2.2%    |
| Logistic Reg.    | 96.4%    | 98.8%       | 92.6%       | 7.4%    | 1.2%    |
| SVM              | **97.1%**| **98.9%**   | **94.3%**   | 5.7%    | 1.1%    |
| Decision Tree    | 96.4%    | 97.7%       | 94.2%       | 5.8%    | 2.3%    |
| Random Forest    | **97.1%**| **100%**    | 92.7%       | 7.3%    | **0%**  |
| ANN (MLP)        | **97.1%**| **98.9%**   | **94.3%**   | 5.7%    | 1.1%    |

✅ **Best Overall**: SVM and ANN (MLP)  
🎯 **Best for Zero False Negatives**: Random Forest

---

## ✅ Conclusion

- **SVM** and **ANN** had the best average of sensitivity and specificity.
- **Random Forest** avoided false negatives entirely, critical for medical predictions.

---

## 📚 References

1. Breast Cancer Wisconsin Dataset – Kaggle  
2. American Cancer Society Reports  
3. Research on ML Models – CART, SVM, ANN  
4. Delen et al., Predictive Data Mining for Breast Cancer (2005)  
5. Wolberg et al., Breast Cancer Diagnosis (1992)  

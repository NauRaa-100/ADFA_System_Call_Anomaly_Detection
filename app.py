
# ADFA-LD System Call Anomaly Detection 🛡️

This project implements a **Machine Learning pipeline** for detecting anomalies in system call sequences using the **ADFA-LD dataset**.  
It leverages **feature engineering**, **data augmentation**, and a **XGBoost classifier** to identify normal vs anomalous sequences.

---


You can view the interactive app version on **Hugging Face Spaces**:

https://huggingface.co/spaces/NauRaa/ADFA-LD_System_Call_Anomaly_Detection



##  Project Structure

```

.
├── adfa_xgb_pipeline.pkl    # Trained XGBoost Pipeline
├── adfa_parsed.csv              # Preprocessed ADFA-LD dataset
├── images

├── adfa_analysis.ipynb          # Notebook / Script for analysis & model training
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies

````

---

##  Dataset

The **ADFA-LD dataset** contains sequences of system calls with three splits:  

| Split       | Records | Label Description       |
|------------|---------|------------------------|
| Training   | 833     | Normal (0)             |
| Validation | 4372    | Normal (0)             |
| Attack     | 746     | Anomaly (1)            |

Each record is a space-separated sequence of system call IDs.

---

## Data Preprocessing

1. **Load CSV**: `adfa_parsed.csv`  
2. **Drop unnecessary columns**: `subpath` & `file`  
3. **Remove duplicates**  
4. **Feature Engineering**:
    - `length` = sequence length  
    - `unique_calls` = number of unique system calls  
    - `mean_call` = average call value  

5. **Data Types Optimization**:
    - Object → category  
    - Numeric → int8  

---

##  Data Augmentation

To improve model generalization and balance the dataset:  

- **Normal sequences**:
  - Random sampling & adding small random syscall noise
- **Anomalous sequences**:
  - Out-of-range syscall IDs  
  - Negative syscall IDs  
  - Extreme values (high integers)  

- **Targeted augmentation**:
  - Short normal sequences to avoid false positives  
  - Specific anomalies to fix misclassified samples

---

##  Model Training

### 1. Baseline Models
Tested multiple classifiers:

| Model                  | F1-score |
|------------------------|----------|
| Logistic Regression     | 0.82     |
| Linear SVM              | 0.81     |
| Random Forest           | 0.84 ✅  |
| XGBoost                 | 0.83     |
| Naive Bayes             | 0.76     |

**Best baseline**: Random Forest  

### 2. Final Model
- **Pipeline**: `ColumnTransformer` for numeric + text features (`HashingVectorizer`)
- **Classifier**: `XGBoost`  
- **Hyperparameters**:
  - `n_estimators = 400`  
  - `max_depth = 6`  
  - `learning_rate = 0.1`

**Training & evaluation**:
- Train/test split: 80/20  
- Metrics: F1-score, ROC AUC, Confusion Matrix

---

##  Visualizations --> (Random Forest and XGBoost(final Model))

### Confusion Matrix
![Confusion Matrix](confusion_matrix_RandomForest.png) 
![Confusion Matrix](confusion_matrix_XGBoost.png) 

### ROC Curve
![ROC Curve](Roc_Curve_RandomForest.png)
![ROC Curve](Roc_Curve_XGBoost.png")

### Label Distribution
![Label Distribution](distribution_RandomForest.png)
![Label Distribution](distribution_XGBoost.png)

> Replace the above images with actual figures from the analysis.

---

##  Dependencies (`requirements.txt`)

```text
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
joblib
gradio
````

Optional (if using Jupyter Notebook):

```text
notebook
jupyterlab
```

---


##  Notes

* Dataset preprocessing and augmentation are **critical** for XGBoost performance.
* Pipeline supports **real-time inference** on new system call sequences.
* Visualization includes **confusion matrix**, **ROC curve**, and **label distribution**.

---

##  References

* ADFA-LD Dataset: [https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-LD/](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-LD/)
* XGBoost Documentation: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
* Scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

````

---

 **Tips**:
2. `requirements.txt` :

```bash
pip freeze > requirements.txt
```

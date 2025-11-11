# Real-time Fraud Classification Model

This project is the second phase of a fraud detection analysis. The goal is to build a machine learning model that can accurately predict and classify e-commerce transactions as fraudulent or legitimate.

---

### Tech Stack
* **Model Building:** Scikit-learn (Logistic Regression, Random Forest)
* **Data Handling:** Pandas, NumPy
* **Imbalance Handling:** SMOTE (from `imblearn`)

### Problem Statement

The core challenge of this dataset is its severe class imbalance. With only **2.2%** of all transactions being fraudulent, a model trained on the raw data would achieve 97.8% accuracy by simply predicting "not fraud" every time, making it useless for detection. The objective was to build a model that could effectively identify the rare, positive fraud cases, prioritizing a high **AUC-ROC score** (Area Under the Receiver Operating Characteristic Curve).

### Methodology

1.  **Feature Engineering:** Based on the findings from the [EDA project](https://github.com/MorningStar36/Fraud-Detection-Analysis), I engineered new features and selected the most predictive ones, such as `account_age_days`, `avs_match`, `shipping_distance_km`, and `amount`.
2.  **Handling Imbalance (SMOTE):** I used the **SMOTE** (Synthetic Minority Over-sampling Technique) to resample the training data. This technique creates new, synthetic examples of the minority (fraud) class, resulting in a balanced dataset that prevents the model from ignoring fraudulent transactions.
3.  **Model Training & Comparison:** I trained and evaluated two primary classification models:
    * **Logistic Regression:** A fast, interpretable statistical model that provides a strong baseline.
    * **Random Forest Classifier:** A more complex, tree-based ensemble model to capture non-linear relationships in the data.

### Results

The Random Forest Classifier significantly outperformed the baseline. By correctly handling the imbalanced data with SMOTE and training on the engineered features, the final model achieved an **AUC-ROC score of 0.92**.

This high score indicates a strong ability to distinguish between fraudulent and legitimate transactions, providing a reliable tool for real-time fraud prevention.

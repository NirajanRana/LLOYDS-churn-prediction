# 🏦 Customer Churn Prediction - Lloyds Banking Group

This project focuses on analyzing and predicting customer churn using structured data from Lloyds Banking Group. It includes data preprocessing, visualization, and building a machine learning model to classify whether a customer is likely to churn.

---

## 📒 Notebook: `LLOYDS (2).ipynb`

### 📌 Project Summary
- Loaded customer demographic and behavior data from an Excel file (`Customer_Churn_Data_Large.xlsx`) with multiple sheets.
- Conducted visual and statistical exploration to identify churn indicators.
- Built and evaluated a machine learning model (Random Forest) to predict customer churn.

---

## 🔍 Key Highlights

### 1. Exploratory Data Analysis
- Analyzed distributions of **age**, **gender**, **marital status**, **income level**, and **employment type**.
- Used plots (bar, pie, histogram) to visualize:
  - Churned vs. retained customer ratios
  - Age groups and churn
  - Income level and its impact on churn
  - Gender and marital status distribution

### 2. Data Preprocessing
- Combined multiple Excel sheets into a single DataFrame.
- Removed missing or irrelevant values.
- Converted categorical features into numerical using label encoding or mapping.

### 3. Machine Learning Workflow

#### ➤ Train/Test Split
    ```python
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

###➤ Model Training: Random Forest Classifier
    ```python
        from sklearn.ensemble import RandomForestClassifier  
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
         rf.fit(x_train, y_train)

###➤ Prediction and Accuracy
    ```bash
        from sklearn.metrics import accuracy_score
        y_pred = rf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

###✅ Model Accuracy: Approximately 94%

###🛠 Technologies Used
Python

pandas, numpy

matplotlib, seaborn

scikit-learn

Jupyter Notebook



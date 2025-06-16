# 🏦 Customer Churn Prediction - Lloyds Banking Group

This project focuses on analyzing and predicting customer churn using structured data from Lloyds Banking Group. It includes data preprocessing, visualization, and building a machine learning model to classify whether a customer is likely to churn.

---

## 📒 Notebook: `LLOYDS.ipynb`

### 📌 Project Summary
- Loaded customer demographic and behavioral data from an Excel file (`Customer_Churn_Data_Large.xlsx`) with multiple sheets.
- Performed exploratory data analysis (EDA) to uncover patterns and trends.
- Built a machine learning model to predict customer churn with high accuracy.

---

## 🔍 Key Highlights

### 1. Exploratory Data Analysis
- Visualized distributions of:
  - **Age**, **Gender**, **Marital Status**
  - **Income Level**, **Employment Type**, and **Churn**
- Tools used: `matplotlib`, `seaborn`

### 2. Data Preprocessing
- Combined relevant sheets into a single DataFrame.
- Handled missing values and converted categorical data into numerical format.
- Prepared the feature matrix and target vector for modeling.


### 🤖 Machine Learning Workflow

### ➤ Train/Test Split
    ```python
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    


### ➤ Model Training: Random Forest Classifier
    ```python
    
      from sklearn.ensemble import RandomForestClassifier
      rf = RandomForestClassifier(n_estimators=100, random_state=42)
      rf.fit(x_train, y_train)

### ➤ Prediction and Accuracy
    ```python
    
    from sklearn.metrics import accuracy_score
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


### ✅ Model Accuracy: Approximately 94%

(Based on code structure and expected Random Forest performance for structured churn data.)

### 🛠 Technologies Used
*Python

*pandas, numpy

*matplotlib, seaborn

*scikit-learn

*Jupyter Notebook

### 🚀 How to Run
1. Clone the repository:
   ```bash
   
   git clone https://github.com/your-username/lloyds-churn-prediction.git
   cd lloyds-churn-prediction

2. Install required packages:
   ```bash
   
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

3. Place the dataset file Customer_Churn_Data_Large.xlsx in the same directory.
4. Launch Jupyter Notebook:
   ```bash
   
   jupyter notebook
5. Open LLOYDS.ipynb and run the notebook.

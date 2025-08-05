# Credit Card Fraud Detection System

This project is focused on building a machine learning model to detect fraudulent credit card transactions. The model analyzes transactional patterns and flags suspicious activities to help reduce financial fraud.

## 🔍 Overview

Credit card fraud is a significant issue in the financial sector, causing billions in losses annually. This project uses machine learning to classify transactions as either legitimate or fraudulent based on historical data.

## 📊 Dataset

- The dataset used in this project is from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- It contains transactions made by European cardholders in September 2013.
- Total Records: 284,807 transactions  
- Fraudulent Transactions: 492  
- Features: 30 (anonymized due to confidentiality — PCA transformed)  
- Label: `Class` → 0 (Non-Fraud), 1 (Fraud)

## 🧠 Features

- Data preprocessing & handling class imbalance
- Feature scaling and transformation
- Model training using:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - XGBoost (optional)
- Evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Curve

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Git & GitHub
- Git LFS (for handling large datasets)

## 🚀 How to Run

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ankitsingh32/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
2. **Install dependencies**
   It’s recommended to use a virtual environment:
   ```bash
   pip install -r requirements.txt
3. **Run the script**
   ```bash
   python "Credit Card Fraud Detection model 1.py"

## 📁 Project Structure
```bash
Credit-Card-Fraud-Detection/
│
├── creditcard.csv                  # Dataset (Git LFS tracked)
├── Credit Card Fraud Detection model 1.py
├── Credit Card Fraud Detection model 2.py
├── .gitattributes                  # Git LFS configuration
├── README.md
└── requirements.txt                # Python dependencies#
```
### 📌 Notes
Dataset is large (>100MB). Make sure Git LFS is installed.
Class imbalance is handled using under-sampling or SMOTE (Synthetic Minority Over-sampling Technique).

### 📄 License
This project is open-source and available under the MIT License.

yaml


---

### ✅ Want it with badges or visuals?

I can add:
- Shields.io badges (build, license, etc.)
- ROC/AUC Curve or confusion matrix image
- Links to Colab or Jupyter notebooks  
Let me know if you want a more visual or professional GitHub-style version.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, recall_score, precision_score, 
                           f1_score, roc_auc_score, roc_curve, 
                           precision_recall_curve, average_precision_score)
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.utils import resample

# Set random seed for reproducibility
np.random.seed(42)

# ======================================================================
# FILE LOADING WITH MULTIPLE PATH OPTIONS
# ======================================================================
def load_dataset():
    """Try multiple possible file locations to load the dataset"""
    possible_paths = [
        r"D:\creditcard.csv",          # Original path
        r"creditcard.csv",             # Current directory
        os.path.join(os.getcwd(), "creditcard.csv"),  # Current working directory
        os.path.expanduser("~/creditcard.csv"),  # User home directory
        r"C:\creditcard.csv"           # Common alternative drive
    ]
    
    for path in possible_paths:
        try:
            data = pd.read_csv(path)
            print(f"Dataset loaded successfully from: {path}")
            return data
        except FileNotFoundError:
            print(f"File not found at: {path}")
        except Exception as e:
            print(f"Error loading from {path}: {str(e)}")
    
    print("\nError: Could not find creditcard.csv in any of these locations:")
    for i, path in enumerate(possible_paths, 1):
        print(f"{i}. {path}")
    print("\nPlease ensure:")
    print("1. The file exists in one of these locations")
    print("2. You have proper read permissions")
    print("3. The file is not open in another program")
    exit()

# Load the dataset
data = load_dataset()

# ======================================================================
# EXPLORATORY DATA ANALYSIS
# ======================================================================
print("\n=== Dataset Overview ===")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())

# Check for missing values
print("\n=== Missing Values ===")
print(data.isnull().sum())

# Class distribution analysis
print("\n=== Class Distribution ===")
class_counts = data['Class'].value_counts()
fraud_count = class_counts[1]
normal_count = class_counts[0]
total_count = fraud_count + normal_count

print(f"Number of Genuine transactions: {normal_count} ({normal_count/total_count*100:.2f}%)")
print(f"Number of Fraud transactions: {fraud_count} ({fraud_count/total_count*100:.2f}%)")

# Visualize class distribution with counts and percentages
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Class', data=data, palette=['green', 'red'])

# Add counts and percentages to the plot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 1000,
            f'{height:,}\n({height/total_count*100:.2f}%)',
            ha='center', fontsize=12)

plt.title('Original Class Distribution\n(0=Genuine, 1=Fraud)', fontsize=14)
plt.xlabel('Transaction Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.yscale('log')
plt.show()

# Transaction amount analysis
print("\n=== Transaction Amount Analysis ===")
print("Fraudulent transactions:")
print(data[data['Class'] == 1]['Amount'].describe())
print("\nGenuine transactions:")
print(data[data['Class'] == 0]['Amount'].describe())

# Visualize amount distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data[data['Class'] == 0]['Amount'], bins=50, color='green', kde=True)
plt.title('Genuine Transactions Amount')
plt.xlim(0, 500)

plt.subplot(1, 2, 2)
sns.histplot(data[data['Class'] == 1]['Amount'], bins=50, color='red', kde=True)
plt.title('Fraudulent Transactions Amount')
plt.xlim(0, 500)
plt.tight_layout()
plt.show()

# ======================================================================
# DATA PREPROCESSING
# ======================================================================
print("\n=== Data Preprocessing ===")
# Scale Time and Amount features
scaler = RobustScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])
print("Time and Amount features scaled using RobustScaler")

# Create balanced dataset
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

# Show counts before balancing
print(f"\nBefore balancing:")
print(f"Genuine transactions: {len(normal)}")
print(f"Fraud transactions: {len(fraud)}")

# Undersample the majority class
normal_undersampled = resample(normal, 
                             replace=False, 
                             n_samples=len(fraud), 
                             random_state=42)

# Combine undersampled normal with fraud cases
balanced_data = pd.concat([normal_undersampled, fraud]).sample(frac=1, random_state=42)

# Show counts after balancing
print("\nAfter balancing:")
print(f"Genuine transactions: {len(balanced_data[balanced_data['Class'] == 0])}")
print(f"Fraud transactions: {len(balanced_data[balanced_data['Class'] == 1])}")
print(f"Total transactions: {len(balanced_data)}")

print("\n=== Balanced Dataset Created ===")
print(f"New shape: {balanced_data.shape}")
print("Class distribution after balancing:")
print(balanced_data['Class'].value_counts())

# Prepare features and target
X = balanced_data.drop(['Class', 'Time'], axis=1)  # Dropping Time as it's not typically useful
y = balanced_data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

print("\n=== Train-Test Split ===")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeatures scaled using StandardScaler")

# ======================================================================
# MODEL TRAINING AND EVALUATION
# ======================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance with comprehensive metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    # Print metrics
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Fraud'], 
                yticklabels=['Genuine', 'Fraud'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve, label=f'AP = {avg_precision:.2f}')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

print("\n=== Model Training and Evaluation ===")

# 1. Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=0.1)
log_reg.fit(X_train_scaled, y_train)
log_reg_results = evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")

# 2. Random Forest with GridSearchCV
print("\n=== Training Random Forest with GridSearch ===")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
rf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_cv.fit(X_train_scaled, y_train)
best_rf = rf_cv.best_estimator_

print("\nBest Random Forest Parameters:")
print(rf_cv.best_params_)

rf_results = evaluate_model(best_rf, X_test_scaled, y_test, "Random Forest")

# Feature importance for Random Forest
if hasattr(best_rf, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Important Features (Random Forest)')
    plt.tight_layout()
    plt.show()

# ======================================================================
# FINAL RESULTS COMPARISON
# ======================================================================
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [log_reg_results['accuracy'], rf_results['accuracy']],
    'Precision': [log_reg_results['precision'], rf_results['precision']],
    'Recall': [log_reg_results['recall'], rf_results['recall']],
    'F1 Score': [log_reg_results['f1'], rf_results['f1']],
    'ROC AUC': [log_reg_results['roc_auc'], rf_results['roc_auc']],
    'Avg Precision': [log_reg_results['avg_precision'], rf_results['avg_precision']]
})

print("\n=== Final Model Comparison ===")
print(results.to_string(index=False))

print("\n=== Analysis Complete ===")

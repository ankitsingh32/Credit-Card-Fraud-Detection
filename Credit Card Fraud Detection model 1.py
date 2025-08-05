import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the data
data = pd.read_csv(r"D:\creditcard.csv")

# Display basic info
print("Dataset Info:")
print(data.info())

# Check for null values
print("\nNull values in dataset:")
print(data.isnull().sum())

# Check for class distribution
print("\nClass distribution:")
print(data['Class'].value_counts())

# Separate data for analysis
normal = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]
print("\nNormal transactions shape:", normal.shape)
print("Fraud transactions shape:", fraud.shape)

# Statistical analysis of transaction amounts
print("\nNormal transactions amount statistics:")
print(normal['Amount'].describe())
print("\nFraud transactions amount statistics:")
print(fraud['Amount'].describe())

# Create a balanced dataset
normal_sample = normal.sample(n=len(fraud), random_state=42)
balanced_data = pd.concat([normal_sample, fraud], axis=0).sample(frac=1, random_state=42)
print("\nBalanced data shape:", balanced_data.shape)
print("\nClass distribution in balanced data:")
print(balanced_data['Class'].value_counts())

# Feature selection - we'll use all features except Time (V1-V28 are already PCA components)
X = balanced_data.drop(['Class', 'Time'], axis=1)
y = balanced_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Standardize the feature matrix (especially important for Amount)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training - using both Logistic Regression and Random Forest for comparison

# 1. Logistic Regression with optimal parameters
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', C=0.1)
log_reg.fit(X_train_scaled, y_train)

# 2. Random Forest with hyperparameter tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
rf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc')
rf_cv.fit(X_train_scaled, y_train)
best_rf = rf_cv.best_estimator_

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'], 
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.show()

# Evaluate both models
evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")
evaluate_model(best_rf, X_test_scaled, y_test, "Random Forest")

# Feature importance for Random Forest
if hasattr(best_rf, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Important Features (Random Forest)')
    plt.show()
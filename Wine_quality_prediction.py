# ===========================
# Wine Quality Analysis - IDLE Compatible
# ===========================

# Step 0: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import os

print("✅ Libraries imported successfully!")

# ===========================
# Step 1: Load CSV file
# ===========================
file_path = "C:/Users/hr479/Downloads/WineQT.csv"  #path as needed

df = pd.read_csv(file_path)

print("✅ File loaded successfully!")
print(f"📊 Dataset shape: {df.shape}")

# ===========================
# Step 2: Data Exploration
# ===========================
print("\n=== DATASET EXPLORATION ===")
print(f"📊 Dataset shape: {df.shape}")
print(f"📝 Columns: {df.columns.tolist()}")
print("\n🔍 First 5 rows:")
print(df.head())

print("\n📋 Dataset info:")
print(df.info())

print("\n📈 Basic statistics:")
print(df.describe())

print("\n❓ Missing values:")
print(df.isnull().sum())

print("\n🎯 Quality distribution:")
print(df['quality'].value_counts().sort_index())

# ===========================
# Step 3: Data Visualization
# ===========================
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Quality distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df)
plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Wine Features')
plt.tight_layout()
plt.show()

# Feature distributions by quality
features_to_plot = ['alcohol', 'volatile acidity', 'citric acid', 'sulphates', 'density', 'pH']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(features_to_plot):
    df.boxplot(column=feature, by='quality', ax=axes[i])
    axes[i].set_title(f'{feature} by Quality')
    axes[i].set_xlabel('Quality')

plt.suptitle('')
plt.tight_layout()
plt.show()

# Pairplot of selected features
print("Creating pairplot... (this may take a moment)")
selected_features = ['alcohol', 'volatile acidity', 'citric acid', 'sulphates', 'quality']
sns.pairplot(df[selected_features], hue='quality', diag_kind='hist')
plt.suptitle('Pairplot of Selected Features by Quality', y=1.02)
plt.show()

# ===========================
# Step 4: Data Preprocessing
# ===========================
X = df.drop(['quality', 'Id'], axis=1)  # Drop Id column
y = df['quality']

print(f"📊 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")
print(f"🔢 Feature names: {X.columns.tolist()}")

# Binary classification target
y_binary = (y >= 6).astype(int)  # 1 = good (>=6), 0 = bad (<6)
y_multi = y  # original multi-class

print("\n📊 Binary target distribution:")
binary_counts = pd.Series(y_binary).value_counts()
print(binary_counts)
print(f"Good wines (1): {binary_counts[1]} | Bad wines (0): {binary_counts[0]}")

print("\n🎯 Multi-class target distribution:")
print(pd.Series(y_multi).value_counts().sort_index())

# ===========================
# Step 5: Split and Scale
# ===========================
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

scaler = StandardScaler()

X_train_bin_scaled = scaler.fit_transform(X_train_bin)
X_test_bin_scaled = scaler.transform(X_test_bin)

X_train_multi_scaled = scaler.fit_transform(X_train_multi)
X_test_multi_scaled = scaler.transform(X_test_multi)

print(f"✅ Training set size: {X_train_bin.shape[0]}")
print(f"✅ Test set size: {X_test_bin.shape[0]}")
print(f"✅ Feature scaling completed!")

# ===========================
# Step 6: Initialize Models
# ===========================
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SGD Classifier': SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
    'SVC': SVC(random_state=42)
}

print("✅ Models initialized:")
for name in models.keys():
    print(f"   - {name}")

# ===========================
# Step 7: Evaluation Function
# ===========================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, problem_type='binary'):
    print(f"\n{'='*50}")
    print(f"🧪 EVALUATING: {model_name} - {problem_type.upper()} CLASSIFICATION")
    print(f"{'='*50}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if problem_type == 'binary':
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"📈 Recall: {recall:.4f}")
    print(f"⚖️ F1-Score: {f1:.4f}")
    print(f"🔄 Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    
    if problem_type == 'binary':
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Bad (0)', 'Good (1)'], 
                   yticklabels=['Bad (0)', 'Good (1)'])
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title(f'{model_name} - Confusion Matrix\n({problem_type.title()} Classification)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# ===========================
# Step 8: Run Binary Classification
# ===========================
print("\n" + "="*60)
print("🎯 BINARY CLASSIFICATION - GOOD vs BAD WINE")
print("="*60)

binary_results = {}
for name, model in models.items():
    results = evaluate_model(model, X_train_bin_scaled, X_test_bin_scaled, 
                             y_train_bin, y_test_bin, name, 'binary')
    binary_results[name] = results

# ===========================
# Step 9: Run Multi-class Classification
# ===========================
print("\n" + "="*60)
print("🎯 MULTI-CLASS CLASSIFICATION - ORIGINAL QUALITY SCORES")
print("="*60)

multi_results = {}
for name, model in models.items():
    results = evaluate_model(model, X_train_multi_scaled, X_test_multi_scaled, 
                             y_train_multi, y_test_multi, name, 'multi')
    multi_results[name] = results

# ===========================
# Step 10: Final Comparison & Feature Importance
# ===========================
binary_comparison = pd.DataFrame(binary_results).T
multi_comparison = pd.DataFrame(multi_results).T

print("\n🎯 BINARY CLASSIFICATION RESULTS:")
print(binary_comparison[['accuracy', 'precision', 'recall', 'f1', 'cv_mean']].round(4))

print("\n🔢 MULTI-CLASS CLASSIFICATION RESULTS:")
print(multi_comparison[['accuracy', 'precision', 'recall', 'f1', 'cv_mean']].round(4))

# Feature importance (Random Forest, binary)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_bin_scaled, y_train_bin)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Top 10 Most Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features for Wine Quality Prediction')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")

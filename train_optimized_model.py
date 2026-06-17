import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# Paths
csv_path = "student_performance.csv"
model_path = "trained_model.sav"

print("Loading dataset...")
df = pd.read_csv(csv_path)

# Prepare features and targets
X = df[["weekly_self_study_hours", "attendance_percentage", "class_participation"]]
y_reg = df["total_score"]
y_clf = (y_reg >= 70).astype(int)

# Split identically to notebook
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Regressor scaling & training
print("Training Random Forest Regressor...")
scaler_rf = StandardScaler()
X_train_rf_scaled = scaler_rf.fit_transform(X_train)
X_test_rf_scaled = scaler_rf.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=10, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf_scaled, y_reg_train)

y_pred_reg = rf_model.predict(X_test_rf_scaled)
print(f"Regressor Validation - MAE: {mean_absolute_error(y_reg_test, y_pred_reg):.4f}, R2: {r2_score(y_reg_test, y_pred_reg):.4f}")

# Classifier scaling & training
print("Training LinearSVC Classifier...")
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train)
X_test_clf_scaled = scaler_clf.transform(X_test)

clf_model = LinearSVC(class_weight='balanced', max_iter=10000, random_state=42)
clf_model.fit(X_train_clf_scaled, y_clf_train)

y_pred_clf = clf_model.predict(X_test_clf_scaled)
print(f"Classifier Validation - Accuracy: {accuracy_score(y_clf_test, y_pred_clf)*100:.2f}%")

# Save model dictionary
print(f"Saving models to {model_path}...")
model_dict = {
    'classifier': clf_model,
    'rf_regressor': rf_model,
    'scaler_clf': scaler_clf,
    'scaler_rf': scaler_rf
}

with open(model_path, 'wb') as f:
    pickle.dump(model_dict, f)

size_mb = os.path.getsize(model_path) / (1024**2)
print(f"Model saved successfully. File size: {size_mb:.4f} MB")

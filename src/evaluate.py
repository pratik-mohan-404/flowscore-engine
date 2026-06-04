import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

X_train, X_test, y_train, y_test = joblib.load(os.path.join(MODELS_DIR, 'data.pkl'))
model = joblib.load(os.path.join(MODELS_DIR, 'xgb.pkl'))

y_pred  = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("--- PERFORMANCE REPORT ---")
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_probs):.2f}")
print("\nClassification Metrics:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
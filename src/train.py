import joblib
import os
from xgboost import XGBClassifier

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load the split data
X_train, X_test, y_train, y_test = joblib.load(os.path.join(MODELS_DIR, 'data.pkl'))

# Hyperparameters tuned for XGBoost per report findings
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

joblib.dump(model, os.path.join(MODELS_DIR, 'xgb.pkl'))
print("✅ Step 4: XGBoost Model Trained (models/xgb.pkl)")
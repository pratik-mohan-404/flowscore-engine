import os
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# Load processed data
# ----------------------------

X_train, X_test, y_train, y_test = joblib.load(
    os.path.join(MODELS_DIR, "data.pkl")
)

# ----------------------------
# Build XGBoost Regressor
# ----------------------------

model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=5,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=42,

    objective="reg:squarederror"
)

# ----------------------------
# Train
# ----------------------------

model.fit(X_train, y_train)

# ----------------------------
# Quick sanity check
# ----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print(f"Validation MAE : {mae:.2f} FlowScore points")

# ----------------------------
# Save Model
# ----------------------------

joblib.dump(
    model,
    os.path.join(MODELS_DIR, "xgb.pkl")
)

print("✅ Step 4 Complete: XGBoost Regressor Trained")
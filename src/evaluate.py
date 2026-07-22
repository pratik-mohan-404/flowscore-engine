import os
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------
# Load Model and Data
# ----------------------------

X_train, X_test, y_train, y_test = joblib.load(
    os.path.join(MODELS_DIR, "data.pkl")
)

model = joblib.load(
    os.path.join(MODELS_DIR, "xgb.pkl")
)

# ----------------------------
# Predict FlowScores
# ----------------------------

y_pred = model.predict(X_test)

# Round to integer FlowScores
y_pred = y_pred.round().astype(int)

# Keep scores within 300-900
y_pred = y_pred.clip(300, 900)

# ----------------------------
# Evaluation Metrics
# ----------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print(" FLOWSCORE MODEL PERFORMANCE")
print("=" * 50)

print(f"\nMean Absolute Error : {mae:.2f}")
print(f"Root Mean Squared Error : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# ----------------------------
# Convert Scores to Categories
# ----------------------------

def category(score):
    if score < 450:
        return "High Risk"
    elif score < 600:
        return "Emerging"
    elif score < 750:
        return "Strong"
    else:
        return "Elite"

results = pd.DataFrame({
    "Actual FlowScore": y_test.values,
    "Predicted FlowScore": y_pred
})

results["Actual Category"] = results["Actual FlowScore"].apply(category)
results["Predicted Category"] = results["Predicted FlowScore"].apply(category)

print("\nFirst 15 Predictions\n")
print(results.head(15))

print("\nPredicted Category Distribution\n")
print(results["Predicted Category"].value_counts())

accuracy = (
    results["Actual Category"] ==
    results["Predicted Category"]
).mean() * 100

print(f"\nCategory Accuracy : {accuracy:.2f}%")

print("\nEvaluation Complete.")
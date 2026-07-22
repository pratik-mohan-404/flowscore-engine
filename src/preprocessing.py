import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Load Engineered Dataset
# ----------------------------

df = pd.read_csv(os.path.join(DATA_DIR, "upi_features.csv"))

# ----------------------------
# Features used by the model
# ----------------------------

features = [
    "transaction_consistency",
    "merchant_diversity",
    "income_stability",
    "savings_score",
    "temporal_score"
]

X = df[features]

# Target is now FlowScore
y = df["flowscore"]

# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------
# Feature Scaling
# ----------------------------

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=features
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=features
)

# ----------------------------
# Save processed data
# ----------------------------

joblib.dump(
    (X_train_scaled, X_test_scaled, y_train, y_test),
    os.path.join(MODELS_DIR, "data.pkl")
)

joblib.dump(
    scaler,
    os.path.join(MODELS_DIR, "scaler.pkl")
)

print("✅ Step 3 Complete: Data Preprocessed")
print(f"Training Samples : {len(X_train_scaled)}")
print(f"Testing Samples  : {len(X_test_scaled)}")
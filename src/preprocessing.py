import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, 'upi_features.csv'))

# Using the 5 engineered scores for the model
features = ["transaction_consistency", "merchant_diversity", "income_stability", "savings_score", "temporal_score"]
X = df[features]
y = df["creditworthy"]

# Split FIRST — then scale to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=features)

joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(MODELS_DIR, 'data.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print("✅ Step 3: Preprocessing Complete & Scaler Saved")
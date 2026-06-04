import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'upi_transactions.csv'))

# Feature Transformation Formulas from Report
df["transaction_consistency"] = df["avg_weekly_transactions"] / 50
df["merchant_diversity"]      = df["unique_merchants"] / 20
df["income_stability"]        = 1 - df["income_volatility"]
df["savings_score"]           = df["savings_ratio"]
df["temporal_score"]          = df["bill_payment_timing"]

df.to_csv(os.path.join(DATA_DIR, 'upi_features.csv'), index=False)
print("✅ Step 2: Feature Engineering Done (data/upi_features.csv)")
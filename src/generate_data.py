import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "vendor_id": range(1, n + 1),
    "avg_weekly_transactions": np.random.randint(1, 50, n),
    "unique_merchants": np.random.randint(1, 20, n),
    "monthly_inflow": np.random.randint(3000, 50000, n),
    "income_volatility": np.random.rand(n),
    "savings_ratio": np.random.rand(n),
    "bill_payment_timing": np.random.rand(n),
})

# Ground truth logic: Mapping raw features to creditworthiness
score = (
    (df["avg_weekly_transactions"] / 50) +
    (1 - df["income_volatility"]) +
    df["savings_ratio"] +
    df["bill_payment_timing"]
)
df["creditworthy"] = (score > 2.0).astype(int)

df.to_csv(os.path.join(DATA_DIR, 'upi_transactions.csv'), index=False)
print("✅ Step 1: Raw Data Generated (data/upi_transactions.csv)")
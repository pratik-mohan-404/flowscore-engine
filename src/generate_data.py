import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
n = 1000

# ----------------------------
# Generate Raw Vendor Data
# ----------------------------
df = pd.DataFrame({
    "vendor_id": range(1, n + 1),
    "avg_weekly_transactions": np.random.randint(5, 51, n),
    "unique_merchants": np.random.randint(2, 21, n),
    "monthly_inflow": np.random.randint(5000, 50001, n),
    "income_volatility": np.random.rand(n),
    "savings_ratio": np.random.rand(n),
    "bill_payment_timing": np.random.rand(n)
})

df.to_csv(os.path.join(DATA_DIR, "upi_transactions.csv"), index=False)

print("✅ Step 1 Complete: Raw vendor data generated.")
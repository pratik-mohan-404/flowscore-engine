import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "upi_transactions.csv"))

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

df["transaction_consistency"] = df["avg_weekly_transactions"] / 50
df["merchant_diversity"] = df["unique_merchants"] / 20
df["income_stability"] = 1 - df["income_volatility"]
df["savings_score"] = df["savings_ratio"]
df["temporal_score"] = df["bill_payment_timing"]

# --------------------------------------------------
# Behaviour Score (Weighted)
# --------------------------------------------------

df["behavior_score"] = (
      0.30 * df["transaction_consistency"]
    + 0.20 * df["merchant_diversity"]
    + 0.25 * df["income_stability"]
    + 0.15 * df["savings_score"]
    + 0.10 * df["temporal_score"]
)

# --------------------------------------------------
# Rank Vendors
# --------------------------------------------------

df = df.sort_values("behavior_score").reset_index(drop=True)

score = df["behavior_score"]

q1 = score.quantile(0.23)
q2 = score.quantile(0.50)
q3 = score.quantile(0.76)

flowscores = []

for s in score:

    if s <= q1:
        # High Risk
        low = 300
        high = 449
        t = (s - score.min()) / (q1 - score.min() + 1e-9)
        flowscore = low + t * (high - low)

    elif s <= q2:
        # Emerging
        low = 450
        high = 599
        t = (s - q1) / (q2 - q1 + 1e-9)
        flowscore = low + t * (high - low)

    elif s <= q3:
        # Strong
        low = 600
        high = 749
        t = (s - q2) / (q3 - q2 + 1e-9)
        flowscore = low + t * (high - low)

    else:
        # Elite
        low = 750
        high = 900
        t = (s - q3) / (score.max() - q3 + 1e-9)
        flowscore = low + t * (high - low)

    flowscores.append(round(flowscore))

df["flowscore"] = flowscores

# --------------------------------------------------
# Risk Categories
# --------------------------------------------------

conditions = [
    df["flowscore"] < 450,
    (df["flowscore"] >= 450) & (df["flowscore"] < 600),
    (df["flowscore"] >= 600) & (df["flowscore"] < 750),
    df["flowscore"] >= 750
]

choices = [
    "High Risk",
    "Emerging",
    "Strong",
    "Elite"
]

df["category"] = np.select(
    conditions,
    choices,
    default="Unknown"
)

# Restore Original Vendor Order
df = df.sort_values("vendor_id").reset_index(drop=True)

df.to_csv(os.path.join(DATA_DIR, "upi_features.csv"), index=False)

print("✅ Step 2 Complete: Feature Engineering & FlowScore Generated")

print("\nCategory Distribution:\n")
print(df["category"].value_counts().sort_index())

print("\nFlowScore Range:")
print(df["flowscore"].min(), "-", df["flowscore"].max())
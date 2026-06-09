from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

model  = joblib.load(os.path.join(MODELS_DIR, 'xgb.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

FEATURE_NAMES = [
    "transaction_consistency",
    "merchant_diversity",
    "income_stability",
    "savings_score",
    "temporal_score"
]

def calculate_features(vendor):
    awt = float(vendor["avg_weekly_transactions"])
    um  = float(vendor["unique_merchants"])
    mi  = float(vendor["monthly_inflow"])
    iv  = float(vendor["income_volatility"])
    sr  = float(vendor["savings_ratio"])
    bpt = float(vendor["bill_payment_timing"])

    eng = {
        "transaction_consistency": round(awt / 50, 4),
        "merchant_diversity":      round(um  / 20, 4),
        "income_stability":        round(1 - iv,   4),
        "savings_score":           round(sr,        4),
        "temporal_score":          round(bpt,       4)
    }
    raw = {
        "avg_weekly_transactions": round(awt, 2),
        "unique_merchants":        round(um,  2),
        "monthly_inflow":          round(mi,  2),
        "income_volatility":       round(iv,  3),
        "savings_ratio":           round(sr,  3),
        "bill_payment_timing":     round(bpt, 3)
    }
    return eng, raw

def score_vendor(vendor):
    eng, raw = calculate_features(vendor)
    X_raw    = pd.DataFrame([eng], columns=FEATURE_NAMES)
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=FEATURE_NAMES)
    prob     = float(model.predict_proba(X_scaled)[0][1])
    score    = round(300 + (prob * 600), 2)
    decision = "Creditworthy" if score >= 600 else "High Risk"
    return {
        "vendor_id":  int(vendor["vendor_id"]),
        "flowscore":  score,
        "prob":       round(prob, 4),
        "decision":   decision,
        "raw_values": raw,
        "engineered": eng,
        "metrics": {
            "Income stability":        round(eng["income_stability"]        * 100, 1),
            "Transaction consistency": round(eng["transaction_consistency"] * 100, 1),
            "Bill payment discipline": round(eng["temporal_score"]          * 100, 1),
            "Merchant diversity":      round(eng["merchant_diversity"]      * 100, 1),
            "Savings behaviour":       round(eng["savings_score"]           * 100, 1)
        }
    }

# ── Pre-compute all vendor scores on startup ───────────────
print("Pre-computing all vendor scores...")
_df            = pd.read_csv(os.path.join(DATA_DIR, 'upi_transactions.csv'))
ALL_SCORES     = [score_vendor(row) for _, row in _df.iterrows()]
ALL_SCORES_MAP = {v["vendor_id"]: v for v in ALL_SCORES}
print(f"Done — {len(ALL_SCORES)} vendors scored and cached.")

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data      = request.json
        vendor_id = int(data.get("vendor_id", -1))
        result    = ALL_SCORES_MAP.get(vendor_id)
        if not result:
            return jsonify({"error": "not_found"}), 404
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/filter", methods=["POST"])
def filter_vendors():
    try:
        data   = request.json
        filter = data.get("filter", "all")
        top_n  = int(data.get("top_n", 5))

        ranges = {
            "elite":    (750, 900),
            "strong":   (600, 749),
            "emerging": (450, 599),
            "poor":     (300, 449),
            "all":      (300, 900)
        }
        lo, hi   = ranges.get(filter, (300, 900))
        filtered = [v for v in ALL_SCORES if lo <= v["flowscore"] <= hi]
        sort_order = data.get("sort_order", "desc")
        filtered   = sorted(filtered, key=lambda x: x["flowscore"], reverse=(sort_order == "desc"))[:top_n]
        return jsonify({"vendors": filtered})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
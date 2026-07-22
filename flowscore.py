from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import shap

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# -------------------------------------------------------
# Load Model & Scaler
# -------------------------------------------------------

model = joblib.load(os.path.join(MODELS_DIR, "xgb.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

explainer = shap.TreeExplainer(model)

FEATURE_NAMES = [
    "transaction_consistency",
    "merchant_diversity",
    "income_stability",
    "savings_score",
    "temporal_score"
]

# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------

def calculate_features(vendor):

    awt = float(vendor["avg_weekly_transactions"])
    um = float(vendor["unique_merchants"])
    mi = float(vendor["monthly_inflow"])
    iv = float(vendor["income_volatility"])
    sr = float(vendor["savings_ratio"])
    bpt = float(vendor["bill_payment_timing"])

    engineered = {
        "transaction_consistency": round(awt / 50, 4),
        "merchant_diversity": round(um / 20, 4),
        "income_stability": round(1 - iv, 4),
        "savings_score": round(sr, 4),
        "temporal_score": round(bpt, 4)
    }

    raw = {
        "avg_weekly_transactions": round(awt, 2),
        "unique_merchants": round(um, 2),
        "monthly_inflow": round(mi, 2),
        "income_volatility": round(iv, 3),
        "savings_ratio": round(sr, 3),
        "bill_payment_timing": round(bpt, 3)
    }

    return engineered, raw


# -------------------------------------------------------
# Category Logic
# -------------------------------------------------------

def get_category(score):

    if score >= 750:
        return "Elite"

    elif score >= 600:
        return "Strong"

    elif score >= 450:
        return "Emerging"

    return "High Risk"



SHAP_EXPLANATIONS = {

    "Transaction Consistency": {
        "positive": "Consistent transaction activity strengthened the FlowScore.",
        "negative": "Irregular transaction activity reduced the FlowScore."
    },

    "Merchant Diversity": {
        "positive": "A diverse merchant base positively influenced the FlowScore.",
        "negative": "Limited merchant diversity reduced the FlowScore."
    },

    "Income Stability": {
        "positive": "Stable income patterns increased the FlowScore.",
        "negative": "Income volatility negatively affected the FlowScore."
    },

    "Savings Behaviour": {
        "positive": "Healthy savings behaviour improved the FlowScore.",
        "negative": "Low savings behaviour reduced the FlowScore."
    },

    "Payment Discipline": {
        "positive": "Timely bill payments strengthened the FlowScore.",
        "negative": "Delayed bill payments negatively impacted the FlowScore."
    }

}

ROADMAP_ACTIONS = {

    "Transaction Consistency": {
        "title": "Improve Transaction Consistency",
        "description": "Maintain regular UPI transactions throughout the month to demonstrate stable business activity."
    },

    "Merchant Diversity": {
        "title": "Expand Merchant Network",
        "description": "Increase the diversity of customers and merchants to reduce dependence on a limited customer base."
    },

    "Income Stability": {
        "title": "Stabilize Monthly Income",
        "description": "Aim for more consistent monthly cash inflows by reducing fluctuations in business revenue."
    },

    "Savings Behaviour": {
        "title": "Increase Savings",
        "description": "Maintain a higher savings ratio to improve financial resilience and repayment capacity."
    },

    "Payment Discipline": {
        "title": "Pay Bills on Time",
        "description": "Continue paying bills before their due dates to strengthen financial credibility."
    }

}



# -------------------------------------------------------
# Vendor Scoring
# -------------------------------------------------------

def score_vendor(vendor):

    engineered, raw = calculate_features(vendor)

    X = pd.DataFrame(
        [[engineered[f] for f in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=FEATURE_NAMES
    )

    # ------------------------------
    # XGBoost Regressor Prediction
    # ------------------------------

    predicted_score = float(model.predict(X_scaled)[0])

    # SHAP explanation
    shap_values = explainer.shap_values(X_scaled)

    predicted_score = int(round(predicted_score))
    predicted_score = max(300, min(900, predicted_score))

    category = get_category(predicted_score)

    # ------------------------------
    # SHAP Feature Contributions
    # ------------------------------

    shap_explanation = []

    feature_names_display = {

        "transaction_consistency": "Transaction Consistency",

        "merchant_diversity": "Merchant Diversity",

        "income_stability": "Income Stability",

        "savings_score": "Savings Behaviour",

        "temporal_score": "Payment Discipline"

    }


    for feature, value in zip(FEATURE_NAMES, shap_values[0]):

        impact = round(float(value), 2)

        display_name = feature_names_display[feature]

        effect = "positive" if impact > 0 else "negative"

        shap_explanation.append({

            "feature": display_name,

            "impact": impact,

            "effect": effect,

            "explanation": SHAP_EXPLANATIONS[display_name][effect]

        })


        shap_explanation = sorted(

            shap_explanation,

            key=lambda x: abs(x["impact"]),

            reverse=True

        )

        score_insights = {

            "positive": [
                x for x in shap_explanation
                if x["effect"] == "positive"
            ],

            "negative": [
                x for x in shap_explanation
                if x["effect"] == "negative"
            ]

        }   

        roadmap = []

        for item in shap_explanation:

            if item["effect"] == "negative":

                action = ROADMAP_ACTIONS[item["feature"]]

                roadmap.append({

                    "priority": len(roadmap) + 1,

                    "title": action["title"],

                    "description": action["description"]

                })


        result = {

            "vendor_id": int(vendor["vendor_id"]),

            "flowscore": predicted_score,

        # Compatibility fields
            "decision": category,
            "category": category,
            "prob": round((predicted_score - 300) / 600, 4),

            "raw_values": raw,

            "engineered": engineered,

            "shap": shap_explanation,

            "score_insights": score_insights,

            "roadmap": roadmap,

            "metrics": {

                "Income stability":
                    round(engineered["income_stability"] * 100, 1),

                "Transaction consistency":
                    round(engineered["transaction_consistency"] * 100, 1),

                "Bill payment discipline":
                    round(engineered["temporal_score"] * 100, 1),

                "Merchant diversity":
                    round(engineered["merchant_diversity"] * 100, 1),

                "Savings behaviour":
                    round(engineered["savings_score"] * 100, 1)
            }
        }

    shap_explanation = sorted(
    shap_explanation,
    key=lambda x: abs(x["impact"]),
    reverse=True
    )
    return result


# -------------------------------------------------------
# Cache Every Vendor
# -------------------------------------------------------

print("Pre-computing vendor FlowScores...")

df = pd.read_csv(
    os.path.join(DATA_DIR, "upi_transactions.csv")
)

ALL_SCORES = [
    score_vendor(row)
    for _, row in df.iterrows()
]

ALL_SCORES_MAP = {
    v["vendor_id"]: v
    for v in ALL_SCORES
}

print(f"Loaded {len(ALL_SCORES)} vendors into memory.")


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        vendor_id = int(data.get("vendor_id", -1))

        result = ALL_SCORES_MAP.get(vendor_id)

        if result is None:
            return jsonify({"error": "not_found"}), 404

        return jsonify(result)

    except Exception as e:

        print(e)

        return jsonify({
            "error": str(e)
        }), 500
    # -------------------------------------------------------
# Filter Vendors
# -------------------------------------------------------

@app.route("/filter", methods=["POST"])
def filter_vendors():

    try:

        data = request.json

        filter_name = data.get("filter", "all")
        top_n = int(data.get("top_n", 5))
        sort_order = data.get("sort_order", "desc")

        ranges = {

            "elite": (750, 900),

            "strong": (600, 749),

            "emerging": (450, 599),

            "poor": (300, 449),

            "all": (300, 900)

        }

        lo, hi = ranges.get(filter_name, (300, 900))

        filtered = [

            vendor

            for vendor in ALL_SCORES

            if lo <= vendor["flowscore"] <= hi

        ]

        filtered = sorted(

            filtered,

            key=lambda x: x["flowscore"],

            reverse=(sort_order == "desc")

        )

        filtered = filtered[:top_n]

        return jsonify({

            "vendors": filtered

        })

    except Exception as e:

        print(e)

        return jsonify({

            "error": str(e)

        }), 500


# -------------------------------------------------------
# Statistics
# -------------------------------------------------------

@app.route("/stats", methods=["GET"])
def stats():

    elite = len(

        [v for v in ALL_SCORES if v["flowscore"] >= 750]

    )

    strong = len(

        [v for v in ALL_SCORES

         if 600 <= v["flowscore"] < 750]

    )

    emerging = len(

        [v for v in ALL_SCORES

         if 450 <= v["flowscore"] < 600]

    )

    poor = len(

        [v for v in ALL_SCORES

         if v["flowscore"] < 450]

    )

    return jsonify({

        "elite": elite,

        "strong": strong,

        "emerging": emerging,

        "poor": poor,

        "total": len(ALL_SCORES)

    })


# -------------------------------------------------------
# Database
# -------------------------------------------------------

@app.route("/database", methods=["POST"])
def database():

    try:

        data = request.json

        search = str(

            data.get("search", "")

        ).strip()

        sort_by = data.get(

            "sort_by",

            "flowscore"

        )

        sort_order = data.get(

            "sort_order",

            "desc"

        )

        filter_cat = data.get(

            "filter",

            "all"

        )

        ranges = {

            "elite": (750, 900),

            "strong": (600, 749),

            "emerging": (450, 599),

            "poor": (300, 449),

            "all": (300, 900)

        }

        lo, hi = ranges.get(

            filter_cat,

            (300, 900)

        )

        results = [

            vendor

            for vendor in ALL_SCORES

            if lo <= vendor["flowscore"] <= hi

        ]

        if search:

            results = [

                vendor

                for vendor in results

                if str(vendor["vendor_id"]).startswith(search)

            ]

        if sort_by == "flowscore":

            results = sorted(

                results,

                key=lambda x: x["flowscore"],

                reverse=(sort_order == "desc")

            )

        elif sort_by == "vendor_id":

            results = sorted(

                results,

                key=lambda x: x["vendor_id"],

                reverse=(sort_order == "desc")

            )

        else:

            results = sorted(

                results,

                key=lambda x: x["raw_values"].get(sort_by, 0),

                reverse=(sort_order == "desc")

            )

        return jsonify({

            "vendors": results,

            "total": len(results)

        })

    except Exception as e:

        print(e)

        return jsonify({

            "error": str(e)

        }), 500


# -------------------------------------------------------
# Run Flask
# -------------------------------------------------------

if __name__ == "__main__":

    app.run(

        debug=True

    )
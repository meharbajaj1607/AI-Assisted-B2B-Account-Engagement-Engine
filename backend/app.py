"""
backend/app.py
--------------
Stateless Flask REST API for the B2B Account Engagement & Next-Best-Action Engine.

Integrates existing backend modules without duplicating their logic:
- ML Model & Feature Order: Loaded via joblib from models/best_model.pkl
  and models/feature_columns.pkl (trained by backend/train_model.py).
- Next-Best-Action Engine: Imported from backend/next_best_action.py (get_next_best_action).
- Gemini Insights Service: Imported from backend/gemini_service.py (generate_account_insights).
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, request

# Ensure imports from the backend/ directory resolve whether run from project root or backend/
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
from gemini_service import generate_account_insights
from next_best_action import get_next_best_action

app = Flask(__name__)

# Enable CORS headers on all responses so a frontend (React/Vite/HTML) can call the API directly
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# -----------------------------------------------------------------------------
# Path Resolution & Artifact Loading Helpers
# -----------------------------------------------------------------------------
def _resolve_path(*relative_candidates: str) -> Path:
    """Finds the first existing file path across project root and backend directory."""
    for rel in relative_candidates:
        for base in (PROJECT_ROOT, BACKEND_DIR, Path.cwd()):
            candidate = (base / rel).resolve()
            if candidate.exists():
                return candidate
    return (PROJECT_ROOT / relative_candidates[0]).resolve()


def _load_artifacts() -> Tuple[pd.DataFrame, Any, List[str]]:
    """Loads raw B2B accounts CSV, trained ML model, and locked feature column order."""
    data_path = _resolve_path("data/raw/b2b_accounts.csv")
    model_path = _resolve_path("models/best_model.pkl", "backend/models/best_model.pkl")
    features_path = _resolve_path("models/feature_columns.pkl", "backend/models/feature_columns.pkl")

    if not data_path.exists():
        raise FileNotFoundError(f"Accounts dataset not found at {data_path}. Run generate_dataset.py first.")
    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            f"Trained model artifacts not found at {model_path}. Run backend/train_model.py first."
        )

    df = pd.read_csv(data_path)
    model = joblib.load(str(model_path))
    feature_columns = list(joblib.load(str(features_path)))
    return df, model, feature_columns


def _score_accounts(df: pd.DataFrame, model: Any, feature_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Applies the exact preprocessing transformation (dropping identifiers, one-hot encoding
    industry, and aligning with feature_columns) and scores accounts using the loaded ML
    model and Next-Best-Action engine without duplicating their internal logic.
    """
    # Prepare features matching backend/preprocess.py
    df_features = df.drop(columns=["account_id", "company_name", "converted"], errors="ignore")
    df_encoded = pd.get_dummies(df_features, columns=["industry"], prefix="industry", dtype=int)
    X = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict probabilities and classes using the trained model
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        account_dict = row.to_dict()
        prob = round(float(probs[idx]), 4)
        pred_class = int(preds[idx])

        # Delegate priority & action logic to backend/next_best_action.py
        nba = get_next_best_action(account_dict, prob)

        records.append({
            **account_dict,
            "conversion_probability": prob,
            "predicted_converted": pred_class,
            "priority": nba["priority"],
            "recommended_action": nba["recommended_action"],
            "reason": nba["reason"],
        })

    return records


def _get_scored_account_by_id(account_id: str) -> Optional[Dict[str, Any]]:
    """Finds a single account by account_id and returns its scored record."""
    df, model, feature_columns = _load_artifacts()
    target_id = account_id.strip().upper()
    matched_df = df[df["account_id"].str.upper() == target_id].reset_index(drop=True)
    if matched_df.empty:
        return None
    scored_list = _score_accounts(matched_df, model, feature_columns)
    return scored_list[0]


# -----------------------------------------------------------------------------
# Core Health & Status Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "B2B Account Engagement Engine API",
        "status": "running",
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
    })


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/accounts", methods=["GET"])
def get_accounts():
    """
    GET /api/accounts
    Returns all B2B accounts enriched with ML conversion predictions and
    Next-Best-Action priorities. Supports optional ?priority= and ?industry= filters.
    """
    try:
        df, model, feature_columns = _load_artifacts()
        accounts = _score_accounts(df, model, feature_columns)

        # Optional query filters for convenience
        priority_filter = request.args.get("priority")
        industry_filter = request.args.get("industry")

        if priority_filter:
            accounts = [
                a for a in accounts if a["priority"].lower() == priority_filter.strip().lower()
            ]
        if industry_filter:
            accounts = [
                a for a in accounts if str(a["industry"]).lower() == industry_filter.strip().lower()
            ]

        return jsonify({
            "count": len(accounts),
            "accounts": accounts,
        }), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/accounts/<string:account_id>", methods=["GET"])
def get_account_detail(account_id: str):
    """
    GET /api/accounts/<id>
    Returns an individual account's details, ML prediction, and NBA recommendation.
    """
    try:
        account = _get_scored_account_by_id(account_id)
        if account is None:
            return jsonify({"error": f"Account '{account_id}' not found."}), 404

        return jsonify(account), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/accounts/<string:account_id>/insights", methods=["POST"])
def create_account_insights(account_id: str):
    """
    POST /api/accounts/<id>/insights
    Generates Gemini AI account summary, prioritization reasons, and a personalized
    outreach email while keeping the ML prediction and NBA recommendation locked.
    """
    try:
        account = _get_scored_account_by_id(account_id)
        if account is None:
            return jsonify({"error": f"Account '{account_id}' not found."}), 404

        nba_recommendation = {
            "priority": account["priority"],
            "recommended_action": account["recommended_action"],
            "reason": account["reason"],
        }

        # Delegate AI insight & email generation to backend/gemini_service.py
        insights = generate_account_insights(
            account_details=account,
            conversion_prob=account["conversion_probability"],
            nba_recommendation=nba_recommendation,
        )

        return jsonify({
            "account_id": account["account_id"],
            "company_name": account["company_name"],
            **insights,
        }), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/accounts/<string:account_id>/approval", methods=["POST"])
def approve_or_reject_outreach(account_id: str):
    """
    POST /api/accounts/<id>/approval
    Stateless human-in-the-loop endpoint to approve or reject recommended outreach.
    Accepts JSON body: {"decision": "approved" | "rejected", "notes": "...", "outreach_email": "..."}
    (or boolean {"approved": true | false}).
    """
    try:
        account = _get_scored_account_by_id(account_id)
        if account is None:
            return jsonify({"error": f"Account '{account_id}' not found."}), 404

        payload = request.get_json(silent=True) or {}

        # Normalize decision from either 'decision' ("approved"/"rejected") or boolean 'approved'
        raw_decision = payload.get("decision")
        if raw_decision is not None:
            decision_norm = str(raw_decision).strip().lower()
            if decision_norm in {"approve", "approved"}:
                decision = "approved"
            elif decision_norm in {"reject", "rejected"}:
                decision = "rejected"
            else:
                return jsonify({
                    "error": "Invalid 'decision'. Must be 'approved' or 'rejected'."
                }), 400
        elif "approved" in payload and isinstance(payload["approved"], bool):
            decision = "approved" if payload["approved"] else "rejected"
        else:
            return jsonify({
                "error": "Request body must include 'decision' ('approved'/'rejected') or boolean 'approved'."
            }), 400

        return jsonify({
            "status": "recorded",
            "account_id": account["account_id"],
            "company_name": account["company_name"],
            "decision": decision,
            "priority": account["priority"],
            "recommended_action": account["recommended_action"],
            "notes": payload.get("notes", ""),
            "outreach_email": payload.get("outreach_email"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
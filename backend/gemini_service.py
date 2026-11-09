"""
backend/gemini_service.py
-------------------------
Generates AI-powered B2B account insights and personalized outreach emails
using the Google GenAI SDK (google-genai) while strictly preserving the
ML conversion probability and Next-Best-Action (NBA) engine outputs.

Security & Architecture Principles:
1. Loads GEMINI_API_KEY from backend/.env via python-dotenv.
2. Never logs, prints, or exposes the API key in outputs or exceptions.
3. Gemini is strictly prohibited from altering the ML conversion probability
   or overriding the Next-Best-Action recommendation; those values are locked
   by the caller and only explained/translated into sales communication.
4. Gracefully handles missing API keys, missing SDK packages, and runtime API
   errors by returning a consistent, structured Python dictionary.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(dotenv_path=None, **_kwargs):
        """Lightweight .env loader fallback if python-dotenv is not installed."""
        if dotenv_path and os.path.exists(dotenv_path):
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return False


def _load_api_key() -> str:
    """Securely loads GEMINI_API_KEY from backend/.env without ever printing or exposing it."""
    backend_dir = Path(__file__).resolve().parent
    env_candidates = [
        backend_dir / ".env",
        Path.cwd() / "backend" / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break
    else:
        load_dotenv()

    return os.getenv("GEMINI_API_KEY", "").strip()


def _build_fallback_response(
    account: Dict[str, Any],
    conversion_prob: float,
    nba: Dict[str, str],
    status_note: str,
) -> Dict[str, Any]:
    """
    Builds a deterministic, structured dictionary when GEMINI_API_KEY is missing
    or an API error occurs so downstream callers never crash.
    """
    company = account.get("company_name", account.get("account_id", "Valued Prospect"))
    industry = account.get("industry", "B2B")
    visits = account.get("website_visits", 0)
    downloads = account.get("content_downloads", 0)
    demo = int(account.get("demo_requested", 0))
    days_inactive = account.get("days_since_activity", 0)
    prob_pct = f"{conversion_prob * 100:.1f}%"

    priority = nba.get("priority", "Medium")
    action = nba.get("recommended_action", "Schedule a sales call")
    reason = nba.get("reason", "Engagement signals align with outreach criteria.")

    return {
        "status": "fallback",
        "error": status_note,
        "conversion_probability": round(float(conversion_prob), 4),
        "priority": priority,
        "recommended_action": action,
        "account_summary": (
            f"{company} is a {industry} account with a predicted conversion probability "
            f"of {prob_pct} ({visits} website visits, {downloads} content downloads, "
            f"last active {days_inactive} days ago)."
        ),
        "prioritization_reasons": [
            f"Assigned '{priority}' priority with action: '{action}'.",
            reason,
            (
                "Explicitly requested a live product demo."
                if demo == 1
                else f"Recorded {visits} website visits and {downloads} content downloads."
            ),
        ],
        "outreach_email": (
            f"Subject: Helping {company} streamline operations\n\n"
            f"Hi {company} Team,\n\n"
            f"We noticed your team exploring our {industry} resources recently. "
            f"Based on your interest, we'd love to follow up on the next step: {action.lower()}.\n\n"
            f"Would you have 15 minutes this week to connect?\n\n"
            f"Best regards,\nAccount Executive Team"
        ),
    }


def generate_account_insights(
    account_details: Union[Dict[str, Any], Any],
    conversion_prob: float,
    nba_recommendation: Dict[str, str],
    model_name: str = "gemini-3.8-flash",
) -> Dict[str, Any]:
    """
    Generates an account summary, prioritization reasons, and a personalized outreach
    email using the google-genai SDK without altering the ML probability or NBA output.

    Parameters
    ----------
    account_details : dict or pandas.Series
        Account attributes (company_name, industry, emp_count, annual_revenue,
        tech_match, website_visits, email_opens, content_downloads,
        event_attended, demo_requested, days_since_activity).
    conversion_prob : float
        Predicted conversion probability in [0.0, 1.0] from the trained ML model.
    nba_recommendation : dict
        Output from get_next_best_action() containing:
        {"priority": str, "recommended_action": str, "reason": str}.
    model_name : str
        Gemini model identifier (default: "gemini-2.5-flash").

    Returns
    -------
    dict
        Structured dictionary with keys:
        - status ("success" or "fallback")
        - error (None or error message string)
        - conversion_probability (float, locked from ML model)
        - priority (str, locked from NBA engine)
        - recommended_action (str, locked from NBA engine)
        - account_summary (str)
        - prioritization_reasons (list[str])
        - outreach_email (str)
    """
    account = dict(account_details) if hasattr(account_details, "keys") else {}
    nba = dict(nba_recommendation) if hasattr(nba_recommendation, "keys") else {}

    # 1. Load GEMINI_API_KEY securely from backend/.env
    api_key = _load_api_key()
    if not api_key or api_key == "your_api_key_here":
        return _build_fallback_response(
            account=account,
            conversion_prob=conversion_prob,
            nba=nba,
            status_note="GEMINI_API_KEY is not set in backend/.env. Returned deterministic fallback insights.",
        )

    # 2. Import google-genai SDK safely
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError:
        return _build_fallback_response(
            account=account,
            conversion_prob=conversion_prob,
            nba=nba,
            status_note="google-genai package is not installed. Returned deterministic fallback insights.",
        )

    # 3. Construct prompt that strictly locks ML prediction and NBA recommendation
    prob_pct = f"{conversion_prob * 100:.1f}%"
    locked_priority = nba.get("priority", "Medium")
    locked_action = nba.get("recommended_action", "Schedule a sales call")
    locked_reason = nba.get("reason", "")

    system_instruction = (
        "You are a B2B Sales Intelligence Assistant. "
        "CRITICAL RULE: You must NEVER change, recalculate, or contradict the provided "
        "ML predicted conversion probability, priority level, or Next-Best-Action recommendation. "
        "Treat those values as fixed ground truth and only generate explanatory insights and "
        "an outreach email aligned with the prescribed action. "
        "Respond strictly in valid JSON matching the requested schema."
    )

    user_prompt = f"""
Generate B2B sales insights and a personalized outreach email for the following account.

[LOCKED ML & NEXT-BEST-ACTION OUTPUTS - DO NOT MODIFY OR OVERRIDE]
- Predicted Conversion Probability: {prob_pct} ({conversion_prob:.4f})
- Assigned Priority: {locked_priority}
- Next-Best-Action Recommendation: {locked_action}
- Engine Rule Reason: {locked_reason}

[ACCOUNT DETAILS]
{json.dumps(account, indent=2, default=str)}

Return a JSON object with exactly these keys:
1. "account_summary": A concise 2-3 sentence executive summary of the account's profile and engagement.
2. "prioritization_reasons": A JSON array of 3 concise bullet strings explaining why this account has "{locked_priority}" priority and why "{locked_action}" is the right next step.
3. "outreach_email": A ready-to-send, personalized B2B outreach email (including Subject line) tailored to execute "{locked_action}".
""".strip()

    # 4. Call Gemini API with structured JSON output and graceful error handling
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )

        raw_text = (response.text or "").strip()
        parsed = json.loads(raw_text)

        summary = str(parsed.get("account_summary", "")).strip()
        reasons = parsed.get("prioritization_reasons", [])
        if isinstance(reasons, str):
            reasons = [reasons]
        email_text = str(parsed.get("outreach_email", "")).strip()

        if not summary or not email_text:
            raise ValueError("Gemini response missed required JSON fields.")

        # Return structured dictionary with ML probability & NBA fields strictly locked
        return {
            "status": "success",
            "error": None,
            "conversion_probability": round(float(conversion_prob), 4),
            "priority": locked_priority,
            "recommended_action": locked_action,
            "account_summary": summary,
            "prioritization_reasons": [str(r) for r in reasons],
            "outreach_email": email_text,
        }

    except Exception as exc:
        # Scrub any accidental key substring from exception text before returning
        safe_err = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        return _build_fallback_response(
            account=account,
            conversion_prob=conversion_prob,
            nba=nba,
            status_note=f"Gemini API error ({type(exc).__name__}): {safe_err}",
        )


if __name__ == "__main__":
    # Demonstration with a sample account and Next-Best-Action recommendation
    sample_account = {
        "account_id": "ACC_0003",
        "company_name": "Nexus Dynamics 3",
        "industry": "FinTech",
        "emp_count": 327,
        "annual_revenue": 42150651.57,
        "tech_match": 0.91,
        "website_visits": 8,
        "email_opens": 11,
        "content_downloads": 2,
        "event_attended": 1,
        "demo_requested": 1,
        "days_since_activity": 34,
    }
    sample_prob = 0.806
    sample_nba = {
        "priority": "High",
        "recommended_action": "Prioritize immediate sales follow-up",
        "reason": (
            "Account explicitly requested a product demo (predicted conversion "
            "probability: 80.6%). Immediate outreach maximizes conversion while buying intent is active."
        ),
    }

    result = generate_account_insights(
        account_details=sample_account,
        conversion_prob=sample_prob,
        nba_recommendation=sample_nba,
    )
    print("=== Structured Output from generate_account_insights() ===")
    print(json.dumps(result, indent=2))
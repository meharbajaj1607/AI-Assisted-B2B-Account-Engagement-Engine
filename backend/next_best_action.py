"""
backend/next_best_action.py
---------------------------
Next-Best-Action (NBA) Recommendation Engine for B2B Sales & Marketing.

Combines an account's behavioral engagement features with the ML model's
predicted conversion probability to output:
1. Priority (High, Medium, Low)
2. Recommended action
3. Reason for recommendation

Rule Evaluation Hierarchy (Strict Priority Order):
1. Demo Requested (Checked FIRST) -> "Prioritize immediate sales follow-up" (High)
2. High Conversion Probability    -> "Schedule a sales call" (High)
3. Medium Prob + High Engagement  -> "Send a personalized case study" (Medium)
4. High Content Downloads         -> "Share relevant product information" (Medium)
5. Low Prob + Recent Inactivity   -> "Send a re-engagement email" (Low)
6. Default Nurture Fallback       -> "Enroll in targeted email nurture sequence" (Medium/Low)
"""
import os
from typing import Any, Dict, Union
import pandas as pd

def get_next_best_action(
    account_features: Union[Dict[str, Any], pd.Series],
    conversion_prob: float,
) -> Dict[str, str]:

    if not 0 <= conversion_prob <= 1:
        raise ValueError("conversion_prob must be between 0 and 1")

    # Rest of your function starts here
    features = dict(account_features)
    """
    Determines the Next-Best-Action for a B2B account based on its engagement
    features and predicted conversion probability.

    Parameters
    ----------
    account_features : dict or pd.Series
        Feature values for a single B2B account (e.g., demo_requested,
        website_visits, content_downloads, email_opens, event_attended,
        days_since_activity, tech_match).
    conversion_prob : float
        Predicted probability of conversion in [0.0, 1.0] from the trained ML model.

    Returns
    -------
    dict
        {
            "priority": "High" | "Medium" | "Low",
            "recommended_action": str,
            "reason": str
        }
    """
    # Normalize input to a dictionary so both dict and pandas Series work seamlessly
    features = dict(account_features)

    # Extract key behavioral signals with safe defaults
    demo_requested = int(features.get("demo_requested", 0))
    website_visits = int(features.get("website_visits", 0))
    content_downloads = int(features.get("content_downloads", 0))
    email_opens = int(features.get("email_opens", 0))
    event_attended = int(features.get("event_attended", 0))
    days_since_activity = int(features.get("days_since_activity", 0))
    tech_match = float(features.get("tech_match", 0.0))

    prob_pct = f"{conversion_prob * 100:.1f}%"

    # -------------------------------------------------------------------------
    # Rule 1 (HIGHEST PRIORITY): Explicit Demo Request
    # Checked first because a prospect raising their hand for a live demo
    # must receive immediate sales follow-up regardless of model score.
    # -------------------------------------------------------------------------
    if demo_requested == 1:
        return {
            "priority": "High",
            "recommended_action": "Prioritize immediate sales follow-up",
            "reason": (
                f"Account explicitly requested a product demo (predicted conversion "
                f"probability: {prob_pct}). Immediate outreach maximizes conversion "
                f"while buying intent is active."
            ),
        }

    # -------------------------------------------------------------------------
    # Rule 2: High Conversion Probability (>= 70%)
    # Strong overall profile and engagement warrant direct sales outreach.
    # -------------------------------------------------------------------------
    if conversion_prob >= 0.70:
        return {
            "priority": "High",
            "recommended_action": "Schedule a sales call",
            "reason": (
                f"High predicted conversion probability ({prob_pct}) backed by strong "
                f"engagement ({website_visits} website visits, {content_downloads} content "
                f"downloads, tech match {tech_match:.2f}). Ready for direct sales qualification."
            ),
        }

    # -------------------------------------------------------------------------
    # Rule 3: Medium Probability (40% - 69%) + High Engagement
    # Active prospects who haven't requested a demo yet benefit from social proof.
    # -------------------------------------------------------------------------
    is_highly_engaged = (
        website_visits >= 12
        or email_opens >= 8
        or event_attended == 1
    )
    if 0.40 <= conversion_prob < 0.70 and is_highly_engaged:
        return {
            "priority": "Medium",
            "recommended_action": "Send a personalized case study",
            "reason": (
                f"Medium conversion probability ({prob_pct}) combined with high engagement "
                f"({website_visits} visits, {email_opens} email opens, event_attended={event_attended}). "
                f"Sharing an industry-specific case study can build trust and nudge a demo request."
            ),
        }

    # -------------------------------------------------------------------------
    # Rule 4: High Content Downloads (>= 4 downloads)
    # Prospects researching heavily need deeper technical/product collateral.
    # -------------------------------------------------------------------------
    if content_downloads >= 4:
        priority = "Medium" if conversion_prob >= 0.30 else "Low"
        return {
            "priority": priority,
            "recommended_action": "Share relevant product information",
            "reason": (
                f"Account has downloaded {content_downloads} content resources "
                f"(predicted conversion probability: {prob_pct}), signaling active "
                f"product research and evaluation."
            ),
        }

    # -------------------------------------------------------------------------
    # Rule 5: Low Probability (< 40%) + Recent Inactivity (>= 30 days)
    # Cold/dormant accounts need a lightweight automated re-engagement touch.
    # -------------------------------------------------------------------------
    if conversion_prob < 0.40 and days_since_activity >= 30:
        return {
            "priority": "Low",
            "recommended_action": "Send a re-engagement email",
            "reason": (
                f"Low conversion probability ({prob_pct}) and no recorded activity "
                f"for {days_since_activity} days. A re-engagement campaign can test "
                f"whether interest can be revived without spending sales rep time."
            ),
        }

    # -------------------------------------------------------------------------
    # Rule 6: Default / Baseline Nurture Recommendation
    # Covers remaining moderate- or low-probability accounts that are still active.
    # -------------------------------------------------------------------------
    if conversion_prob >= 0.40:
        return {
            "priority": "Medium",
            "recommended_action": "Send a personalized case study",
            "reason": (
                f"Moderate conversion probability ({prob_pct}) with recent activity "
                f"({days_since_activity} days ago). Targeted educational content will "
                f"help deepen engagement."
            ),
        }

    return {
        "priority": "Low",
        "recommended_action": "Send a re-engagement email",
        "reason": (
            f"Low conversion probability ({prob_pct}) and limited engagement signals "
            f"({website_visits} visits, {content_downloads} downloads). Keep in "
            f"automated marketing nurture until engagement increases."
        ),
    }


def main() -> None:
    """Demonstrates and validates the Next-Best-Action engine on sample profiles and test data."""
    print("=== 1. Rule Validation Across Target Scenarios ===")
    sample_scenarios = [
        (
            "Scenario A: Demo Requested (Even with moderate probability)",
            {
                "demo_requested": 1,
                "website_visits": 9,
                "content_downloads": 2,
                "email_opens": 4,
                "event_attended": 0,
                "days_since_activity": 12,
                "tech_match": 0.65,
            },
            0.52,
        ),
        (
            "Scenario B: High Conversion Probability (No Demo Yet)",
            {
                "demo_requested": 0,
                "website_visits": 20,
                "content_downloads": 6,
                "email_opens": 11,
                "event_attended": 1,
                "days_since_activity": 3,
                "tech_match": 0.88,
            },
            0.81,
        ),
        (
            "Scenario C: Medium Probability + High Engagement",
            {
                "demo_requested": 0,
                "website_visits": 15,
                "content_downloads": 2,
                "email_opens": 9,
                "event_attended": 1,
                "days_since_activity": 10,
                "tech_match": 0.62,
            },
            0.55,
        ),
        (
            "Scenario D: High Content Downloads",
            {
                "demo_requested": 0,
                "website_visits": 8,
                "content_downloads": 5,
                "email_opens": 4,
                "event_attended": 0,
                "days_since_activity": 18,
                "tech_match": 0.50,
            },
            0.36,
        ),
        (
            "Scenario E: Low Probability + Recent Inactivity",
            {
                "demo_requested": 0,
                "website_visits": 4,
                "content_downloads": 1,
                "email_opens": 2,
                "event_attended": 0,
                "days_since_activity": 65,
                "tech_match": 0.30,
            },
            0.14,
        ),
    ]

    for label, feats, prob in sample_scenarios:
        rec = get_next_best_action(feats, prob)
        print(f"\n{label}")
        print(f"  -> Priority:           {rec['priority']}")
        print(f"  -> Recommended Action: {rec['recommended_action']}")
        print(f"  -> Reason:             {rec['reason']}")

    # Optional: Run on first 5 test accounts using the saved model if available
    x_test_path = os.path.join("data", "processed", "X_test.csv")
    model_path = os.path.join("backend", "models", "best_model.pkl")
    
if __name__ == "__main__":
    main()
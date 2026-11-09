"""
generate_dataset.py
-------------------
Generates a synthetic B2B customer conversion dataset (1,000 company records)
for training and evaluating binary classification models.

Author Level: 3rd-Year Undergraduate Data Science / ML Project
Key ML Design Choices:
1. Fixed Random Seed: Ensures 100% reproducibility across runs.
2. Zero Target Leakage: Input features are generated strictly from firmographic
   and pre-conversion behavioral distributions before calculating the target.
   Identifier columns (account_id, company_name) contain no target signal.
3. Latent Log-Odds + Sigmoid + Noise (Logistic Generative Process): Instead of hard
   if-else rules (which make tree models trivially memorize thresholds), we compute
   a continuous latent score z, inject Gaussian noise (to simulate unobserved real-world
   factors / irreducible error), pass z through the sigmoid function to get a
   conversion probability p, and sample the binary target via a Bernoulli trial.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Standard logistic sigmoid function to map log-odds to probabilities in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def generate_b2b_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generates a realistic synthetic B2B account dataset with logical feature relationships."""
    # Set fixed random seed so every run produces the exact same dataset
    np.random.seed(seed)

    # -------------------------------------------------------------------------
    # 1. Identifier & Firmographic Features
    # -------------------------------------------------------------------------
    # Unique account IDs and synthetic company names (pure identifiers, drop before ML training)
    account_id = [f"ACC_{i:04d}" for i in range(1, n_samples + 1)]

    prefixes = [
        "Apex", "Vertex", "Nova", "Quantum", "Pulse", "Stratos", "Vanguard",
        "Nexus", "Horizon", "Catalyst", "Summit", "Zenith", "Prism", "Vector"
    ]
    suffixes = [
        "Systems", "Solutions", "Technologies", "Labs", "Dynamics", "Analytics",
        "Group", "Cloud", "Works", "Ventures", "Corp", "Partners"
    ]
    company_name = [
        f"{np.random.choice(prefixes)} {np.random.choice(suffixes)} {i}"
        for i in range(1, n_samples + 1)
    ]

    # B2B industry verticals with realistic market representation
    industries = ["SaaS", "FinTech", "Healthcare", "Manufacturing", "E-Commerce", "Logistics"]
    industry_probs = [0.28, 0.20, 0.16, 0.14, 0.12, 0.10]
    industry = np.random.choice(industries, size=n_samples, p=industry_probs)

    # Employee count follows a right-skewed log-normal distribution (most B2B prospects
    # are SMB/mid-market, with fewer large enterprises), clipped between 10 and 10,000.
    emp_count = np.clip(
        np.random.lognormal(mean=4.8, sigma=1.1, size=n_samples).astype(int),
        10,
        10000,
    )

    # Annual revenue (in USD) scales logically with employee count (~$120k-$250k per employee)
    # plus multiplicative log-normal variance so revenue isn't purely deterministic.
    revenue_per_emp = np.random.lognormal(mean=12.0, sigma=0.35, size=n_samples)
    annual_revenue = np.round(emp_count * revenue_per_emp, 2)

    # Tech match score (0.0 to 1.0): measures how well the prospect's current tech stack
    # integrates with our product. SaaS and FinTech tend to have slightly higher tech compatibility.
    base_tech_match = np.random.beta(a=2.5, b=2.2, size=n_samples)
    industry_tech_boost = np.where(np.isin(industry, ["SaaS", "FinTech"]), 0.10, 0.0)
    tech_match = np.round(np.clip(base_tech_match + industry_tech_boost, 0.05, 1.00), 2)

    # -------------------------------------------------------------------------
    # 2. Behavioral Engagement Features (Logical Co-variation, No Target Leakage)
    # -------------------------------------------------------------------------
    # Create an unobserved "underlying interest factor" (0.5 to 2.0) driven partly by tech_match
    # so engaged companies naturally show correlated behavior across channels.
    interest_factor = 0.6 + 0.9 * tech_match + np.random.gamma(shape=2.0, scale=0.25, size=n_samples)

    # Website visits: Poisson-distributed around interest_factor
    website_visits = np.random.poisson(lam=8.0 * interest_factor, size=n_samples)

    # Email opens: correlated with overall engagement
    email_opens = np.random.poisson(lam=4.5 * interest_factor, size=n_samples)

    # Content downloads (whitepapers, case studies): higher when website visits are higher
    content_downloads = np.random.poisson(
        lam=1.8 * interest_factor + 0.08 * website_visits, size=n_samples
    )

    # Event attended (0 or 1): probability increases with email opens and interest
    event_prob = sigmoid(-1.8 + 0.12 * email_opens + 0.8 * tech_match)
    event_attended = np.random.binomial(n=1, p=event_prob, size=n_samples)

    # Days since last activity: more engaged prospects typically visited more recently.
    # Drawn from an exponential distribution inversely scaled by interest_factor, clipped to [0, 180].
    mean_days_inactive = np.clip(45.0 / interest_factor, 5.0, 90.0)
    days_since_activity = np.clip(
        np.random.exponential(scale=mean_days_inactive, size=n_samples).astype(int),
        0,
        180,
    )

    # Demo requested (0 or 1): prospects who visit the site often, download content,
    # and have recent activity are much more likely to request a live product demo.
    demo_logit = (
        -2.6
        + 0.07 * website_visits
        + 0.25 * content_downloads
        + 0.60 * event_attended
        + 0.90 * tech_match
        - 0.025 * days_since_activity
    )
    demo_requested = np.random.binomial(n=1, p=sigmoid(demo_logit), size=n_samples)

    # -------------------------------------------------------------------------
    # 3. Target Variable Generation (Probabilistic Logistic Formula)
    # -------------------------------------------------------------------------
    # Important ML Requirement:
    # We generate 'converted' using a well-calibrated logistic model with added
    # Gaussian noise (epsilon ~ N(0, 0.65^2)). This prevents a deterministic linear
    # boundary, mimics real-world unobserved variables (budget freezes, competitor offers),
    # and ensures realistic precision/recall trade-offs during ML model training.
    noise = np.random.normal(loc=0.0, scale=0.65, size=n_samples)

    log_odds = (
        -3.10                                  # Baseline intercept (keeps conversion rate realistic ~25-35%)
        + 1.85 * demo_requested                # Demo requested -> significantly higher conversion probability
        + 0.08 * website_visits                # More website visits -> higher conversion probability
        + 0.28 * content_downloads             # More content downloads -> higher conversion probability
        - 0.045 * days_since_activity          # Recent activity (fewer days) -> higher conversion probability
        + 0.05 * email_opens                   # Email engagement -> modest positive lift
        + 0.55 * event_attended                # Webinar/event attendance -> solid positive lift
        + 1.10 * tech_match                    # Strong product/tech fit -> higher conversion probability
        + 0.12 * np.log10(annual_revenue / 1e6 + 1)  # Slight lift for well-resourced accounts
        + noise                                # Unobserved stochastic noise
    )

    conversion_prob = sigmoid(log_odds)
    converted = np.random.binomial(n=1, p=conversion_prob, size=n_samples)

    # Assemble into a clean DataFrame with the exact required feature order
    df = pd.DataFrame({
        "account_id": account_id,
        "company_name": company_name,
        "industry": industry,
        "emp_count": emp_count,
        "annual_revenue": annual_revenue,
        "tech_match": tech_match,
        "website_visits": website_visits,
        "email_opens": email_opens,
        "content_downloads": content_downloads,
        "event_attended": event_attended,
        "demo_requested": demo_requested,
        "days_since_activity": days_since_activity,
        "converted": converted,
    })

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Runs data quality and ML readiness checks on the generated dataset."""
    # 1. Row and column count check
    assert df.shape == (1000, 13), f"Expected shape (1000, 13), got {df.shape}"

    # 2. Missing value check
    assert df.isnull().sum().sum() == 0, "Dataset contains missing (NaN) values!"

    # 3. Duplicate account_id check
    assert df["account_id"].nunique() == 1000, "account_id values must be unique!"

    # 4. Data type & domain range checks
    assert pd.api.types.is_integer_dtype(df["emp_count"]) and (df["emp_count"] > 0).all()
    assert pd.api.types.is_float_dtype(df["annual_revenue"]) and (df["annual_revenue"] > 0).all()
    assert df["tech_match"].between(0.0, 1.0).all(), "tech_match must be in [0, 1]"
    assert (df["website_visits"] >= 0).all(), "website_visits cannot be negative"
    assert (df["content_downloads"] >= 0).all(), "content_downloads cannot be negative"
    assert (df["days_since_activity"] >= 0).all(), "days_since_activity cannot be negative"
    assert set(df["demo_requested"].unique()).issubset({0, 1}), "demo_requested must be binary (0/1)"

    # 5. Target class presence and ML class balance check (avoid extreme imbalance)
    unique_classes = set(df["converted"].unique())
    assert unique_classes == {0, 1}, f"Target must contain both classes 0 and 1, found {unique_classes}"
    pos_rate = df["converted"].mean()
    assert 0.15 <= pos_rate <= 0.50, (
        f"ML Requirement Failed: Positive conversion rate ({pos_rate:.2%}) is outside the "
        "healthy training range [15%, 50%]."
    )

    # 6. Logical directional relationship verification (ML sanity check)
    corr_visits = df["website_visits"].corr(df["converted"])
    corr_demo = df["demo_requested"].corr(df["converted"])
    corr_downloads = df["content_downloads"].corr(df["converted"])
    corr_recency = df["days_since_activity"].corr(df["converted"])

    assert corr_visits > 0.15, f"Expected positive correlation for website_visits, got {corr_visits:.3f}"
    assert corr_demo > 0.30, f"Expected strong positive correlation for demo_requested, got {corr_demo:.3f}"
    assert corr_downloads > 0.15, f"Expected positive correlation for content_downloads, got {corr_downloads:.3f}"
    assert corr_recency < -0.15, f"Expected negative correlation for days_since_activity, got {corr_recency:.3f}"


def main() -> None:
    # Generate dataset
    df = generate_b2b_dataset(n_samples=1000, seed=42)

    # Validate dataset integrity and ML requirements before saving
    validate_dataset(df)

    # Ensure output directory exists and save to data/raw/b2b_accounts.csv
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "raw" / "b2b_accounts.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print required summary diagnostics
    print("=== 1. Dataset Shape ===")
    print(df.shape)

    print("\n=== 2. First Five Records ===")
    print(df.head(5).to_string(index=False))

    print("\n=== 3. Missing-Value Count ===")
    print(df.isnull().sum())

    print("\n=== 4. Conversion Class Distribution ===")
    counts = df["converted"].value_counts().sort_index()
    pcts = (df["converted"].value_counts(normalize=True).sort_index() * 100).round(2)
    dist_df = pd.DataFrame({"count": counts, "percentage (%)": pcts})
    print(dist_df)

    print("\n=== 5. Data Types ===")
    print(df.dtypes)

    print(f"\n[SUCCESS] Validated dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
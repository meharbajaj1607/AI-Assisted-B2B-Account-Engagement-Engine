"""
explore_dataset.py
------------------
Exploratory Data Analysis (EDA) script for the synthetic B2B conversion dataset.
Loads data/raw/b2b_accounts.csv and inspects summary statistics, target class
balance, industry-level conversion rates, and numerical feature correlations.
"""

import os
import pandas as pd


def main() -> None:
    # Configure Pandas display options so all columns are visible in terminal output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # -------------------------------------------------------------------------
    # 1. Load data/raw/b2b_accounts.csv using Pandas
    # -------------------------------------------------------------------------
    data_path = os.path.join("data", "raw", "b2b_accounts.csv")
    df = pd.read_csv(data_path)
    print(f"[INFO] Successfully loaded dataset from: {data_path}\n")

    # -------------------------------------------------------------------------
    # 2. Display dataset shape and column names
    # -------------------------------------------------------------------------
    print("=== Step 2: Dataset Shape & Column Names ===")
    print(f"Shape (rows, columns): {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}\n")

    # -------------------------------------------------------------------------
    # 3. Display df.describe()
    # -------------------------------------------------------------------------
    print("=== Step 3: Summary Statistics (df.describe()) ===")
    print(df.describe().round(2))
    print()

    # -------------------------------------------------------------------------
    # 4. Display the conversion count and percentage
    # -------------------------------------------------------------------------
    print("=== Step 4: Conversion Count & Percentage ===")
    conv_counts = df["converted"].value_counts().sort_index()
    conv_pct = (df["converted"].value_counts(normalize=True).sort_index() * 100).round(2)
    conversion_summary = pd.DataFrame({
        "count": conv_counts,
        "percentage (%)": conv_pct,
    })
    conversion_summary.index.name = "converted"
    print(conversion_summary)
    print()

    # -------------------------------------------------------------------------
    # 5. Calculate conversion rates by industry
    # -------------------------------------------------------------------------
    print("=== Step 5: Conversion Rates by Industry ===")
    # Group by industry to see total accounts, converted accounts, and conversion rate (%)
    industry_conversion = (
        df.groupby("industry")["converted"]
        .agg(
            total_accounts="count",
            converted_accounts="sum",
            conversion_rate_pct=lambda x: round(x.mean() * 100, 2),
        )
        .sort_values(by="conversion_rate_pct", ascending=False)
    )
    print(industry_conversion)
    print()

    # -------------------------------------------------------------------------
    # 6. Display the correlation matrix for numerical columns
    # -------------------------------------------------------------------------
    print("=== Step 6: Correlation Matrix (Numerical Columns) ===")
    # Select only numeric columns (int64, float64) before computing Pearson correlation
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr().round(3)
    print(corr_matrix)


if __name__ == "__main__":
    main()
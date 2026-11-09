"""
backend/preprocess.py
---------------------
Preprocesses the raw B2B customer conversion dataset (data/raw/b2b_accounts.csv)
for machine learning model training:
1. Drops non-predictive identifier columns (account_id, company_name).
2. One-hot encodes the categorical 'industry' feature.
3. Separates features (X) and target label (y = converted).
4. Splits data 80:20 into training and testing sets using stratified sampling
   (stratify=y, random_state=42) to preserve class distribution.
5. Saves X_train.csv, X_test.csv, y_train.csv, and y_test.csv to data/processed/.
"""

import os
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    def train_test_split(X, y, test_size=0.20, stratify=None, random_state=42):
        """Stratified train/test split fallback when scikit-learn is not installed."""
        rng = np.random.default_rng(random_state)
        train_idx, test_idx = [], []
        for cls in np.unique(stratify):
            cls_indices = np.where(stratify == cls)[0]
            rng.shuffle(cls_indices)
            n_test = int(round(len(cls_indices) * test_size))
            test_idx.extend(cls_indices[:n_test])
            train_idx.extend(cls_indices[n_test:])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        return (
            X.iloc[train_idx].reset_index(drop=True),
            X.iloc[test_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            y.iloc[test_idx].reset_index(drop=True),
        )


def preprocess_data(
    raw_data_path: str = os.path.join("data", "raw", "b2b_accounts.csv"),
    processed_dir: str = os.path.join("data", "processed"),
) -> None:
    """Loads raw B2B data, preprocesses features, performs an 80:20 stratified split, and saves outputs."""
    # 1. Load the raw dataset
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_data_path}")

    df = pd.read_csv(raw_data_path)
    print(f"[INFO] Loaded raw dataset from {raw_data_path} with shape: {df.shape}")

    # 2. Drop identifier columns that carry no predictive signal and could cause overfitting
    cols_to_drop = ["account_id", "company_name"]
    df_cleaned = df.drop(columns=cols_to_drop)
    print(f"[INFO] Dropped identifier columns: {cols_to_drop}")

    # 3. One-hot encode the categorical 'industry' column
    # Using dtype=int so encoded columns are clean 0/1 integers ready for ML models
    df_encoded = pd.get_dummies(df_cleaned, columns=["industry"], prefix="industry", dtype=int)
    print(f"[INFO] One-hot encoded 'industry'. New feature count: {df_encoded.shape[1] - 1}")

    # 4. Separate features (X) and target variable (y)
    X = df_encoded.drop(columns=["converted"])
    y = df_encoded["converted"]

    # 5. Split data 80:20 using train_test_split with stratify=y and random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # 6. Save processed splits to data/processed/
    os.makedirs(processed_dir, exist_ok=True)

    x_train_path = os.path.join(processed_dir, "X_train.csv")
    x_test_path = os.path.join(processed_dir, "X_test.csv")
    y_train_path = os.path.join(processed_dir, "y_train.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    # Print verification summary
    print("\n=== Preprocessing & Split Summary ===")
    print(f"X_train shape: {X_train.shape} | Positive rate: {y_train.mean():.2%}")
    print(f"X_test shape:  {X_test.shape}  | Positive rate: {y_test.mean():.2%}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"\nFeatures ({len(X.columns)}): {list(X.columns)}")
    print(f"\n[SUCCESS] Saved X_train, X_test, y_train, and y_test to: {processed_dir}/")


if __name__ == "__main__":
    preprocess_data()
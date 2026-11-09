"""
backend/train_model.py
----------------------
Trains, evaluates, and compares three classification models on the processed
B2B customer conversion dataset:
1. Logistic Regression (with StandardScaler pipeline)
2. Random Forest Classifier
3. Gradient Boosting Classifier

Evaluates each model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC,
verifies feature column alignment between X_train and X_test, and saves the
best-performing model using joblib.
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_processed_data(processed_dir: str = os.path.join("data", "processed")):
    """Loads X_train, X_test, y_train, and y_test and validates feature column alignment."""
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze("columns")

    # Important ML Check: Verify X_train and X_test have the exact same columns in the same order
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Feature mismatch between X_train and X_test! "
            f"X_train columns: {list(X_train.columns)} vs X_test columns: {list(X_test.columns)}"
        )

    print(f"[CHECK PASSED] X_train {X_train.shape} and X_test {X_test.shape} have identical feature order.")
    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    """Trains Logistic Regression, Random Forest, and Gradient Boosting, compares metrics, and saves the best model."""
    # 1. Load preprocessed train and test sets
    X_train, X_test, y_train, y_test = load_processed_data()

    # 2. Define the 3 candidate classification models
    # Note: Logistic Regression uses StandardScaler in a Pipeline so large-scale features
    # (like annual_revenue) don't dominate smaller features (like tech_match).
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    # 3. Train and evaluate each model on the held-out test set
    results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

        # Generate class predictions (0/1) and positive conversion probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute all 5 required evaluation metrics
        metrics = {
            "Model": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        }
        results.append(metrics)

    # 4. Compare models in a clean table sorted by ROC-AUC (and F1-Score as tie-breaker)
    results_df = pd.DataFrame(results).sort_values(
        by=["ROC-AUC", "F1-Score"], ascending=False
    ).reset_index(drop=True)

    print("\n=== Model Evaluation Comparison (Test Set) ===")
    print(results_df.to_string(index=False))

    # 5. Select the best model
    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]
    best_roc_auc = results_df.loc[0, "ROC-AUC"]
    best_f1 = results_df.loc[0, "F1-Score"]

    print(
        f"\n[BEST MODEL SELECTED] {best_model_name} "
        f"(ROC-AUC: {best_roc_auc:.4f}, F1-Score: {best_f1:.4f})"
    )

    # 6. Save the best model (and feature column order metadata) using joblib
    # Saved to both models/ and backend/models/ so scripts running from root or backend/ find it
    for model_dir in ["models", os.path.join("backend", "models")]:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "best_model.pkl")
        features_path = os.path.join(model_dir, "feature_columns.pkl")
        joblib.dump(best_model, model_path)
        joblib.dump(list(X_train.columns), features_path)

    print("[SUCCESS] Saved best model to: models/best_model.pkl (and backend/models/best_model.pkl)")


if __name__ == "__main__":
    train_and_evaluate()
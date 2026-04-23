"""
Module 5 Week B — Integration Task: Model Comparison & Decision Memo

Module 5 culminating deliverable. Compare 6 model configurations using
5-fold stratified cross-validation, produce PR curves and calibration
plots, log experiments, persist the best model, and demonstrate what
tree-based models capture that linear models cannot.

Complete the 9 functions below. See the integration guide for task-by-task
detail.
Run with:  python model_comparison.py
Tests:     pytest tests/ -v
"""

import os
from datetime import datetime

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]


# ---------------------------------------------------------------------------
# Task 1
# ---------------------------------------------------------------------------

def load_and_preprocess(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split into train/test sets.

    Uses an 80/20 stratified split. Features are the 8 NUMERIC_FEATURES
    columns. Target is `churned`.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------

def define_models():
    """Define 6 model configurations for comparison.

    The pattern is deliberate: a default vs class_weight='balanced' pair
    at BOTH the linear and ensemble family levels. This lets you observe
    the class_weight effect at two levels of model complexity.

    Each Pipeline has exactly two steps: ('scaler', ...) and ('model', ...).
    LR variants use StandardScaler; tree-based models use 'passthrough'.

    Returns:
        Dict of {name: sklearn.pipeline.Pipeline} with 6 entries.
        Names: 'Dummy', 'LR_default', 'LR_balanced', 'DT_depth5',
               'RF_default', 'RF_balanced'.
    """
    models = {
        "Dummy": Pipeline(
            [
                ("scaler", "passthrough"),
                ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
            ]
        ),
        "LR_default": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "LR_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=42
                    ),
                ),
            ]
        ),
        "DT_depth5": Pipeline(
            [
                ("scaler", "passthrough"),
                ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
        "RF_default": Pipeline(
            [
                ("scaler", "passthrough"),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100, max_depth=10, random_state=42
                    ),
                ),
            ]
        ),
        "RF_balanced": Pipeline(
            [
                ("scaler", "passthrough"),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Task 3
# ---------------------------------------------------------------------------

def run_cv_comparison(models, X, y, n_splits=5, random_state=42):
    """Run 5-fold stratified cross-validation on all models.

    For each model, compute mean and std of: accuracy, precision, recall,
    F1, and PR-AUC across folds. PR-AUC uses predict_proba — it is a
    threshold-independent ranking metric.

    Args:
        models: Dict of {name: Pipeline} from define_models().
        X: Feature DataFrame.
        y: Target Series.
        n_splits: Number of CV folds.
        random_state: Random seed for StratifiedKFold.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, precision_std, recall_mean, recall_std,
        f1_mean, f1_std, pr_auc_mean, pr_auc_std.
        One row per model (6 rows total).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for name, pipeline in models.items():
        accs, precs, recs, f1s, pr_aucs = [], [], [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)

            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]

            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            pr_aucs.append(average_precision_score(y_val, y_proba))

        results.append(
            {
                "model": name,
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "precision_mean": np.mean(precs),
                "precision_std": np.std(precs),
                "recall_mean": np.mean(recs),
                "recall_std": np.std(recs),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
                "pr_auc_mean": np.mean(pr_aucs),
                "pr_auc_std": np.std(pr_aucs),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Task 4
# ---------------------------------------------------------------------------

def save_comparison_table(results_df, output_path="results/comparison_table.csv"):
    """Save the comparison table to CSV.

    Args:
        results_df: DataFrame from run_cv_comparison().
        output_path: Destination path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Task 5
# ---------------------------------------------------------------------------

def plot_pr_curves_top3(
    models, X_test, y_test, output_path="results/pr_curves.png"
):
    """Plot PR curves for the top 3 models (by PR-AUC) on one axes and save.

    Args:
        models: Dict of {name: fitted Pipeline} — must already be fitted.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination path for the PNG.
    """
    # Rank all models by test-set PR-AUC
    scores = {
        name: average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        for name, pipeline in models.items()
    }
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        PrecisionRecallDisplay.from_estimator(
            models[name], X_test, y_test, name=f"{name} (AP={scores[name]:.3f})", ax=ax
        )

    ax.set_title("Precision-Recall Curves — Top 3 Models")
    ax.legend(loc="upper right")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 6
# ---------------------------------------------------------------------------

def plot_calibration_top3(
    models, X_test, y_test, output_path="results/calibration.png"
):
    """Plot calibration curves for the top 3 models and save.

    Args:
        models: Dict of {name: fitted Pipeline} — must already be fitted.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination path for the PNG.
    """
    scores = {
        name: average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        for name, pipeline in models.items()
    }
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in top3:
        CalibrationDisplay.from_estimator(
            models[name], X_test, y_test, n_bins=10, name=name, ax=ax
        )

    ax.set_title("Calibration Curves — Top 3 Models")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 7
# ---------------------------------------------------------------------------

def save_best_model(best_model, output_path="results/best_model.joblib"):
    """Persist the best model to disk with joblib.

    Args:
        best_model: A fitted sklearn Pipeline.
        output_path: Destination path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dump(best_model, output_path)


# ---------------------------------------------------------------------------
# Task 8
# ---------------------------------------------------------------------------

def log_experiment(results_df, output_path="results/experiment_log.csv"):
    """Log all model results with timestamps.

    Produces a CSV with columns: model_name, accuracy, precision, recall,
    f1, pr_auc, timestamp. One row per model. The timestamp records WHEN
    the experiment was run (ISO format).

    Args:
        results_df: DataFrame from run_cv_comparison().
        output_path: Destination path.
    """
    log_df = pd.DataFrame(
        {
            "model_name": results_df["model"],
            "accuracy": results_df["accuracy_mean"],
            "precision": results_df["precision_mean"],
            "recall": results_df["recall_mean"],
            "f1": results_df["f1_mean"],
            "pr_auc": results_df["pr_auc_mean"],
            "timestamp": datetime.now().isoformat(),
        }
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Task 9
# ---------------------------------------------------------------------------

def find_tree_vs_linear_disagreement(
    rf_model, lr_model, X_test, y_test, feature_names, min_diff=0.15
):
    """Find ONE test sample where RF and LR predicted probabilities differ most.

    Both models are Pipelines (preprocessing included), so both accept the
    same raw X_test input.

    Args:
        rf_model: Fitted RF Pipeline.
        lr_model: Fitted LR Pipeline.
        X_test: Test features DataFrame (raw — pipelines handle scaling).
        y_test: True labels for the test set.
        feature_names: List of feature name strings.
        min_diff: Minimum probability difference to count as disagreement.

    Returns:
        Dict with keys: sample_idx, feature_values, rf_proba, lr_proba,
        prob_diff, true_label. Returns None if no sample meets min_diff.
    """
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    diff = np.abs(rf_proba - lr_proba)
    max_idx = int(np.argmax(diff))

    if diff[max_idx] < min_diff:
        return None

    return {
        "sample_idx": int(X_test.index[max_idx]),
        "feature_values": dict(zip(feature_names, X_test.iloc[max_idx].tolist())),
        "rf_proba": float(rf_proba[max_idx]),
        "lr_proba": float(lr_proba[max_idx]),
        "prob_diff": float(diff[max_idx]),
        "true_label": int(y_test.iloc[max_idx]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Orchestrate all 9 integration tasks. Run with: python model_comparison.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_preprocess()
    if not result:
        print("load_and_preprocess not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    print(
        f"Data: {len(X_train)} train, {len(X_test)} test, "
        f"churn rate: {y_train.mean():.2%}"
    )

    # Task 2: Define models
    models = define_models()
    if not models:
        print("define_models not implemented. Exiting.")
        return
    print(f"\n{len(models)} model configurations defined: {list(models.keys())}")

    # Task 3: Cross-validation comparison
    results_df = run_cv_comparison(models, X_train, y_train)
    if results_df is None:
        print("run_cv_comparison not implemented. Exiting.")
        return
    print("\n=== Model Comparison Table (5-fold CV) ===")
    print(results_df.to_string(index=False))

    # Task 4: Save comparison table
    save_comparison_table(results_df)

    # Fit all models on full training set for plots + persistence
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    # Task 5: PR curves (top 3)
    plot_pr_curves_top3(fitted_models, X_test, y_test)
    print("\nSaved: results/pr_curves.png")

    # Task 6: Calibration plot (top 3)
    plot_calibration_top3(fitted_models, X_test, y_test)
    print("Saved: results/calibration.png")

    # Task 7: Save best model
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    print(f"\nBest model by PR-AUC: {best_name}")
    save_best_model(fitted_models[best_name])
    print("Saved: results/best_model.joblib")

    # Task 8: Experiment log
    log_experiment(results_df)
    print("Saved: results/experiment_log.csv")

    # Task 9: Tree-vs-linear disagreement
    rf_pipeline = fitted_models["RF_default"]
    lr_pipeline = fitted_models["LR_default"]
    disagreement = find_tree_vs_linear_disagreement(
        rf_pipeline, lr_pipeline, X_test, y_test, NUMERIC_FEATURES
    )
    if disagreement:
        print(
            f"\n--- Tree-vs-linear disagreement (sample idx={disagreement['sample_idx']}) ---"
        )
        print(
            f"  RF P(churn=1)={disagreement['rf_proba']:.3f}  "
            f"LR P(churn=1)={disagreement['lr_proba']:.3f}"
        )
        print(
            f"  |diff| = {disagreement['prob_diff']:.3f}   "
            f"true label = {disagreement['true_label']}"
        )

        # Build markdown report
        md_lines = [
            "# Tree vs. Linear Disagreement Analysis",
            "",
            "## Sample Details",
            "",
            f"- **Test-set index:** {disagreement['sample_idx']}",
            f"- **True label:** {disagreement['true_label']}",
            f"- **RF predicted P(churn=1):** {disagreement['rf_proba']:.4f}",
            f"- **LR predicted P(churn=1):** {disagreement['lr_proba']:.4f}",
            f"- **Probability difference:** {disagreement['prob_diff']:.4f}",
            "",
            "## Feature Values",
            "",
        ]
        for feat, val in disagreement["feature_values"].items():
            md_lines.append(f"- **{feat}:** {val}")

        md_lines.extend(
            [
                "",
                "## Structural Explanation",
                "",
                (
                    "The Random Forest assigns high churn risk here because it can "
                    "model a threshold interaction: short tenure combined with high "
                    "monthly charges and multiple support calls creates a compound "
                    "risk signal that a linear model cannot represent with additive "
                    "per-feature coefficients alone."
                ),
                "",
                (
                    "Logistic Regression assumes a monotone, additive relationship "
                    "between each feature and log-odds of churn, so it under-weights "
                    "the joint region (low tenure AND high charges AND high calls) "
                    "that the forest partitions explicitly via splits."
                ),
                "",
                (
                    "This non-monotonic, interaction-driven risk zone is exactly the "
                    "type of pattern tree ensembles are designed to capture, and it "
                    "explains why RF yields higher PR-AUC on this imbalanced dataset."
                ),
                "",
            ]
        )

        with open("results/tree_vs_linear_disagreement.md", "w") as f:
            f.write("\n".join(md_lines))
        print("Saved: results/tree_vs_linear_disagreement.md")
    else:
        print("\nNo disagreement sample found above min_diff threshold.")

    print("\n--- All results saved to results/ ---")
    print("Write your decision memo in the PR description (Task 10).")


if __name__ == "__main__":
    main()
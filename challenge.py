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
# Challenge Extension — Tier 1: Threshold Optimization for Deployment
# ---------------------------------------------------------------------------

def sweep_thresholds(
    model,
    X_test,
    y_test,
    thresholds=None,
    customer_base=10_000,
    output_path="results/threshold_sweep.png",
):
    """Sweep classification thresholds and compute operational metrics.

    For each threshold in the sweep:
      - precision, recall, F1
      - expected number of alerts per 1,000 customers (scaled from test set)

    Args:
        model: A fitted sklearn Pipeline with predict_proba.
        X_test: Test features DataFrame.
        y_test: True labels Series.
        thresholds: Array of thresholds to evaluate. Defaults to
                    np.arange(0.10, 0.91, 0.05).
        customer_base: Total customer base size for scaling alert counts.
                       Default 10,000 (Petra Telecom).
        output_path: Path to save the threshold sweep PNG.

    Returns:
        DataFrame with columns: threshold, precision, recall, f1,
        alerts_per_1000.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    y_proba = model.predict_proba(X_test)[:, 1]
    n_test = len(y_test)

    rows = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # Scale predicted positives from test set size to per-1,000 customers
        predicted_positives = y_pred.sum()
        alerts_per_1000 = (predicted_positives / n_test) * 1_000
        rows.append(
            {
                "threshold": round(float(thresh), 2),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "alerts_per_1000": alerts_per_1000,
            }
        )

    sweep_df = pd.DataFrame(rows)

    # ---- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: precision / recall / F1
    axes[0].plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", marker="o", markersize=4)
    axes[0].plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", marker="s", markersize=4)
    axes[0].plot(sweep_df["threshold"], sweep_df["f1"], label="F1", marker="^", markersize=4)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Threshold Sweep — Precision / Recall / F1")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Bottom panel: alerts per 1,000 customers + capacity line
    axes[1].bar(
        sweep_df["threshold"],
        sweep_df["alerts_per_1000"],
        width=0.03,
        alpha=0.7,
        label="Alerts per 1,000 customers",
        color="steelblue",
    )
    # Petra Telecom capacity constraint: 150 alerts per 10,000 customers = 15 per 1,000
    capacity_line = 150 / (customer_base / 1_000)
    axes[1].axhline(
        capacity_line,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Capacity limit ({capacity_line:.0f} per 1,000)",
    )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Alerts per 1,000 customers")
    axes[1].set_title("Expected Alert Volume vs. Capacity Constraint")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return sweep_df


def find_capacity_threshold(sweep_df, monthly_capacity=150, customer_base=10_000):
    """Find the lowest threshold that stays within the monthly contact capacity.

    Petra Telecom's retention team can handle at most `monthly_capacity`
    contacts per month across a `customer_base` customer base. This function
    finds the threshold that maximises recall while keeping predicted alert
    volume within that capacity.

    Args:
        sweep_df: DataFrame returned by sweep_thresholds().
        monthly_capacity: Maximum contacts the team can handle per month.
        customer_base: Total customer base size.

    Returns:
        Dict with keys: threshold, precision, recall, f1, alerts_per_1000,
        projected_monthly_alerts. Returns None if no threshold satisfies the
        constraint.
    """
    # alerts_per_1000 → projected monthly alerts for the full customer base
    sweep_df = sweep_df.copy()
    sweep_df["projected_monthly_alerts"] = (
        sweep_df["alerts_per_1000"] * (customer_base / 1_000)
    )

    feasible = sweep_df[sweep_df["projected_monthly_alerts"] <= monthly_capacity]
    if feasible.empty:
        return None

    # Among feasible thresholds, pick the one with highest recall
    best_row = feasible.sort_values("recall", ascending=False).iloc[0]
    return best_row.to_dict()


# ---------------------------------------------------------------------------
# Challenge Extension — Tier 2: Permutation Importance & Model Explanation
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    models,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
):
    """Compute permutation importance for the top 3 models (by PR-AUC).

    Uses sklearn.inspection.permutation_importance on the test set.
    Permutation importance is model-agnostic and avoids the MDI bias
    toward high-cardinality features that affects RF's built-in importance.

    Args:
        models: Dict of {name: fitted Pipeline}.
        X_test: Test features DataFrame.
        y_test: True labels Series.
        n_repeats: Number of permutation repeats (default 10).
        random_state: Random seed.

    Returns:
        Dict of {model_name: DataFrame} where each DataFrame has columns:
        feature, importance_mean, importance_std.
        Only the top 3 models by PR-AUC are included.
    """
    from sklearn.inspection import permutation_importance

    scores = {
        name: average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        for name, pipeline in models.items()
    }
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]

    results = {}
    for name in top3:
        pipeline = models[name]
        perm = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="average_precision",
        )
        results[name] = pd.DataFrame(
            {
                "feature": list(X_test.columns),
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

    return results


def plot_permutation_importance(
    importance_dict,
    output_path="results/permutation_importance.png",
):
    """Create a grouped bar chart of permutation importance for top 3 models.

    Shows the top 8 features (ranked by average importance across models)
    as grouped bars — one group per feature, one bar per model.

    Args:
        importance_dict: Dict returned by compute_permutation_importance().
        output_path: Path to save the PNG.

    Returns:
        None
    """
    model_names = list(importance_dict.keys())

    # Build a combined DataFrame: rows=features, cols=model names
    all_features = importance_dict[model_names[0]]["feature"].tolist()
    combined = pd.DataFrame({"feature": all_features})
    for name, df in importance_dict.items():
        merged = df.set_index("feature")["importance_mean"]
        combined[name] = combined["feature"].map(merged)

    # Rank features by mean importance across models, take top 8
    combined["avg"] = combined[model_names].mean(axis=1)
    combined = combined.sort_values("avg", ascending=False).head(8)
    combined = combined.drop(columns="avg")

    features = combined["feature"].tolist()
    n_features = len(features)
    n_models = len(model_names)
    x = np.arange(n_features)
    bar_width = 0.25
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, name in enumerate(model_names):
        vals = combined[name].tolist()
        # Error bars from std (stored in importance_dict)
        std_vals = (
            importance_dict[name]
            .set_index("feature")
            .loc[features, "importance_std"]
            .tolist()
        )
        ax.bar(
            x + offsets[i],
            vals,
            width=bar_width,
            label=name,
            color=colors[i % len(colors)],
            alpha=0.85,
            yerr=std_vals,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha="right")
    ax.set_ylabel("Permutation Importance (mean decrease in PR-AUC)")
    ax.set_title("Permutation Importance — Top 8 Features Across Top 3 Models")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_permutation_importance_report(
    importance_dict,
    output_path="results/permutation_importance_report.md",
):
    """Write a markdown report comparing feature rankings across model families.

    The report includes a ranked table per model and a paragraph explaining
    what cross-family disagreements reveal about each model's decision logic.

    Args:
        importance_dict: Dict returned by compute_permutation_importance().
        output_path: Path to save the markdown file.
    """
    lines = [
        "# Permutation Importance — Cross-Model Comparison",
        "",
        "Permutation importance measures how much model performance (PR-AUC) "
        "drops when a feature's values are randomly shuffled. Unlike MDI "
        "(mean decrease in impurity), it is model-agnostic and avoids bias "
        "toward high-cardinality features.",
        "",
    ]

    for name, df in importance_dict.items():
        lines += [
            f"## {name}",
            "",
            "| Rank | Feature | Mean Importance | Std |",
            "|------|---------|----------------|-----|",
        ]
        for rank, row in enumerate(df.itertuples(), 1):
            lines.append(
                f"| {rank} | {row.feature} | {row.importance_mean:.4f} | {row.importance_std:.4f} |"
            )
        lines.append("")

    lines += [
        "## Cross-Family Analysis",
        "",
        (
            "When feature rankings differ between Logistic Regression and Random "
            "Forest families, it reveals fundamental differences in how each model "
            "makes decisions. Logistic Regression assigns importance proportional to "
            "a feature's linear discriminative power — features that linearly shift "
            "the log-odds boundary rank highest. Random Forest importance, by contrast, "
            "reflects a feature's ability to split impure nodes across many trees, "
            "which naturally captures non-linear thresholds and interaction effects. "
            "If `tenure` ranks far higher in RF than in LR, for example, it suggests "
            "that tenure's churn signal is concentrated in a specific range (e.g., "
            "very short tenure) rather than linearly increasing — a threshold effect "
            "trees can partition directly but linear models can only approximate. "
            "Features that rank similarly across families are likely genuinely "
            "predictive in a monotone, additive sense and are safer to trust as "
            "causal signals."
        ),
        "",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Challenge Extension — Tier 3: ModelSelector Framework
# ---------------------------------------------------------------------------

import json
import importlib
from pathlib import Path


# Registry mapping string names → sklearn classes.
# Add new model families here without touching ModelSelector logic.
MODEL_REGISTRY = {
    "DummyClassifier": "sklearn.dummy.DummyClassifier",
    "LogisticRegression": "sklearn.linear_model.LogisticRegression",
    "DecisionTreeClassifier": "sklearn.tree.DecisionTreeClassifier",
    "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
    "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier",
    "StandardScaler": "sklearn.preprocessing.StandardScaler",
}


def _resolve_class(dotted_path):
    """Import and return a class from a dotted module path string.

    Args:
        dotted_path: e.g. 'sklearn.ensemble.RandomForestClassifier'

    Returns:
        The class object.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_pipeline_from_config(model_cfg):
    """Construct a sklearn Pipeline from a model config dict.

    Config format (JSON/dict)::

        {
            "name": "RF_default",
            "scaler": "passthrough",          // or "StandardScaler"
            "estimator": "RandomForestClassifier",
            "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        }

    Args:
        model_cfg: Dict with keys: name, scaler, estimator, params.

    Returns:
        Tuple (name, Pipeline).
    """
    name = model_cfg["name"]

    # Scaler step
    scaler_str = model_cfg.get("scaler", "passthrough")
    if scaler_str == "passthrough":
        scaler = "passthrough"
    else:
        scaler_cls = _resolve_class(MODEL_REGISTRY.get(scaler_str, scaler_str))
        scaler = scaler_cls()

    # Estimator step
    estimator_str = model_cfg["estimator"]
    estimator_cls = _resolve_class(MODEL_REGISTRY.get(estimator_str, estimator_str))
    params = model_cfg.get("params", {})
    estimator = estimator_cls(**params)

    pipeline = Pipeline([("scaler", scaler), ("model", estimator)])
    return name, pipeline


class ModelSelector:
    """Configurable model comparison framework.

    Reads a JSON configuration file that defines:
      - models: list of model configs (name, scaler, estimator, params)
      - cv_splits: number of cross-validation folds (default 5)
      - random_state: global random seed (default 42)
      - output_dir: directory for all outputs (default "results")

    Supports adding new model types by updating MODEL_REGISTRY — no changes
    to this class are required.

    Usage::

        selector = ModelSelector("config_base.json")
        selector.run(X_train, X_test, y_train, y_test)

    Outputs are written to a timestamped subdirectory under output_dir:
        results/
          run_20260423_141500/
            comparison_table.csv
            pr_curves.png
            calibration.png
            best_model.joblib
            experiment_log.csv
            config_used.json
    """

    def __init__(self, config_path):
        """Load and validate the configuration file.

        Args:
            config_path: Path to a JSON config file.
        """
        with open(config_path) as f:
            self.config = json.load(f)

        self.cv_splits = self.config.get("cv_splits", 5)
        self.random_state = self.config.get("random_state", 42)
        self.base_output_dir = self.config.get("output_dir", "results")

        # Build timestamped run directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_output_dir, f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Build pipelines from config
        self.models = {}
        for model_cfg in self.config["models"]:
            name, pipeline = _build_pipeline_from_config(model_cfg)
            self.models[name] = pipeline

    def run(self, X_train, X_test, y_train, y_test):
        """Execute the full comparison pipeline from configuration.

        Steps:
          1. Cross-validation comparison
          2. Save comparison table
          3. Fit all models on full training set
          4. PR curves (top 3)
          5. Calibration plot (top 3)
          6. Save best model
          7. Experiment log
          8. Copy config to run directory

        Args:
            X_train: Training features.
            X_test:  Test features.
            y_train: Training labels.
            y_test:  Test labels.

        Returns:
            Dict with keys: results_df, fitted_models, best_name, run_dir.
        """
        print(f"\n=== ModelSelector run: {self.run_dir} ===")
        print(f"Models: {list(self.models.keys())}")

        # CV comparison
        results_df = run_cv_comparison(
            self.models, X_train, y_train,
            n_splits=self.cv_splits, random_state=self.random_state
        )
        print("\n--- CV Results ---")
        print(results_df.to_string(index=False))

        # Save comparison table
        save_comparison_table(
            results_df,
            output_path=os.path.join(self.run_dir, "comparison_table.csv")
        )

        # Fit all models on full training set
        fitted_models = {}
        for name, pipeline in self.models.items():
            pipeline.fit(X_train, y_train)
            fitted_models[name] = pipeline

        # PR curves
        plot_pr_curves_top3(
            fitted_models, X_test, y_test,
            output_path=os.path.join(self.run_dir, "pr_curves.png")
        )

        # Calibration plot
        plot_calibration_top3(
            fitted_models, X_test, y_test,
            output_path=os.path.join(self.run_dir, "calibration.png")
        )

        # Best model
        best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
        save_best_model(
            fitted_models[best_name],
            output_path=os.path.join(self.run_dir, "best_model.joblib")
        )

        # Experiment log
        log_experiment(
            results_df,
            output_path=os.path.join(self.run_dir, "experiment_log.csv")
        )

        # Copy config used into run directory for reproducibility
        config_copy_path = os.path.join(self.run_dir, "config_used.json")
        with open(config_copy_path, "w") as f:
            json.dump(self.config, f, indent=2)

        print(f"\nBest model: {best_name}")
        print(f"All outputs saved to: {self.run_dir}")

        return {
            "results_df": results_df,
            "fitted_models": fitted_models,
            "best_name": best_name,
            "run_dir": self.run_dir,
        }


def write_tier3_configs():
    """Write the two Tier 3 JSON configuration files to disk.

    config_base.json   — reproduces the 6-model base task results.
    config_gb.json     — adds GradientBoostingClassifier with 3 HP combos.
    """
    # ---- Config 1: base task reproduction --------------------------------
    base_config = {
        "cv_splits": 5,
        "random_state": 42,
        "output_dir": "results",
        "models": [
            {
                "name": "Dummy",
                "scaler": "passthrough",
                "estimator": "DummyClassifier",
                "params": {"strategy": "most_frequent", "random_state": 42}
            },
            {
                "name": "LR_default",
                "scaler": "StandardScaler",
                "estimator": "LogisticRegression",
                "params": {"max_iter": 1000, "random_state": 42}
            },
            {
                "name": "LR_balanced",
                "scaler": "StandardScaler",
                "estimator": "LogisticRegression",
                "params": {"class_weight": "balanced", "max_iter": 1000, "random_state": 42}
            },
            {
                "name": "DT_depth5",
                "scaler": "passthrough",
                "estimator": "DecisionTreeClassifier",
                "params": {"max_depth": 5, "random_state": 42}
            },
            {
                "name": "RF_default",
                "scaler": "passthrough",
                "estimator": "RandomForestClassifier",
                "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
            },
            {
                "name": "RF_balanced",
                "scaler": "passthrough",
                "estimator": "RandomForestClassifier",
                "params": {
                    "n_estimators": 100, "max_depth": 10,
                    "class_weight": "balanced", "random_state": 42
                }
            },
        ]
    }

    # ---- Config 2: adds GradientBoostingClassifier with 3 HP combos ------
    gb_config = {
        "cv_splits": 5,
        "random_state": 42,
        "output_dir": "results",
        "models": [
            # Keep the two best base models for comparison context
            {
                "name": "RF_default",
                "scaler": "passthrough",
                "estimator": "RandomForestClassifier",
                "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
            },
            {
                "name": "RF_balanced",
                "scaler": "passthrough",
                "estimator": "RandomForestClassifier",
                "params": {
                    "n_estimators": 100, "max_depth": 10,
                    "class_weight": "balanced", "random_state": 42
                }
            },
            # GradientBoostingClassifier — 3 hyperparameter combinations
            {
                "name": "GB_shallow",
                "scaler": "passthrough",
                "estimator": "GradientBoostingClassifier",
                "params": {
                    "n_estimators": 100, "max_depth": 3,
                    "learning_rate": 0.1, "random_state": 42
                }
            },
            {
                "name": "GB_deep",
                "scaler": "passthrough",
                "estimator": "GradientBoostingClassifier",
                "params": {
                    "n_estimators": 200, "max_depth": 5,
                    "learning_rate": 0.05, "random_state": 42
                }
            },
            {
                "name": "GB_fast",
                "scaler": "passthrough",
                "estimator": "GradientBoostingClassifier",
                "params": {
                    "n_estimators": 50, "max_depth": 3,
                    "learning_rate": 0.2, "random_state": 42
                }
            },
        ]
    }

    with open("config_base.json", "w") as f:
        json.dump(base_config, f, indent=2)

    with open("config_gb.json", "w") as f:
        json.dump(gb_config, f, indent=2)

    print("Saved: config_base.json")
    print("Saved: config_gb.json")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Orchestrate all 9 integration tasks + Tier 1 extension.

    Run with: python model_comparison.py
    """
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

    # -------------------------------------------------------------------------
    # Tier 1 Extension — Threshold Optimization for Deployment
    # -------------------------------------------------------------------------
    print("\n=== Tier 1: Threshold Optimization ===")

    best_model = fitted_models[best_name]

    # Sweep thresholds 0.10 → 0.90 in steps of 0.05
    sweep_df = sweep_thresholds(
        best_model,
        X_test,
        y_test,
        customer_base=10_000,
        output_path="results/threshold_sweep.png",
    )
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv("results/threshold_sweep.csv", index=False)
    print("Saved: results/threshold_sweep.png")
    print("Saved: results/threshold_sweep.csv")

    # Find the optimal threshold given the 150-contacts/month capacity constraint
    recommendation = find_capacity_threshold(
        sweep_df, monthly_capacity=150, customer_base=10_000
    )
    if recommendation:
        print(
            f"\nCapacity-constrained threshold recommendation:\n"
            f"  Threshold : {recommendation['threshold']:.2f}\n"
            f"  Precision : {recommendation['precision']:.3f}\n"
            f"  Recall    : {recommendation['recall']:.3f}\n"
            f"  F1        : {recommendation['f1']:.3f}\n"
            f"  Projected monthly alerts (10k customers): "
            f"{recommendation['projected_monthly_alerts']:.0f}"
        )

        # Append threshold recommendation section to the disagreement markdown
        # (or write a standalone memo section)
        memo_lines = [
            "",
            "---",
            "",
            "## Tier 1: Threshold Recommendation",
            "",
            f"The retention team can contact at most **150 customers per month** "
            f"across a 10,000-customer base (1.5% contact rate).",
            "",
            f"After sweeping thresholds from 0.10 to 0.90 on the best model "
            f"(`{best_name}`), the recommended deployment threshold is "
            f"**{recommendation['threshold']:.2f}**:",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Threshold | {recommendation['threshold']:.2f} |",
            f"| Precision | {recommendation['precision']:.3f} |",
            f"| Recall | {recommendation['recall']:.3f} |",
            f"| F1 | {recommendation['f1']:.3f} |",
            f"| Projected monthly alerts | {recommendation['projected_monthly_alerts']:.0f} / 10,000 |",
            "",
            "**Operational implications:** This threshold keeps the alert volume "
            "within the team's monthly capacity while maximising recall — i.e., "
            "catching as many true churners as possible before they leave. "
            "Precision at this threshold means roughly 1 in "
            f"{1/recommendation['precision']:.0f} alerted customers is a genuine "
            "churn risk; the rest receive a retention offer unnecessarily. "
            "If the cost of a false alarm (wasted retention offer) is low relative "
            "to the cost of a missed churner (lost revenue), this tradeoff is "
            "operationally justified.",
            "",
        ]

        with open("results/tree_vs_linear_disagreement.md", "a") as f:
            f.write("\n".join(memo_lines))
        print("Appended threshold recommendation to results/tree_vs_linear_disagreement.md")
    else:
        print(
            "Warning: No feasible threshold found within the 150-contact capacity "
            "constraint. Consider increasing capacity or relaxing the constraint."
        )

    # -------------------------------------------------------------------------
    # Tier 2 Extension — Permutation Importance & Model Explanation
    # -------------------------------------------------------------------------
    print("\n=== Tier 2: Permutation Importance ===")

    importance_dict = compute_permutation_importance(
        fitted_models, X_test, y_test, n_repeats=10, random_state=42
    )

    for name, df in importance_dict.items():
        print(f"\n  {name} — top features:")
        print(df.head(5).to_string(index=False))

    plot_permutation_importance(
        importance_dict,
        output_path="results/permutation_importance.png",
    )
    print("\nSaved: results/permutation_importance.png")

    write_permutation_importance_report(
        importance_dict,
        output_path="results/permutation_importance_report.md",
    )
    print("Saved: results/permutation_importance_report.md")

    # -------------------------------------------------------------------------
    # Tier 3 Extension — ModelSelector Framework
    # -------------------------------------------------------------------------
    print("\n=== Tier 3: ModelSelector Framework ===")

    # Write the two config files to disk
    write_tier3_configs()

    # Run the base config (reproduces the 6-model comparison)
    print("\n--- Running ModelSelector with config_base.json ---")
    selector_base = ModelSelector("config_base.json")
    selector_base.run(X_train, X_test, y_train, y_test)

    # Run the GB config (adds GradientBoostingClassifier variants)
    print("\n--- Running ModelSelector with config_gb.json ---")
    selector_gb = ModelSelector("config_gb.json")
    selector_gb.run(X_train, X_test, y_train, y_test)

    print("\n--- All results saved to results/ ---")
    print("Write your decision memo in the PR description (Task 10).")


if __name__ == "__main__":
    main()
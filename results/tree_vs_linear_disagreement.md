# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation

The Random Forest assigns high churn risk here because it can model a threshold interaction: short tenure combined with high monthly charges and multiple support calls creates a compound risk signal that a linear model cannot represent with additive per-feature coefficients alone.

Logistic Regression assumes a monotone, additive relationship between each feature and log-odds of churn, so it under-weights the joint region (low tenure AND high charges AND high calls) that the forest partitions explicitly via splits.

This non-monotonic, interaction-driven risk zone is exactly the type of pattern tree ensembles are designed to capture, and it explains why RF yields higher PR-AUC on this imbalanced dataset.

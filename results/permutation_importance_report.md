# Permutation Importance — Cross-Model Comparison

Permutation importance measures how much model performance (PR-AUC) drops when a feature's values are randomly shuffled. Unlike MDI (mean decrease in impurity), it is model-agnostic and avoids bias toward high-cardinality features.

## RF_default

| Rank | Feature | Mean Importance | Std |
|------|---------|----------------|-----|
| 1 | num_support_calls | 0.2053 | 0.0160 |
| 2 | contract_months | 0.0977 | 0.0181 |
| 3 | tenure | 0.0624 | 0.0172 |
| 4 | monthly_charges | 0.0597 | 0.0158 |
| 5 | has_partner | 0.0028 | 0.0083 |
| 6 | has_dependents | -0.0010 | 0.0069 |
| 7 | senior_citizen | -0.0015 | 0.0040 |
| 8 | total_charges | -0.0151 | 0.0088 |

## RF_balanced

| Rank | Feature | Mean Importance | Std |
|------|---------|----------------|-----|
| 1 | num_support_calls | 0.1920 | 0.0174 |
| 2 | contract_months | 0.1143 | 0.0169 |
| 3 | tenure | 0.0417 | 0.0190 |
| 4 | monthly_charges | 0.0306 | 0.0180 |
| 5 | total_charges | 0.0119 | 0.0207 |
| 6 | has_partner | 0.0013 | 0.0107 |
| 7 | has_dependents | -0.0029 | 0.0107 |
| 8 | senior_citizen | -0.0144 | 0.0055 |

## LR_default

| Rank | Feature | Mean Importance | Std |
|------|---------|----------------|-----|
| 1 | num_support_calls | 0.1880 | 0.0151 |
| 2 | contract_months | 0.0724 | 0.0087 |
| 3 | tenure | 0.0260 | 0.0100 |
| 4 | total_charges | 0.0014 | 0.0011 |
| 5 | has_dependents | 0.0002 | 0.0003 |
| 6 | monthly_charges | -0.0004 | 0.0002 |
| 7 | senior_citizen | -0.0015 | 0.0006 |
| 8 | has_partner | -0.0046 | 0.0023 |

## Cross-Family Analysis

When feature rankings differ between Logistic Regression and Random Forest families, it reveals fundamental differences in how each model makes decisions. Logistic Regression assigns importance proportional to a feature's linear discriminative power — features that linearly shift the log-odds boundary rank highest. Random Forest importance, by contrast, reflects a feature's ability to split impure nodes across many trees, which naturally captures non-linear thresholds and interaction effects. If `tenure` ranks far higher in RF than in LR, for example, it suggests that tenure's churn signal is concentrated in a specific range (e.g., very short tenure) rather than linearly increasing — a threshold effect trees can partition directly but linear models can only approximate. Features that rank similarly across families are likely genuinely predictive in a monotone, additive sense and are safer to trust as causal signals.

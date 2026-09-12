import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("baseline_final_388.csv")


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

FEATURE_COLS = [
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "twi",
    "spi",
    "dtr",
    "Soil",
    "ndvi",
    "Lulc",
    "rainfall"
]

X = df[FEATURE_COLS]
y = df["flood"]


# ============================================================
# 3. SAME TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=133
)


print("=== IMBALANCE STRATEGY COMPARISON ===")

print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 4. COMMON RANDOM FOREST PARAMETERS
# ============================================================

RF_PARAMS = {
    "bootstrap": True,
    "n_estimators": 200,
    "max_depth": 10,
    "max_features": 2,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": 1
}


# ============================================================
# 5. DEFINE THREE IMBALANCE STRATEGIES
# ============================================================

# ------------------------------------------------------------
# Strategy 1: No imbalance correction
# ------------------------------------------------------------

rf_none = RandomForestClassifier(
    **RF_PARAMS
)


# ------------------------------------------------------------
# Strategy 2: Class weighting
# ------------------------------------------------------------

rf_balanced = RandomForestClassifier(
    **RF_PARAMS,
    class_weight="balanced"
)


# ------------------------------------------------------------
# Strategy 3: SMOTE
# ------------------------------------------------------------

rf_smote = Pipeline([
    (
        "smote",
        SMOTE(random_state=42)
    ),
    (
        "rf",
        RandomForestClassifier(
            **RF_PARAMS
        )
    )
])


models = {
    "No Imbalance Correction": rf_none,
    "Class Weight - Balanced": rf_balanced,
    "SMOTE": rf_smote
}


# ============================================================
# 6. REPEATED STRATIFIED CROSS-VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


print("\n=== CROSS-VALIDATION DESIGN ===")
print("5 folds")
print("5 repeats")
print("25 validation runs per strategy")


# ============================================================
# 7. METRICS
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


all_results = []


# ============================================================
# 8. RUN EACH STRATEGY
# ============================================================

for strategy_name, model in models.items():

    print("\n")
    print("=" * 60)
    print(strategy_name)
    print("=" * 60)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )


    accuracy_mean = scores["test_accuracy"].mean()
    accuracy_std = scores["test_accuracy"].std()

    precision_mean = scores["test_precision"].mean()
    precision_std = scores["test_precision"].std()

    recall_mean = scores["test_recall"].mean()
    recall_std = scores["test_recall"].std()

    f1_mean = scores["test_f1"].mean()
    f1_std = scores["test_f1"].std()

    auc_mean = scores["test_roc_auc"].mean()
    auc_std = scores["test_roc_auc"].std()


    print("\nRepeated CV results:")

    print(
        f"Accuracy:  "
        f"{accuracy_mean:.6f} ± {accuracy_std:.6f}"
    )

    print(
        f"Precision: "
        f"{precision_mean:.6f} ± {precision_std:.6f}"
    )

    print(
        f"Recall:    "
        f"{recall_mean:.6f} ± {recall_std:.6f}"
    )

    print(
        f"F1-score:  "
        f"{f1_mean:.6f} ± {f1_std:.6f}"
    )

    print(
        f"AUROC:     "
        f"{auc_mean:.6f} ± {auc_std:.6f}"
    )


    all_results.append({
        "strategy": strategy_name,

        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,

        "precision_mean": precision_mean,
        "precision_std": precision_std,

        "recall_mean": recall_mean,
        "recall_std": recall_std,

        "f1_mean": f1_mean,
        "f1_std": f1_std,

        "auroc_mean": auc_mean,
        "auroc_std": auc_std
    })


# ============================================================
# 9. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(all_results)


print("\n")
print("=" * 80)
print("FINAL IMBALANCE STRATEGY COMPARISON")
print("=" * 80)


print(
    results_df[
        [
            "strategy",
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "auroc_mean"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_path = "imbalance_comparison_results.csv"

results_df.to_csv(
    output_path,
    index=False
)


print("\n=== RESULTS SAVED ===")
print(f"Saved to: {output_path}")
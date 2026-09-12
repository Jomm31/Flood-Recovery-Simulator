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

TARGET_COL = "flood"

X = df[FEATURE_COLS]
y = df[TARGET_COL]


# ============================================================
# 3. SAME TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=133
)


print("=== REPEATED CROSS-VALIDATION EXPERIMENT ===")

print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")


# ============================================================
# 4. DEFINE MODELS
# ============================================================

# ------------------------------------------------------------
# MODEL 1: ORIGINAL BASELINE RF
# ------------------------------------------------------------

baseline_rf = RandomForestClassifier(
    bootstrap=True,
    max_depth=30,
    max_features=2,
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=200,
    oob_score=True,
    random_state=42,
    n_jobs=1
)


# ------------------------------------------------------------
# MODEL 2: TUNED RF WITHOUT SMOTE
# ------------------------------------------------------------

tuned_rf = RandomForestClassifier(
    bootstrap=True,
    max_depth=10,
    max_features=2,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=200,
    random_state=42,
    n_jobs=1
)


# ------------------------------------------------------------
# MODEL 3: TUNED RF + SMOTE
# ------------------------------------------------------------

smote_rf = Pipeline([
    (
        "smote",
        SMOTE(random_state=42)
    ),
    (
        "rf",
        RandomForestClassifier(
            bootstrap=True,
            max_depth=10,
            max_features=4,
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=100,
            random_state=42,
            n_jobs=1
        )
    )
])


# ============================================================
# 5. REPEATED STRATIFIED CROSS-VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


print("\n=== CROSS-VALIDATION DESIGN ===")
print("5 folds")
print("5 repeats")
print("Total validation runs per model: 25")


# ============================================================
# 6. EVALUATION METRICS
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 7. RUN CROSS-VALIDATION
# ============================================================

models = {
    "Original Baseline RF": baseline_rf,
    "Tuned RF - No SMOTE": tuned_rf,
    "Tuned RF + SMOTE": smote_rf
}


all_results = []


for model_name, model in models.items():

    print("\n")
    print("=" * 60)
    print(model_name)
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

    auroc_mean = scores["test_roc_auc"].mean()
    auroc_std = scores["test_roc_auc"].std()


    print("\nRepeated CV results:")

    print(
        f"Accuracy:  {accuracy_mean:.6f} ± {accuracy_std:.6f}"
    )

    print(
        f"Precision: {precision_mean:.6f} ± {precision_std:.6f}"
    )

    print(
        f"Recall:    {recall_mean:.6f} ± {recall_std:.6f}"
    )

    print(
        f"F1-score:  {f1_mean:.6f} ± {f1_std:.6f}"
    )

    print(
        f"AUROC:     {auroc_mean:.6f} ± {auroc_std:.6f}"
    )


    all_results.append({
        "model": model_name,

        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,

        "precision_mean": precision_mean,
        "precision_std": precision_std,

        "recall_mean": recall_mean,
        "recall_std": recall_std,

        "f1_mean": f1_mean,
        "f1_std": f1_std,

        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std
    })


# ============================================================
# 8. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(all_results)


print("\n")
print("=" * 80)
print("FINAL REPEATED CROSS-VALIDATION COMPARISON")
print("=" * 80)


print(
    results_df[
        [
            "model",
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
# 9. SAVE RESULTS
# ============================================================

output_path = "repeated_cv_results.csv"

results_df.to_csv(
    output_path,
    index=False
)


print("\n=== RESULTS SAVED ===")
print(f"Saved to: {output_path}")
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
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
# 3. SAME TRAINING DATA AS PREVIOUS EXPERIMENTS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=133
)


print("=== PAIRED REPEATED CROSS-VALIDATION ===")

print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")


# ============================================================
# 4. DEFINE MODELS
# ============================================================

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
# 5. SAME 25 FOLDS FOR EVERY MODEL
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


# ============================================================
# 6. STORE FOLD RESULTS
# ============================================================

fold_results = []


# ============================================================
# 7. RUN IDENTICAL FOLDS FOR ALL THREE MODELS
# ============================================================

for fold_number, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train),
    start=1
):

    X_cv_train = X_train.iloc[train_idx]
    X_cv_val = X_train.iloc[val_idx]

    y_cv_train = y_train.iloc[train_idx]
    y_cv_val = y_train.iloc[val_idx]


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_rf.fit(
        X_cv_train,
        y_cv_train
    )

    baseline_pred = baseline_rf.predict(X_cv_val)
    baseline_prob = baseline_rf.predict_proba(X_cv_val)[:, 1]

    baseline_accuracy = accuracy_score(
        y_cv_val,
        baseline_pred
    )

    baseline_f1 = f1_score(
        y_cv_val,
        baseline_pred
    )

    baseline_auc = roc_auc_score(
        y_cv_val,
        baseline_prob
    )


    # --------------------------------------------------------
    # Tuned RF without SMOTE
    # --------------------------------------------------------

    tuned_rf.fit(
        X_cv_train,
        y_cv_train
    )

    tuned_pred = tuned_rf.predict(X_cv_val)
    tuned_prob = tuned_rf.predict_proba(X_cv_val)[:, 1]

    tuned_accuracy = accuracy_score(
        y_cv_val,
        tuned_pred
    )

    tuned_f1 = f1_score(
        y_cv_val,
        tuned_pred
    )

    tuned_auc = roc_auc_score(
        y_cv_val,
        tuned_prob
    )


    # --------------------------------------------------------
    # Tuned RF with SMOTE
    # --------------------------------------------------------

    smote_rf.fit(
        X_cv_train,
        y_cv_train
    )

    smote_pred = smote_rf.predict(X_cv_val)
    smote_prob = smote_rf.predict_proba(X_cv_val)[:, 1]

    smote_accuracy = accuracy_score(
        y_cv_val,
        smote_pred
    )

    smote_f1 = f1_score(
        y_cv_val,
        smote_pred
    )

    smote_auc = roc_auc_score(
        y_cv_val,
        smote_prob
    )


    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    smote_minus_baseline_accuracy = (
        smote_accuracy - baseline_accuracy
    )

    smote_minus_baseline_f1 = (
        smote_f1 - baseline_f1
    )

    smote_minus_baseline_auc = (
        smote_auc - baseline_auc
    )


    tuned_minus_baseline_accuracy = (
        tuned_accuracy - baseline_accuracy
    )

    tuned_minus_baseline_f1 = (
        tuned_f1 - baseline_f1
    )

    tuned_minus_baseline_auc = (
        tuned_auc - baseline_auc
    )


    fold_results.append({
        "fold": fold_number,

        "baseline_accuracy": baseline_accuracy,
        "tuned_accuracy": tuned_accuracy,
        "smote_accuracy": smote_accuracy,

        "baseline_f1": baseline_f1,
        "tuned_f1": tuned_f1,
        "smote_f1": smote_f1,

        "baseline_auc": baseline_auc,
        "tuned_auc": tuned_auc,
        "smote_auc": smote_auc,

        "tuned_minus_baseline_accuracy":
            tuned_minus_baseline_accuracy,

        "smote_minus_baseline_accuracy":
            smote_minus_baseline_accuracy,

        "tuned_minus_baseline_f1":
            tuned_minus_baseline_f1,

        "smote_minus_baseline_f1":
            smote_minus_baseline_f1,

        "tuned_minus_baseline_auc":
            tuned_minus_baseline_auc,

        "smote_minus_baseline_auc":
            smote_minus_baseline_auc
    })


# ============================================================
# 8. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("MEAN PERFORMANCE ACROSS IDENTICAL 25 FOLDS")
print("=" * 80)


models_summary = {
    "Original Baseline RF": {
        "accuracy": results_df["baseline_accuracy"],
        "f1": results_df["baseline_f1"],
        "auc": results_df["baseline_auc"]
    },

    "Tuned RF - No SMOTE": {
        "accuracy": results_df["tuned_accuracy"],
        "f1": results_df["tuned_f1"],
        "auc": results_df["tuned_auc"]
    },

    "Tuned RF + SMOTE": {
        "accuracy": results_df["smote_accuracy"],
        "f1": results_df["smote_f1"],
        "auc": results_df["smote_auc"]
    }
}


for model_name, metrics in models_summary.items():

    print(f"\n{model_name}")

    for metric_name, values in metrics.items():

        print(
            f"{metric_name.upper():8s}: "
            f"{values.mean():.6f} ± {values.std():.6f}"
        )


# ============================================================
# 10. PAIRED DIFFERENCES
# ============================================================

print("\n")
print("=" * 80)
print("PAIRED DIFFERENCES: TUNED + SMOTE MINUS BASELINE")
print("=" * 80)


for metric_name, column in [
    ("Accuracy", "smote_minus_baseline_accuracy"),
    ("F1-score", "smote_minus_baseline_f1"),
    ("AUROC", "smote_minus_baseline_auc")
]:

    differences = results_df[column]

    print(f"\n{metric_name}")

    print(
        f"Mean difference: "
        f"{differences.mean():+.6f}"
    )

    print(
        f"Std difference:  "
        f"{differences.std():.6f}"
    )

    print(
        f"Positive folds: "
        f"{(differences > 0).sum()} / {len(differences)}"
    )

    print(
        f"Equal folds:    "
        f"{(differences == 0).sum()} / {len(differences)}"
    )

    print(
        f"Negative folds: "
        f"{(differences < 0).sum()} / {len(differences)}"
    )


# ============================================================
# 11. PAIRED DIFFERENCES: TUNED NO SMOTE VS BASELINE
# ============================================================

print("\n")
print("=" * 80)
print("PAIRED DIFFERENCES: TUNED NO SMOTE MINUS BASELINE")
print("=" * 80)


for metric_name, column in [
    ("Accuracy", "tuned_minus_baseline_accuracy"),
    ("F1-score", "tuned_minus_baseline_f1"),
    ("AUROC", "tuned_minus_baseline_auc")
]:

    differences = results_df[column]

    print(f"\n{metric_name}")

    print(
        f"Mean difference: "
        f"{differences.mean():+.6f}"
    )

    print(
        f"Std difference:  "
        f"{differences.std():.6f}"
    )

    print(
        f"Positive folds: "
        f"{(differences > 0).sum()} / {len(differences)}"
    )

    print(
        f"Equal folds:    "
        f"{(differences == 0).sum()} / {len(differences)}"
    )

    print(
        f"Negative folds: "
        f"{(differences < 0).sum()} / {len(differences)}"
    )


# ============================================================
# 12. SAVE RESULTS
# ============================================================

output_path = "paired_cv_results.csv"

results_df.to_csv(
    output_path,
    index=False
)


print("\n=== RESULTS SAVED ===")
print(f"Saved to: {output_path}")
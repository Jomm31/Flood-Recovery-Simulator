import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("baseline_final_388.csv")


# ============================================================
# 2. DEFINE FEATURE GROUPS
# ============================================================

feature_groups = {

    "All 11 Features": [
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
    ],

    "Top 8 Features": [
        "elevation",
        "slope",
        "twi",
        "Lulc",
        "spi",
        "ndvi",
        "dtr",
        "rainfall"
    ],

    "Top 6 Features": [
        "elevation",
        "slope",
        "twi",
        "Lulc",
        "spi",
        "ndvi"
    ],

    "Top 4 Features": [
        "elevation",
        "slope",
        "twi",
        "Lulc"
    ],

    "Top 2 Features": [
        "elevation",
        "slope"
    ]
}


# ============================================================
# 3. TARGET
# ============================================================

y = df["flood"]


# ============================================================
# 4. SAME TRAIN / TEST SPLIT
# ============================================================

train_indices, test_indices = train_test_split(
    np.arange(len(df)),
    test_size=0.30,
    random_state=133
)


# ============================================================
# 5. TRAINING DATA
# ============================================================

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


print("=== FEATURE SELECTION EXPERIMENT ===")

print(f"Total observations: {len(df)}")
print(f"Training observations: {len(train_indices)}")
print(f"Testing observations: {len(test_indices)}")


# ============================================================
# 6. TUNED RANDOM FOREST
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
# 7. REPEATED STRATIFIED CV
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)


print("\n=== CROSS-VALIDATION DESIGN ===")
print("5 folds")
print("5 repeats")
print("25 validation runs per feature subset")


# ============================================================
# 8. SCORING
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
# 9. EVALUATE EACH FEATURE SUBSET
# ============================================================

for group_name, features in feature_groups.items():

    print("\n")
    print("=" * 70)
    print(group_name)
    print("=" * 70)

    print("Features:")
    print(", ".join(features))


    X_group = df[features]

    X_train = X_group.iloc[train_indices]


    model = RandomForestClassifier(
        **RF_PARAMS
    )


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
        "feature_group": group_name,
        "number_of_features": len(features),

        "features": ", ".join(features),

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
# 10. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")
print("=" * 100)
print("FINAL FEATURE SUBSET COMPARISON")
print("=" * 100)


print(
    results_df[
        [
            "feature_group",
            "number_of_features",
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
# 11. IDENTIFY BEST SUBSETS
# ============================================================

best_accuracy = results_df.loc[
    results_df["accuracy_mean"].idxmax()
]

best_f1 = results_df.loc[
    results_df["f1_mean"].idxmax()
]

best_auc = results_df.loc[
    results_df["auroc_mean"].idxmax()
]


print("\n")
print("=" * 70)
print("BEST FEATURE SUBSETS")
print("=" * 70)

print(
    f"\nBest mean Accuracy: "
    f"{best_accuracy['feature_group']} "
    f"({best_accuracy['accuracy_mean']:.6f})"
)

print(
    f"Best mean F1-score: "
    f"{best_f1['feature_group']} "
    f"({best_f1['f1_mean']:.6f})"
)

print(
    f"Best mean AUROC: "
    f"{best_auc['feature_group']} "
    f"({best_auc['auroc_mean']:.6f})"
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

output_path = "feature_selection_results.csv"

results_df.to_csv(
    output_path,
    index=False
)


print("\n=== RESULTS SAVED ===")
print(f"Saved to: {output_path}")
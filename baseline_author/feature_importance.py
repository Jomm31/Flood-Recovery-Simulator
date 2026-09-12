import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


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


# ============================================================
# 4. BEST TUNED RF WITHOUT SMOTE
# ============================================================

model = RandomForestClassifier(
    bootstrap=True,
    n_estimators=200,
    max_depth=10,
    max_features=2,
    min_samples_leaf=1,
    min_samples_split=5,
    random_state=42,
    n_jobs=1
)


print("=== FEATURE IMPORTANCE ANALYSIS ===")

print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")
print(f"Number of features: {len(FEATURE_COLS)}")


# ============================================================
# 5. FIT MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# 6. BUILT-IN RF FEATURE IMPORTANCE
# ============================================================

rf_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
})


rf_importance = rf_importance.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)


rf_importance["rank"] = (
    np.arange(len(rf_importance)) + 1
)


print("\n")
print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    rf_importance[
        ["rank", "feature", "importance"]
    ].to_string(index=False)
)


# ============================================================
# 7. PERMUTATION IMPORTANCE
# ============================================================

print("\n")
print("=" * 60)
print("CALCULATING PERMUTATION IMPORTANCE")
print("=" * 60)

print("Using the untouched test set.")
print("Repeats per feature: 30")


permutation = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=30,
    random_state=42,
    n_jobs=1
)


permutation_importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "permutation_mean": permutation.importances_mean,
    "permutation_std": permutation.importances_std
})


permutation_importance_df = (
    permutation_importance_df
    .sort_values(
        by="permutation_mean",
        ascending=False
    )
    .reset_index(drop=True)
)


permutation_importance_df["rank"] = (
    np.arange(len(permutation_importance_df)) + 1
)


print("\n")
print("=" * 60)
print("PERMUTATION IMPORTANCE")
print("=" * 60)

print(
    permutation_importance_df[
        [
            "rank",
            "feature",
            "permutation_mean",
            "permutation_std"
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. COMBINE BOTH IMPORTANCE METHODS
# ============================================================

combined = rf_importance.merge(
    permutation_importance_df[
        [
            "feature",
            "permutation_mean",
            "permutation_std"
        ]
    ],
    on="feature"
)


combined = combined.sort_values(
    by="importance",
    ascending=False
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

combined.to_csv(
    "feature_importance_results.csv",
    index=False
)


print("\n=== RESULTS SAVED ===")
print("feature_importance_results.csv")
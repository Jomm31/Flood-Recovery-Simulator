import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv("baseline_final_388.csv")

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
    "rainfall",
]

X = df[FEATURE_COLS]
y = df["flood"]

# --------------------------------------------------
# 2. Define three models
# --------------------------------------------------

models = {
    "Original Baseline RF": RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        max_features=2,
        min_samples_leaf=4,
        min_samples_split=10,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=1,
    ),

    "Tuned RF - No SMOTE": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features=2,
        min_samples_leaf=1,
        min_samples_split=5,
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    ),

    "RandomizedSearch RF": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="log2",
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True,
        criterion="entropy",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    ),
}

# --------------------------------------------------
# 3. Same repeated 5-fold splits for every model
# --------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42,
)

results = []

for model_name, model in models.items():

    print("\nRunning:", model_name)

    fold_number = 0

    for train_idx, val_idx in cv.split(X, y):

        fold_number += 1

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        results.append({
            "model": model_name,
            "fold": fold_number,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "auroc": roc_auc_score(y_val, y_prob),
        })

        print(
            f"  Fold {fold_number:02d}/25 complete"
        )

# --------------------------------------------------
# 4. Convert to DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)

# --------------------------------------------------
# 5. Calculate mean and standard deviation
# --------------------------------------------------

summary = (
    results_df
    .groupby("model")
    [["accuracy", "precision", "recall", "f1", "auroc"]]
    .agg(["mean", "std"])
)

print("\n==============================================")
print("REPEATED 5x5 CROSS-VALIDATION RESULTS")
print("==============================================")

print(summary)

# --------------------------------------------------
# 6. Save detailed and summary results
# --------------------------------------------------

results_df.to_csv(
    "random_search_repeated_cv_folds.csv",
    index=False,
)

summary.to_csv(
    "random_search_repeated_cv_summary.csv"
)

print("\nResults saved:")
print("random_search_repeated_cv_folds.csv")
print("random_search_repeated_cv_summary.csv")
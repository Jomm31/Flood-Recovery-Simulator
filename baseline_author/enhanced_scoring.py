import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    mean_squared_error,
    confusion_matrix
)


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


print("=== SCORING-FUNCTION EXPERIMENT ===")
print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 4. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    bootstrap=True,
    random_state=42,
    n_jobs=1
)


# ============================================================
# 5. SAME 72-PARAMETER GRID
# ============================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, 30],
    "max_features": [2, 4],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5]
}


# ============================================================
# 6. SCORING FUNCTIONS
# ============================================================

scoring_functions = [
    "accuracy",
    "f1",
    "roc_auc"
]


all_results = []


# ============================================================
# 7. RUN GRID SEARCH FOR EACH SCORING FUNCTION
# ============================================================

for scoring in scoring_functions:

    print("\n")
    print("=" * 60)
    print(f"GRID SEARCH USING SCORING = {scoring}")
    print("=" * 60)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(f"\nBest cross-validation {scoring}:")
    print(f"{grid_search.best_score_:.6f}")


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print("\nTest confusion matrix:")
    print(cm)

    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")


    # ========================================================
    # TEST METRICS
    # ========================================================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)


    print("\nTest results:")
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1-score:  {f1:.6f}")
    print(f"Kappa:     {kappa:.6f}")
    print(f"AUROC:     {auroc:.6f}")
    print(f"MSE:       {mse:.6f}")
    print(f"RMSE:      {rmse:.6f}")


    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_results.append({
        "scoring": scoring,
        "best_cv_score": grid_search.best_score_,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "kappa": kappa,
        "auroc": auroc,
        "mse": mse,
        "rmse": rmse,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_estimators": best_model.n_estimators,
        "max_depth": best_model.max_depth,
        "max_features": best_model.max_features,
        "min_samples_leaf": best_model.min_samples_leaf,
        "min_samples_split": best_model.min_samples_split
    })


# ============================================================
# 8. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(all_results)


print("\n")
print("=" * 60)
print("FINAL SCORING-FUNCTION COMPARISON")
print("=" * 60)

print(
    results_df[
        [
            "scoring",
            "best_cv_score",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "kappa",
            "auroc",
            "rmse"
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

output_path = "enhanced_scoring_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\n=== RESULTS SAVED ===")
print(f"Saved to: {output_path}")
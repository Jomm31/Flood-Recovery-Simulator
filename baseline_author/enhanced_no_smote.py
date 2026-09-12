import pandas as pd

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
# 2. DEFINE FEATURES AND TARGET
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
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=133
)


print("=== NO-SMOTE ENHANCED RF DATASET ===")
print(f"Dataset: baseline_final_388.csv")
print(f"Total observations: {len(df)}")
print(f"Number of features: {len(FEATURE_COLS)}")

print("\nFeatures:")
for i, feature in enumerate(FEATURE_COLS, start=1):
    print(f"{i}. {feature}")

print(f"\nTarget: {TARGET_COL}")


print("\n=== TRAIN / TEST SPLIT ===")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 4. HYPERPARAMETER SEARCH
# ============================================================

rf = RandomForestClassifier(
    bootstrap=True,
    random_state=42,
    n_jobs=1
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, 30],
    "max_features": [2, 4],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5]
}


print("\n=== ENHANCED RF WITHOUT SMOTE ===")
print("SMOTE: Disabled")
print("Random Forest: Enabled")
print("5-fold cross-validation: Enabled")


print("\n=== HYPERPARAMETER SEARCH SPACE ===")
for parameter, values in param_grid.items():
    print(f"{parameter}: {values}")


# ============================================================
# 5. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=1,
    verbose=1,
    return_train_score=True
)


print("\n=== STARTING GRID SEARCH ===")

grid_search.fit(X_train, y_train)


print("\n=== GRID SEARCH COMPLETE ===")

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")
print(f"{grid_search.best_score_:.6f}")


# ============================================================
# 6. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

print("\n=== BEST NO-SMOTE MODEL ===")
print(best_model)


# ============================================================
# 7. TEST SET PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n=== NO-SMOTE CONFUSION MATRIX ===")
print(cm)

print(f"\nTrue Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive (TP): {tp}")


# ============================================================
# 9. TEST METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_prob)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5


print("\n=== NO-SMOTE TEST RESULTS ===")

print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"Kappa:     {kappa:.6f}")
print(f"AUROC:     {auroc:.6f}")
print(f"MSE:       {mse:.6f}")
print(f"RMSE:      {rmse:.6f}")


# ============================================================
# 10. PROBABILITY CHECK
# ============================================================

print("\n=== PROBABILITY CHECK ===")

print(f"Minimum test probability: {y_prob.min()}")
print(f"Maximum test probability: {y_prob.max()}")
print(f"Unique test probability values: {len(set(y_prob))}")


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "model": ["Enhanced RF - No SMOTE"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "kappa": [kappa],
    "auroc": [auroc],
    "mse": [mse],
    "rmse": [rmse],
    "tn": [tn],
    "fp": [fp],
    "fn": [fn],
    "tp": [tp],
    "best_cv_accuracy": [grid_search.best_score_],
    "n_estimators": [best_model.n_estimators],
    "max_depth": [best_model.max_depth],
    "max_features": [best_model.max_features],
    "min_samples_leaf": [best_model.min_samples_leaf],
    "min_samples_split": [best_model.min_samples_split]
})

output_path = "enhanced_no_smote_results.csv"

results.to_csv(output_path, index=False)

print("\n=== RESULTS SAVED ===")
print(f"Saved to: {output_path}")
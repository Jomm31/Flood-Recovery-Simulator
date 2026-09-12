import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    mean_squared_error,
)

# --------------------------------------------------
# 1. Load the reproduced baseline dataset
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

TARGET_COL = "flood"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# --------------------------------------------------
# 2. Reproduce the exact baseline split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=133,
    stratify=None,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 3. Define the expanded Random Forest search space
# --------------------------------------------------

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=1,
)

param_distributions = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [None, 5, 10, 15, 20, 30, 40],
    "max_features": ["sqrt", "log2", 1, 2, 4, 6, 8],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 6, 8],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy", "log_loss"],
    "class_weight": [None, "balanced", "balanced_subsample"],
}

# --------------------------------------------------
# 4. Randomized hyperparameter search
# --------------------------------------------------

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=100,
    scoring="accuracy",
    cv=5,
    random_state=42,
    n_jobs=1,
    verbose=1,
    return_train_score=True,
)

random_search.fit(X_train, y_train)

# --------------------------------------------------
# 5. Display best parameters
# --------------------------------------------------

print("\nBest parameters:")
print(random_search.best_params_)

print("\nBest CV accuracy:")
print(random_search.best_score_)

# --------------------------------------------------
# 6. Evaluate the best model on the same test set
# --------------------------------------------------

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\nTest results:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print("Kappa    :", kappa)
print("AUROC    :", auc)
print("MSE      :", mse)
print("RMSE     :", rmse)

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
        "auroc",
        "mse",
        "rmse",
    ],
    "value": [
        accuracy,
        precision,
        recall,
        f1,
        kappa,
        auc,
        mse,
        rmse,
    ],
})

results.to_csv(
    "enhanced_random_search_results.csv",
    index=False,
)

print("\nResults saved to:")
print("enhanced_random_search_results.csv")
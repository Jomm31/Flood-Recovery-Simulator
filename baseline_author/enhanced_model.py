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

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# ENHANCED RANDOM FOREST MODEL
# ============================================================
#
# Dataset:
#   Mangkhaseum et al. baseline dataset
#   388 observations
#
# Enhancement implemented here:
#   1. SMOTE class balancing
#   2. Grid-search hyperparameter optimization
#
# Important:
#   SMOTE is applied INSIDE the cross-validation pipeline.
#   The test set remains completely untouched.
#
# ============================================================


# ============================================================
# STEP 1 — LOAD BASELINE DATASET
# ============================================================

DATA_PATH = "baseline_author/baseline_final_388.csv"

df = pd.read_csv(DATA_PATH)

print("=== ENHANCED RF DATASET ===")
print("Dataset:", DATA_PATH)
print("Total observations:", len(df))


# ============================================================
# STEP 2 — DEFINE FEATURES AND TARGET
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


print("\nNumber of features:", len(FEATURE_COLS))

print("\nFeatures:")
for i, feature in enumerate(FEATURE_COLS, start=1):
    print(f"{i}. {feature}")

print("\nTarget:", TARGET_COL)


# ============================================================
# STEP 3 — SAME TRAIN / TEST SPLIT AS BASELINE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=133
)

print("\n=== TRAIN / TEST SPLIT ===")

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# STEP 4 — VERIFY THE TEST SET IS UNTOUCHED
# ============================================================

print("\n=== TEST SET CHECK ===")

print("Test observations before enhancement:", len(X_test))

print("Test class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# STEP 5 — DEFINE ENHANCED RF PIPELINE
# ============================================================
#
# SMOTE is placed INSIDE the pipeline.
#
# During each CV fold:
#
#   CV training fold
#          ↓
#        SMOTE
#          ↓
#     Random Forest
#          ↓
#    validation fold
#
# This prevents synthetic samples from leaking into
# the validation data.
#
# ============================================================

enhanced_pipeline = Pipeline([
    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),

    (
        "rf",
        RandomForestClassifier(
            bootstrap=True,
            random_state=42,
            n_jobs=1
        )
    )
])


print("\n=== ENHANCED RF PIPELINE ===")

print("SMOTE: Enabled")
print("Random Forest: Enabled")
print("SMOTE is applied inside cross-validation.")


# ============================================================
# STEP 6 — DEFINE HYPERPARAMETER SEARCH
# ============================================================

param_grid = {

    "rf__n_estimators": [
        100,
        200
    ],

    "rf__max_depth": [
        10,
        20,
        30
    ],

    "rf__max_features": [
        2,
        4
    ],

    "rf__min_samples_leaf": [
        1,
        2,
        4
    ],

    "rf__min_samples_split": [
        2,
        5
    ]
}


print("\n=== HYPERPARAMETER SEARCH SPACE ===")

for parameter, values in param_grid.items():
    print(f"{parameter}: {values}")


# ============================================================
# STEP 7 — GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=enhanced_pipeline,

    param_grid=param_grid,

    scoring="accuracy",

    cv=5,

    n_jobs=1,

    verbose=1,

    return_train_score=True
)


print("\n=== STARTING SMOTE + GRID SEARCH ===")

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# STEP 8 — BEST MODEL
# ============================================================

print("\n=== GRID SEARCH COMPLETE ===")

print("\nBest parameters:")

print(grid_search.best_params_)

print("\nBest cross-validation accuracy:")

print(
    f"{grid_search.best_score_:.6f}"
)


# ============================================================
# STEP 9 — GET BEST ENHANCED MODEL
# ============================================================

best_model = grid_search.best_estimator_


print("\n=== BEST ENHANCED MODEL ===")

print(best_model)


# ============================================================
# STEP 10 — TEST PREDICTIONS
# ============================================================
#
# IMPORTANT:
# The test set was NOT used during SMOTE or grid search.
#
# ============================================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]


# ============================================================
# STEP 11 — CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n=== ENHANCED RF CONFUSION MATRIX ===")

print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ============================================================
# STEP 12 — CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

kappa = cohen_kappa_score(
    y_test,
    y_pred
)


# ============================================================
# STEP 13 — AUROC
# ============================================================

test_auroc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# STEP 14 — MSE AND RMSE
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5


# ============================================================
# STEP 15 — DISPLAY FINAL RESULTS
# ============================================================

print("\n=== ENHANCED RF TEST RESULTS ===")

print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"Kappa:     {kappa:.6f}")
print(f"AUROC:     {test_auroc:.6f}")
print(f"MSE:       {mse:.6f}")
print(f"RMSE:      {rmse:.6f}")


# ============================================================
# STEP 16 — PROBABILITY CHECK
# ============================================================

print("\n=== PROBABILITY CHECK ===")

print("Minimum test probability:", y_prob.min())
print("Maximum test probability:", y_prob.max())

print(
    "Unique test probability values:",
    len(set(y_prob))
)


# ============================================================
# STEP 17 — SAVE ENHANCED RESULTS
# ============================================================

enhanced_results = pd.DataFrame({

    "Metric": [

        "TN",
        "FP",
        "FN",
        "TP",

        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Kappa",

        "MSE",
        "RMSE",

        "Test_AUROC",

        "Best_CV_Accuracy"
    ],

    "Value": [

        tn,
        fp,
        fn,
        tp,

        accuracy,
        precision,
        recall,
        f1,
        kappa,

        mse,
        rmse,

        test_auroc,

        grid_search.best_score_
    ]
})


output_path = (
    "baseline_author/"
    "enhanced_model_results.csv"
)


enhanced_results.to_csv(
    output_path,
    index=False
)


print("\n=== RESULTS SAVED ===")

print(
    f"Saved to: {output_path}"
)
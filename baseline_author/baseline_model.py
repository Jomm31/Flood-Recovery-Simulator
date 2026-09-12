import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# AUTHOR BASELINE RANDOM FOREST MODEL
# ============================================================

DATA_PATH = "baseline_author/baseline_final_388.csv"

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


# ------------------------------------------------------------
# Load author-provided final modeling dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# Prepare predictors and target
# ------------------------------------------------------------

X = df[FEATURE_COLS]
y = df[TARGET_COL]


# ------------------------------------------------------------
# Reproduce the baseline train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=133
)


# ------------------------------------------------------------
# Author baseline Random Forest configuration
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    max_features=2,
    min_samples_leaf=4,
    min_samples_split=10,
    bootstrap=True,
    oob_score=True,
    random_state=42
)


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

rf_model.fit(X_train, y_train)


# ------------------------------------------------------------
# Display basic model information
# ------------------------------------------------------------

print("=== AUTHOR BASELINE MODEL ===")
print("Dataset:", DATA_PATH)
print("Observations:", len(df))
print("Number of features:", len(FEATURE_COLS))
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("OOB score:", rf_model.oob_score_)

print("\nModel:")
print(rf_model)


from sklearn.metrics import confusion_matrix

# Test predictions
y_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\n=== BASELINE CONFUSION MATRIX ===")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
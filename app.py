import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    mean_squared_error,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Flood Simulator", layout="wide")
st.title("🌊 Flood Susceptibility Simulator (Baseline Model)")
st.write("This dashboard replicates the Mangkhaseum et al. (2024) Random Forest baseline.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'Elevation', 'Slope', 'Aspect', 'Curvature', 'TWI', 'SPI', 'DTR',
    'Soil_Type', 'NDVI', 'LULC', 'Rainfall',
]

# ---------------------------------------------------------------------------
# Helpers — VIF, IGR, Pearson (match paper Section 3.4)
# ---------------------------------------------------------------------------

def calculate_vif(data, features):
    """Compute VIF and Tolerance for each feature (paper Eq. 1, Section 3.4).
    VIF_i = 1 / (1 - R²_i), where R²_i is from regressing feature i on all others."""
    values = data[features].astype(float)
    rows = []
    for feature in features:
        others = [f for f in features if f != feature]
        target = values[feature].to_numpy()
        predictors = values[others].to_numpy()
        model = LinearRegression().fit(predictors, target)
        r_squared = model.score(predictors, target)
        vif = np.inf if r_squared >= 1 else 1 / (1 - r_squared)
        rows.append({
            "Factor": feature,
            "VIF": round(vif, 3),
            "Tolerance": round(1 / vif if vif != np.inf else 0, 3),
        })
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def entropy(values):
    """Shannon entropy in bits."""
    probs = pd.Series(values).value_counts(normalize=True)
    return float(-(probs * np.log2(probs)).sum())


def calculate_igr(data, features, target_col):
    """Information Gain Ratio for each feature vs target (paper Section 3.4)."""
    target = data[target_col].astype(int).to_numpy()
    target_ent = entropy(target)
    rows = []
    for feature in features:
        feat_vals = data[feature].astype(int).to_numpy()
        cond_ent = 0.0
        split_info = entropy(feat_vals)
        for v in np.unique(feat_vals):
            mask = feat_vals == v
            cond_ent += mask.mean() * entropy(target[mask])
        ig = target_ent - cond_ent
        igr = ig / split_info if split_info > 0 else 0.0
        rows.append({
            "Factor": feature,
            "Information Gain": round(ig, 4),
            "IGR": round(igr, 4),
        })
    return pd.DataFrame(rows).sort_values("IGR", ascending=False)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv('data/baseline_training_data.csv')


try:
    df = load_data()
    st.success(f"✅ Training dataset loaded — {len(df)} rows, {df.shape[1]} columns")

    st.write("**Data Preview (first 5 rows):**")
    st.dataframe(df.head())

    X = df[FEATURE_COLS]
    y = df['Flood_Class']

    # ------------------------------------------------------------------
    # STEP 1 — Multicollinearity & Feature Selection (paper Section 3.4)
    # ------------------------------------------------------------------
    st.header("📐 Step 1 — Factor Selection & Evaluation")

    col_vif, col_igr = st.columns(2)

    with col_vif:
        st.subheader("VIF / Tolerance")
        vif_df = calculate_vif(df, FEATURE_COLS)
        st.dataframe(vif_df, use_container_width=True)
        all_pass_vif = (vif_df["VIF"] < 10).all() and (vif_df["Tolerance"] > 0.1).all()
        if all_pass_vif:
            st.success("All factors pass: VIF < 10 and Tolerance > 0.1 — no multicollinearity detected.")
        else:
            st.warning("Some factors have VIF ≥ 10 or Tolerance ≤ 0.1 — consider removing them.")

    with col_igr:
        st.subheader("Information Gain Ratio (IGR)")
        igr_df = calculate_igr(df, FEATURE_COLS, 'Flood_Class')
        st.dataframe(igr_df, use_container_width=True)
        st.info("Higher IGR = stronger influence on flood prediction.")

    # Pearson Correlation
    st.subheader("Pearson Correlation Matrix")
    corr_matrix = df[FEATURE_COLS].corr()
    st.dataframe(corr_matrix.round(3), use_container_width=True)

    # ------------------------------------------------------------------
    # STEP 2 — Model Training with GridSearchCV (paper Section 3.5.2)
    # ------------------------------------------------------------------
    st.header("🤖 Step 2 — Model Training & Evaluation")

    # Stratified 70/30 split (paper Section 3.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    st.write(f"**Training:** {len(X_train)} points | **Testing:** {len(X_test)} points "
             f"(stratified 70/30 split)")

    # Feature Scaling (Fig. 2: "Training Dataset -> Feature Scaling -> Scaled
    # Training data / Scaled Test Data"). Fitted on the training partition only,
    # then applied to both train and test, to avoid leaking test-set statistics
    # into the scaler -- matches the paper's stated convention.
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    if st.button("🚀 Train Baseline Model (with GridSearchCV)"):
        with st.spinner("Running GridSearchCV — this may take a minute..."):

            # GridSearchCV — paper mentions hyperparameter tuning via GridSearchCV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            param_grid = {
                "n_estimators": [100, 300, 600],
                "max_depth": [None, 8, 12],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", None],
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_

            # Predictions
            predictions = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]

            # ----------------------------------------------------------
            # STEP 3 — Full Metrics (paper Section 3.6 / 4.2)
            # ----------------------------------------------------------
            st.header("📊 Step 3 — Performance Metrics")

            # Best params from GridSearchCV
            st.subheader("GridSearchCV Results")
            st.write(f"**Best CV Accuracy:** {grid_search.best_score_:.4f}")
            st.write(f"**Best Parameters:** `{grid_search.best_params_}`")

            # Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

            # All 9 metrics from the paper
            acc = accuracy_score(y_test, predictions)
            auroc = roc_auc_score(y_test, probabilities)
            f1 = f1_score(y_test, predictions)
            kappa = cohen_kappa_score(y_test, predictions)
            sensitivity = recall_score(y_test, predictions)        # TP / (TP + FN)
            specificity = tn / (tn + fp)                           # TN / (TN + FP)
            precision = precision_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = mse ** 0.5

            # Training AUROC (paper reports train AUROC = 1.000 for RF)
            train_proba = model.predict_proba(X_train_scaled)[:, 1]
            train_auroc = roc_auc_score(y_train, train_proba)

            # Display metrics in columns
            st.subheader("Test Set Metrics (vs Paper RF)")
            metrics_data = {
                "Metric": ["Accuracy", "AUROC (test)", "AUROC (train)", "F1-Score",
                           "Kappa", "Sensitivity", "Specificity", "Precision", "MSE", "RMSE"],
                "Your Result": [f"{acc:.4f}", f"{auroc:.4f}", f"{train_auroc:.4f}", f"{f1:.4f}",
                                f"{kappa:.4f}", f"{sensitivity:.4f}", f"{specificity:.4f}",
                                f"{precision:.4f}", f"{mse:.4f}", f"{rmse:.4f}"],
                "Paper RF": ["0.957", "0.993", "1.000", "0.962",
                             "0.914", "0.969", "—", "—", "0.207", "0.043"],
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Confusion Matrix display
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual: Non-Flood", "Actual: Flood"],
                columns=["Predicted: Non-Flood", "Predicted: Flood"],
            )
            st.dataframe(cm_df, use_container_width=True)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, predictions, digits=4,
                                           target_names=["Non-Flood", "Flood"])
            st.code(report, language="text")

            # Feature Importance (from RF)
            st.subheader("RF Feature Importance (Gini)")
            importance_df = pd.DataFrame({
                "Factor": FEATURE_COLS,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Factor"), horizontal=True)

except FileNotFoundError:
    st.error(
        "Could not find data/baseline_training_data.csv. "
        "Make sure your data_prep.py script ran successfully!"
    )

import pandas as pd
df = pd.read_csv('data/baseline_training_data.csv')

print("Aspect distribution by Flood_Class:")
print(pd.crosstab(df['Aspect'], df['Flood_Class']))
print()
print("LULC value counts:")
print(df['LULC'].value_counts())
print()
print("LULC distribution by Flood_Class:")
print(pd.crosstab(df['LULC'], df['Flood_Class']))
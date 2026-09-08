from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "NamNgum_baseline_11factors_fixed_terrain.csv"
OUTPUT = ROOT / "data" / "processed" / "paper_baseline_training_data.csv"
RESULTS = ROOT / "results"
FEATURES = [
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "twi",
    "spi",
    "dtr",
    "soil_type",
    "ndvi",
    "lulc",
    "rainfall",
]


def natural_breaks(values, classes=5):
    values = pd.Series(values, dtype=float)
    unique_values = values.nunique()
    if unique_values < 2:
        return np.zeros(len(values), dtype=int)
    classes = min(classes, unique_values)
    try:
        import jenkspy

        breaks = jenkspy.jenks_breaks(values.to_numpy(), n_classes=classes)
        if len(set(breaks)) < 2:
            return np.zeros(len(values), dtype=int)
        return pd.cut(
            values,
            bins=breaks,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        ).fillna(0).astype(int).to_numpy()
    except ImportError:
        estimator = KBinsDiscretizer(
            n_bins=classes,
            encode="ordinal",
            strategy="kmeans",
            random_state=42,
        )
        return estimator.fit_transform(values.to_numpy().reshape(-1, 1)).ravel().astype(int)


def quantile_classes(values, classes=5):
    result = pd.qcut(
        pd.Series(values, dtype=float),
        q=classes,
        labels=False,
        duplicates="drop",
    )
    return result.fillna(0).astype(int).to_numpy()


def reclassify(data):
    output = data.copy()
    for column in ["elevation", "slope", "aspect", "rainfall"]:
        output[column] = natural_breaks(output[column])
    for column in ["ndvi", "twi", "spi", "dtr"]:
        output[column] = quantile_classes(output[column])

    curvature = output["curvature"]
    output["curvature"] = np.select(
        [curvature < -0.01, curvature.abs() <= 0.01, curvature > 0.01],
        [0, 1, 2],
        default=1,
    )
    return output


def calculate_vif(data):
    values = data[FEATURES].astype(float)
    rows = []
    for feature in FEATURES:
        other_features = [name for name in FEATURES if name != feature]
        target = values[feature].to_numpy()
        predictors = values[other_features].to_numpy()
        model = LinearRegression().fit(predictors, target)
        r_squared = model.score(predictors, target)
        vif = np.inf if r_squared >= 1 else 1 / (1 - r_squared)
        rows.append({"Feature": feature, "VIF": vif, "Tolerance": 1 / vif})
    return pd.DataFrame(rows)


def entropy(values):
    probabilities = pd.Series(values).value_counts(normalize=True)
    return float(-(probabilities * np.log2(probabilities)).sum())


def calculate_information_gain_ratio(data):
    target = data["flood_class"].astype(int).to_numpy()
    target_entropy = entropy(target)
    rows = []
    for feature in FEATURES:
        feature_values = data[feature].astype(int).to_numpy()
        conditional_entropy = 0.0
        split_information = entropy(feature_values)
        for value in np.unique(feature_values):
            mask = feature_values == value
            conditional_entropy += mask.mean() * entropy(target[mask])
        information_gain = target_entropy - conditional_entropy
        ratio = information_gain / split_information if split_information else 0.0
        rows.append(
            {
                "Feature": feature,
                "Information_Gain": information_gain,
                "Information_Gain_Ratio": ratio,
            }
        )
    return pd.DataFrame(rows).sort_values("Information_Gain_Ratio", ascending=False)


def main():
    data = pd.read_csv(INPUT)
    data.columns = data.columns.str.strip().str.lower()
    required = set(FEATURES + ["latitude", "longitude", "flood_class"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(data) != 776:
        raise ValueError(f"Expected 776 inventory points, found {len(data)}")
    if data[FEATURES + ["flood_class"]].isna().any().any():
        raise ValueError("Training data contains missing values")

    classified = reclassify(data)
    classified.to_csv(OUTPUT, index=False)

    vif = calculate_vif(classified)
    igr = calculate_information_gain_ratio(classified)
    RESULTS.mkdir(exist_ok=True)
    vif.to_csv(RESULTS / "vif_tolerance.csv", index=False)
    igr.to_csv(RESULTS / "information_gain_proxy.csv", index=False)

    train_idx, test_idx = train_test_split(
        np.arange(len(classified)),
        test_size=0.30,
        random_state=42,
        stratify=classified["flood_class"],
    )
    training_features = classified[FEATURES].iloc[train_idx]
    training_labels = classified["flood_class"].iloc[train_idx]
    test_features = classified[FEATURES].iloc[test_idx]
    labels = classified["flood_class"].iloc[test_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [300, 600],
            "max_depth": [None, 8, 12],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", None],
        },
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(training_features, training_labels)
    model = search.best_estimator_
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    metrics = {
        "rows": len(classified),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "cv_accuracy": search.best_score_,
        "best_parameters": search.best_params_,
        "accuracy": accuracy_score(labels, predictions),
        "auroc": roc_auc_score(labels, probabilities),
        "f1": f1_score(labels, predictions),
        "sensitivity": recall_score(labels, predictions),
        "specificity": tn / (tn + fp),
        "precision": precision_score(labels, predictions),
        "kappa": cohen_kappa_score(labels, predictions),
        "mse": mean_squared_error(labels, predictions),
        "rmse": mean_squared_error(labels, predictions) ** 0.5,
    }
    pd.DataFrame([metrics]).to_csv(RESULTS / "paper_baseline_metrics.csv", index=False)

    print("Paper-style adapted baseline complete")
    print(f"Saved: {OUTPUT}")
    print(metrics)
    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))
    print(classification_report(labels, predictions, digits=4))
    print("VIF/tolerance:")
    print(vif.to_string(index=False))
    print("Information-gain ratio:")
    print(igr.to_string(index=False))


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

try:
    import jenkspy
    HAS_JENKS = True
except ImportError:
    HAS_JENKS = False
    from sklearn.preprocessing import KBinsDiscretizer

print("Starting data preparation (11-factor baseline)...")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_factor(path, value_name):
    """Load a point-sampled factor CSV and rename its value column to
    something predictable, regardless of whether the source used
    'SAMPLE_1', 'slope', 'elevation', etc."""
    df = pd.read_csv(path)
    id_cols = {'X', 'Y', 'id'}
    value_col = [c for c in df.columns if c not in id_cols][0]
    return df.rename(columns={value_col: value_name})

def match_by_xy(reference_xy, factor_df, value_name):
    """KDTree-match a factor dataframe (X, Y, value_name) onto the
    inventory's coordinates."""
    tree = KDTree(factor_df[['X', 'Y']].values)
    _, idx = tree.query(reference_xy)
    return factor_df.iloc[idx].reset_index(drop=True)[value_name]

def classify_natural_breaks(series, k=10):
    """Reclassify a continuous factor into k natural-breaks (Jenks) classes,
    matching the paper's treatment of elevation/slope/aspect/rainfall.
    Default k=10 per Figure 3 (elevation/slope/aspect use 10 classes)."""
    values = series.values
    if HAS_JENKS:
        breaks = jenkspy.jenks_breaks(values, n_classes=k)
        return pd.cut(series, bins=breaks, labels=False, include_lowest=True)
    else:
        # Fallback: 1D k-means binning is a close practical approximation
        # of Jenks natural breaks when jenkspy isn't installed.
        est = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans')
        return est.fit_transform(values.reshape(-1, 1)).astype(int).ravel()

def classify_quantile(series, k=5):
    """Reclassify a continuous factor into k equal-count quantile classes,
    matching the paper's treatment of NDVI/TWI/SPI/DTR.
    SPI uses k=10 per Figure 3; NDVI/TWI/DTR use k=5."""
    return pd.qcut(series, q=k, labels=False, duplicates='drop')

def classify_curvature(series):
    """Manual 3-class split: concave (<0), flat (~0), convex (>0),
    matching the paper's manual classification of curvature."""
    flat_tol = 0.01
    conditions = [series < -flat_tol, series.abs() <= flat_tol, series > flat_tol]
    choices = [0, 1, 2]  # 0=concave, 1=flat, 2=convex
    return np.select(conditions, choices, default=1)

# ---------------------------------------------------------------------------
# 1. Load the Flood Inventory (776 points)
# ---------------------------------------------------------------------------
inventory = pd.read_csv('data/Sentinel_1__NamNgum_Flood_Inventory_Baseline.csv')
inv_xy = inventory[['longitude', 'latitude']].values

# ---------------------------------------------------------------------------
# 2. Load and match all 11 conditioning factors
#    NOTE: filenames below match what's on disk as of this build. If you
#    rename/re-export any factor file, update the path here.
# ---------------------------------------------------------------------------
factor_files_xy = {
    'Elevation': 'data/ALOS_Elevation_Dataset.csv',
    'Slope':     'data/ALOS_Slope_Dataset.csv',
    'Aspect':    'data/ALOS_Aspect_Dataset.csv',
    'Curvature': 'data/ALOS_Curvature_Dataset.csv',
    'TWI':       'data/ALOS_TWI_Dataset.csv',
    'SPI':       'data/ALOS_SPI_Dataset.csv',
    'DTR':       'data/ALOS_DTR_Dataset.csv',
    'Soil_Type': 'data/NamNgum_FAO_Soil.csv',
    'NDVI':      'data/Landsat8_NDVI.csv',
    'LULC':      'data/Nam_Ngum_Baseline_Sentinel-2.csv',
}

raw_columns = {}
for name, path in factor_files_xy.items():
    print(f"Matching {name} from {path} ...")
    factor_df = load_factor(path, name)
    raw_columns[name] = match_by_xy(inv_xy, factor_df, name)

# Rainfall uses longitude/latitude instead of X/Y, and has its own value col
print("Matching Rainfall from data/ERA5_Rainfall.csv ...")
rain_data = pd.read_csv('data/ERA5_Rainfall.csv')
rain_tree = KDTree(rain_data[['longitude', 'latitude']].values)
_, rain_idx = rain_tree.query(inv_xy)
raw_columns['Rainfall'] = rain_data.iloc[rain_idx].reset_index(drop=True)['tp']

# ---------------------------------------------------------------------------
# 3. Compile the raw (pre-reclassification) dataset
# ---------------------------------------------------------------------------
final_data = pd.DataFrame({
    'Latitude': inventory['latitude'],
    'Longitude': inventory['longitude'],
    **raw_columns,
    'Flood_Class': inventory['flood_class'],
})

# ---------------------------------------------------------------------------
# 4. Handle missing values
# ---------------------------------------------------------------------------
n_before = len(final_data)
final_data = final_data.dropna()
n_dropped = n_before - len(final_data)
if n_dropped:
    print(f"Dropped {n_dropped} row(s) with missing values across the 11 factors.")

# ---------------------------------------------------------------------------
# 5. Reclassification (matches the baseline paper's methodology):
#    - Natural breaks: Elevation (k=10), Slope (k=10), Aspect (k=10),
#                      Rainfall (k=5) — Figure 3 shows 10 classes for
#                      elevation/slope/aspect; rainfall class count not
#                      explicitly stated so kept at 5.
#    - Quantile division: NDVI (k=5), TWI (k=5), SPI (k=10), DTR (k=5)
#                         Figure 3 shows 10 classes for SPI.
#    - Manual: LULC, Soil_Type (already categorical class codes -- kept as-is)
#              Curvature (concave/flat/convex)
# ---------------------------------------------------------------------------
print("Applying reclassification...")

# Natural breaks — Elevation/Slope/Aspect use k=10 per Figure 3; Rainfall k=5
for col in ['Elevation', 'Slope', 'Aspect']:
    final_data[col] = classify_natural_breaks(final_data[col], k=10)

final_data['Rainfall'] = classify_natural_breaks(final_data['Rainfall'], k=5)

# Quantile — SPI uses k=10 per Figure 3; NDVI/TWI/DTR use k=5
for col in ['NDVI', 'TWI', 'DTR']:
    final_data[col] = classify_quantile(final_data[col], k=5)

final_data['SPI'] = classify_quantile(final_data['SPI'], k=10)

final_data['Curvature'] = classify_curvature(final_data['Curvature'])
# LULC and Soil_Type are left as their original class codes (manual/categorical).

# ---------------------------------------------------------------------------
# 6. Save the ready-to-train dataset
# ---------------------------------------------------------------------------
final_data.to_csv('data/baseline_training_data.csv', index=False)
print(f"Successfully generated baseline_training_data.csv! "
      f"({len(final_data)} rows, {final_data.shape[1]} columns, 11 conditioning factors)")
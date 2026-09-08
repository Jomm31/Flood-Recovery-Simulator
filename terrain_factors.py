"""
Computes the 5 missing baseline conditioning factors (Aspect, Curvature, TWI,
SPI, Distance-to-River) from the raw ALOS-PALSAR DEM raster, then samples
them at the flood inventory point locations and merges into the existing
6-factor baseline_training_data.csv.

Requires: rasterio, richdem, numpy, pandas, scipy
    pip install rasterio richdem numpy pandas scipy

Usage:
    python terrain_factors.py data/alos_dem_nnrb.tif
"""

import sys
import numpy as np
import pandas as pd
import rasterio
import richdem as rd
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

def load_dem(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    return arr, transform

def pixel_to_lonlat(row, col, transform):
    lon, lat = rasterio.transform.xy(transform, row, col)
    return lon, lat

def lonlat_to_pixel(lon, lat, transform):
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    return row, col

def compute_terrain_factors(dem_path):
    print(f"Loading DEM: {dem_path}")
    arr, transform = load_dem(dem_path)
    rd_dem = rd.rdarray(arr, no_data=np.nan)
    rd_dem.geotransform = transform.to_gdal()

    print("Computing slope...")
    slope = rd.TerrainAttribute(rd_dem, attrib='slope_degrees')

    print("Computing aspect...")
    aspect = rd.TerrainAttribute(rd_dem, attrib='aspect')

    print("Computing curvature...")
    curvature = rd.TerrainAttribute(rd_dem, attrib='curvature')

    print("Computing flow accumulation (needed for TWI, SPI, DTR)...")
    filled = rd.FillDepressions(rd_dem, epsilon=True, in_place=False)
    flow_acc = rd.FlowAccumulation(filled, method='D8')

    # Pixel size in meters (approx, for TWI/SPI formulas)
    pixel_size = abs(transform.a)  # degrees; for 12.5m ALOS data this is
    # typically already in meters if the raster is in a projected CRS.
    # If your raster is in geographic (lat/lon) coords, reproject to a
    # projected CRS (e.g. UTM 48N for Laos) BEFORE running this script,
    # or the slope/TWI/SPI values will be wrong.

    slope_rad = np.radians(np.array(slope))
    catchment_area = np.array(flow_acc) * (pixel_size ** 2)

    print("Computing TWI...")
    twi = np.log((catchment_area + 1e-6) / (np.tan(slope_rad) + 1e-6))

    print("Computing SPI...")
    spi = catchment_area * np.tan(slope_rad)

    print("Deriving stream network + Distance-to-River...")
    # Threshold flow accumulation to define the stream network.
    # This threshold is empirical -- adjust based on visual inspection
    # against known rivers in the basin (start around the 99th percentile).
    stream_threshold = np.nanpercentile(np.array(flow_acc), 99)
    stream_mask = np.array(flow_acc) >= stream_threshold
    dtr_pixels = distance_transform_edt(~stream_mask)
    dtr_meters = dtr_pixels * pixel_size

    return {
        'transform': transform,
        'shape': arr.shape,
        'aspect': np.array(aspect),
        'curvature': np.array(curvature),
        'twi': twi,
        'spi': spi,
        'dtr': dtr_meters,
    }

def sample_at_points(factors, lon_arr, lat_arr):
    transform = factors['transform']
    rows_max, cols_max = factors['shape']
    results = {k: [] for k in ['aspect', 'curvature', 'twi', 'spi', 'dtr']}

    for lon, lat in zip(lon_arr, lat_arr):
        row, col = lonlat_to_pixel(lon, lat, transform)
        row = int(np.clip(row, 0, rows_max - 1))
        col = int(np.clip(col, 0, cols_max - 1))
        for k in results:
            results[k].append(factors[k][row, col])

    return pd.DataFrame(results)

def main(dem_path):
    inventory = pd.read_csv('data/Sentinel_1__NamNgum_Flood_Inventory_Baseline.csv')
    existing = pd.read_csv('data/baseline_training_data.csv')

    factors = compute_terrain_factors(dem_path)
    sampled = sample_at_points(factors, inventory['longitude'].values, inventory['latitude'].values)

    # Merge on row order -- both are derived from the same inventory,
    # in the same order, so a simple concat is safe here.
    assert len(sampled) == len(existing), "Row count mismatch -- check inventory alignment"
    merged = pd.concat([existing.reset_index(drop=True), sampled.reset_index(drop=True)], axis=1)

    n_before = len(merged)
    merged = merged.dropna()
    if n_before != len(merged):
        print(f"Dropped {n_before - len(merged)} row(s) with missing terrain values (edge-of-raster points).")

    merged.to_csv('data/baseline_training_data_full11.csv', index=False)
    print(f"Saved data/baseline_training_data_full11.csv with {len(merged)} rows, "
          f"{merged.shape[1]} columns (should now have all 11 factors).")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python terrain_factors.py <path_to_dem.tif>")
        sys.exit(1)
    main(sys.argv[1])

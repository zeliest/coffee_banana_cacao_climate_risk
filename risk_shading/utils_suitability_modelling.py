from pygbif import occurrences as gbif
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

import time
import os

def clean_species_name(name: str) -> str:
    """
    Standardise species name & fix 'Coffee' to 'Coffea'.
    """
    name = name.replace('_', ' ').replace('×', '').strip()
    parts = name.split()
    if len(parts) >= 2:
        genus, species = parts[0].capitalize(), parts[1].lower()
        if genus == "Coffee":
            genus = "Coffea"
        return f"{genus} {species}"
    return name

def fetch_gbif_species(species: str, wkt_bbox: str, year_min=1960, max_records=None, delay=0.5) -> pd.DataFrame:
    """
    Fetch GBIF records >year_min for a species inside bbox.
    """
    all_results, offset, batch_size = [], 0, 300

    while True:
        res = gbif.search(
            scientificName=species,
            hasCoordinate=True,
            geometry=wkt_bbox,
            limit=batch_size,
            offset=offset
        )
        results = res.get("results", [])
        if not results:
            break

        filtered = [r for r in results if r.get("year") and r["year"] > year_min]
        all_results.extend(filtered)
        offset += batch_size

        print(f"  📦 {offset} total | {len(all_results)} kept >{year_min} for {species}")
        if max_records and offset >= max_records:
            break
        time.sleep(delay)

    return pd.json_normalize(all_results)

def load_existing(path="occurrences_raw.parquet"):
    """
    Load existing GBIF data if available.
    """
    if os.path.exists(path):
        df = pd.read_parquet(path)
        fetched = set(df['species_query'].unique()) if not df.empty else set()
        print(f"📄 Loaded {len(fetched)} species from {path}")
    else:
        df, fetched = pd.DataFrame(), set()
        print(f"📄 No existing data at {path}, starting fresh.")
    return df, fetched


def generate_background_points(lon_min, lon_max, lat_min, lat_max, n_points=5000, seed=42):
    """
    Generate random background points (pseudo-absences) within a given bounding box.

    Parameters:
        lon_min, lon_max: float — longitude bounds
        lat_min, lat_max: float — latitude bounds
        n_points: int — number of background points to generate
        seed: int — random seed for reproducibility

    Returns:
        pd.DataFrame with columns ['lon', 'lat', 'presence'] where presence=0
    """
    np.random.seed(seed)
    bg_lons = np.random.uniform(lon_min, lon_max, n_points)
    bg_lats = np.random.uniform(lat_min, lat_max, n_points)

    background = pd.DataFrame({
        'lon': bg_lons,
        'lat': bg_lats,
        'presence': 0
    })
    print(f"✅ Generated {len(background)} background points.")
    return background




def extract_predictors_batch(lons, lats, bio_stack, predictors):
    """
    Extract predictors for many points at once using KDTree.
    """
    xv, yv = np.meshgrid(bio_stack.x.values, bio_stack.y.values)
    flat_coords = np.column_stack([xv.ravel(), yv.ravel()])

    tree = cKDTree(flat_coords)
    query_points = np.column_stack([lons, lats])
    dists, idxs = tree.query(query_points)

    results = []
    for pred in predictors:
        values = bio_stack[pred].values.ravel()[idxs]
        results.append(values)

    return np.column_stack(results)


def build_training_dataset(
    presence_df,
    bio_stack,
    bbox,
    selected_predictors,
    background_ratio=5,
    max_background=5000,
    seed=42
):
    """
    Batch extraction using KDTree, with optional max_background cap.
    """
    lon_min, lon_max, lat_min, lat_max = bbox

    # 🎯 Filter presence points
    presence_in_bbox = presence_df[
        (presence_df['lon'] >= lon_min) &
        (presence_df['lon'] <= lon_max) &
        (presence_df['lat'] >= lat_min) &
        (presence_df['lat'] <= lat_max)
    ].copy()

    print(f"✅ Found {len(presence_in_bbox)} presence points in study area.")

    # 📋 Background points
    n_background = len(presence_in_bbox) * background_ratio
    np.random.seed(seed)

    bg_lons = np.random.uniform(lon_min, lon_max, n_background)
    bg_lats = np.random.uniform(lat_min, lat_max, n_background)

    # 📋 Combine presence & background
    all_lons = np.concatenate([presence_in_bbox.lon.values, bg_lons])
    all_lats = np.concatenate([presence_in_bbox.lat.values, bg_lats])
    labels = np.array([1]*len(presence_in_bbox) + [0]*n_background)
    species = list(presence_in_bbox.species) + ["background"]*n_background

    print(f"🚀 Querying predictors for {len(all_lons)} points…")
    predictors_array = extract_predictors_batch(all_lons, all_lats, bio_stack, selected_predictors)

    df = pd.DataFrame(predictors_array, columns=selected_predictors)
    df["lon"] = all_lons
    df["lat"] = all_lats
    df["presence"] = labels
    df["species"] = species

    # drop invalid rows
    df = df[np.all(np.isfinite(df[selected_predictors]), axis=1)].reset_index(drop=True)

    # cap background points
    bg = df[df.presence == 0].copy()
    pres = df[df.presence == 1].copy()

    if len(bg) > max_background:
        bg = bg.sample(n=max_background, random_state=seed)

    df_final = pd.concat([pres, bg], ignore_index=True)

    print(f"🎉 Final dataset: {len(df_final)} records.")
    print(df_final["presence"].value_counts())
    print(df_final["species"].value_counts())

    return df_final

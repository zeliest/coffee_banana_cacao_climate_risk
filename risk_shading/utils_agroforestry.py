import numpy as np
import random
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from config import RASTER_PATHS
import rasterio
from rasterio.windows import from_bounds
from shapely import wkt
import matplotlib.pyplot as plt
from climada.entity import ImpactFunc, ImpactFuncSet


# === Land cover value mapping ===
LAND_COVER_LOOKUP = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    # Extend as needed
}

# === Default plot size distribution ===
DEFAULT_PLOT_SIZE_DISTRIBUTION = [
    (300, 800),
    (800, 1200),
    (1200, 1800),
    (1800, 2500),
]

def get_paths_for_country(country_name: str) -> dict:
    return RASTER_PATHS.get(country_name.lower(), {})

def validate_paths(paths_dict) -> bool:
    all_exist = True
    for category, path_list in paths_dict.items():
        for path in path_list:
            if not path.exists():
                print(f"\u26a0\ufe0f Missing file for '{category}': {path}")
                all_exist = False
    return all_exist

def sample_plot_size(distribution=DEFAULT_PLOT_SIZE_DISTRIBUTION):
    lower, upper = random.choice(distribution)
    return random.uniform(lower, upper)

def meters_to_degrees(meters):
    return meters / 111_000

def extract_mean_raster_value_by_area(paths, lon, lat, plot_area_m2=4184):
    side_m = np.sqrt(plot_area_m2)
    buffer_deg = meters_to_degrees(side_m) / 2

    for path in paths:
        try:
            with rasterio.open(path) as src:
                if not (src.bounds.left <= lon <= src.bounds.right and src.bounds.bottom <= lat <= src.bounds.top):
                    continue

                bounds = (
                    lon - buffer_deg,
                    lat - buffer_deg,
                    lon + buffer_deg,
                    lat + buffer_deg,
                )
                window = from_bounds(*bounds, transform=src.transform)
                data = src.read(1, window=window, masked=True)

                if data.count() > 0:
                    mean_val = data.mean()
                    print(f"✅ Extracted from: {path.name}, mean: {mean_val:.2f}")
                    return float(mean_val)
                else:
                    print(f"\u26a0\ufe0f No valid data in window: {path.name}")
        except Exception as e:
            print(f"❌ Failed to read {path.name}: {e}")

    return np.nan

def get_plot_satellite_features(lon, lat, paths_dict, plot_area_m2=4184):
    return {
        "land_use_class": extract_mean_raster_value_by_area(paths_dict["land_cover"], lon, lat, plot_area_m2),
        "canopy_height": extract_mean_raster_value_by_area(paths_dict["canopy_height"], lon, lat, plot_area_m2),
        "forest_cover": extract_mean_raster_value_by_area(paths_dict["forest_cover"], lon, lat, plot_area_m2),
    }

def apply_satellite_features_to_geodf(gdf, country="dominican_republic", plot_area_col="plot_area_m2"):
    paths = get_paths_for_country(country)
    if not validate_paths(paths):
        print("❌ Not all raster files found. Check the paths before proceeding.")
        return gdf.copy()

    results = []
    for _, row in gdf.iterrows():
        lon, lat = row["lon"], row["lat"]
        plot_area = row[plot_area_col] if pd.notna(row[plot_area_col]) else sample_plot_size()

        try:
            features = get_plot_satellite_features(lon, lat, paths, plot_area_m2=plot_area)
            features["land_use_class"] = LAND_COVER_LOOKUP.get(features["land_use_class"], "Unknown")
            features["lon"] = lon
            features["lat"] = lat
            features["plot_area_m2"] = plot_area
            results.append(features)
        except Exception as e:
            print(f"⚠️ Failed to process point at ({lat}, {lon}): {e}")

    return pd.DataFrame(results)

def classify_agroforestry(row):
    if pd.isna(row["canopy_height"]) or pd.isna(row["forest_cover"]) or pd.isna(row["land_use_class"]):
        return "Unknown"
    
    if row["canopy_height"] >= 6 and row["forest_cover"] >= 50 and row["land_use_class"] in ["Tree cover", "Cropland"]:
        return "Likely agroforestry"
    
    if row["canopy_height"] < 3 and row["forest_cover"] < 30 and row["land_use_class"] in ["Grassland", "Built-up"]:
        return "Likely sun-grown"
    
    return "Mixed or unclear"

def extract_features_with_composition(
    gdf,
    agroforest_dict,
    raster_source="dominican_republic",
    lon_col="lon",
    lat_col="lat",
    plot_area_col="plot_area_m2",
    system_col="system_id",
    raster_paths=None
):
    paths = raster_paths or get_paths_for_country(raster_source)
    if not validate_paths(paths):
        print("❌ Raster paths not valid.")
        return gdf.copy()

    results = []

    for _, row in gdf.iterrows():
        lon, lat = row[lon_col], row[lat_col]
        plot_area = row[plot_area_col] if pd.notna(row[plot_area_col]) else sample_plot_size()
        system_id = row[system_col]
        species_df = agroforest_dict.get(system_id)

        if species_df is None:
            print(f"⚠️ No agroforestry system defined for system_id={system_id}")
            continue

        try:
            features = get_plot_satellite_features(lon, lat, paths, plot_area_m2=plot_area)
            features.update({
                "lon": lon,
                "lat": lat,
                "plot_area_m2": plot_area,
                "system_id": system_id,
                "agroforestry_species": species_df.to_dict(orient="records")
            })
            features["land_use_class"] = LAND_COVER_LOOKUP.get(features["land_use_class"], "Unknown")
            results.append(features)
        except Exception as e:
            print(f"⚠️ Failed to process ({lat}, {lon}) with system_id {system_id}: {e}")

    return pd.DataFrame(results)


def simulate_agroforestry_plot(
    row,
    plot_size_m2=None,
    species_df=None,
    plot_size_distribution=[(300, 800), (800, 1200), (1200, 1800), (1800, 2500)],
    plot_size_weights=[0.3, 0.4, 0.2, 0.1]
):
    """
    Simulates an agroforestry plot composition keeping all species entries as-is,
    even if species names repeat.

    Parameters:
        row: A GeoDataFrame row with geometry and satellite-derived columns: forest_cover, canopy_height, land_use_class.
        plot_size_m2: Optional fixed plot size. If None, a random one is selected.
        species_df: DataFrame with one row per species entry (even if repeated).
    Returns:
        GeoDataFrame with one row per species entry in the simulated plot.
    """

    from shapely import wkt
    import random
    import numpy as np
    import geopandas as gpd
    import pandas as pd

    geometry = wkt.loads(str(row.geometry))
    forest_cover = row.get("forest_cover", 0)
    canopy_height = row.get("canopy_height", None)
    land_use_class = row.get("land_use_class", "Unknown")

    records = []

    # Choose plot size
    if plot_size_m2 is None:
        r = random.choices(plot_size_distribution, weights=plot_size_weights)[0]
        plot_size_m2 = random.randint(r[0], r[1])

    # Loop through all species entries
    for idx, row_species in species_df.iterrows():
        species = row_species["species"]
        height_min = row_species.get("height_min_m", 5)
        height_max = row_species.get("height_max_m", 15)
        height = random.uniform(height_min, height_max)
        type_ = row_species.get("type", "")
        cover_ratio = row_species.get("cover_ratio", np.nan)

        # Determine number of trees
        if pd.notnull(row_species.get("quantity", None)):
            try:
                n = int(row_species["quantity"])
                if n <= 0:
                    continue
            except:
                continue
        elif pd.notnull(row_species.get("probability", None)):
            if np.random.rand() < row_species["probability"]:
                n = np.random.randint(1, 6)
            else:
                continue
        else:
            continue  # Skip if neither valid quantity nor probability

        # Fallback for cover ratio
        if pd.isnull(cover_ratio):
            cover_ratio = 1.5 * height

        # Assign role
        if height >= 15:
            role = "upper canopy"
        elif height >= 5:
            role = "mid canopy"
        else:
            role = "understory"

        records.append({
            "species": species,
            "n_trees": n,
            "height_m": height,
            "height_min_m": height_min,
            "height_max_m": height_max,
            "type": type_,
            "cover_ratio": cover_ratio,
            "role": role,
            "plot_size_m2": plot_size_m2,
            "forest_cover": forest_cover,
            "canopy_height": canopy_height,
            "land_use_class": land_use_class,
            "system_type": "Agroforestry",
            "source": "table",
            "geometry": geometry
        })

    if not records:
        return None

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Shading calculations
    gdf["adjusted_cover_ratio"] = gdf["cover_ratio"] * gdf["n_trees"]
    gdf["adjusted_cover_area_m2"] = (gdf["adjusted_cover_ratio"] / 100) * gdf["plot_size_m2"]
    gdf["species_based_cover_area_m2"] = gdf["adjusted_cover_area_m2"]

    total_simulated_cover_area = gdf["adjusted_cover_area_m2"].sum()
    satellite_forest_area = (forest_cover / 100) * plot_size_m2
    estimated_forest_cover_area = min(total_simulated_cover_area, satellite_forest_area)

    gdf["estimated_forest_cover_area_m2"] = estimated_forest_cover_area
    gdf["expected_forest_area_m2"] = satellite_forest_area

    if total_simulated_cover_area > 0:
        gdf["shading_m2"] = (gdf["adjusted_cover_area_m2"] / total_simulated_cover_area) * estimated_forest_cover_area
    else:
        gdf["shading_m2"] = 0

    return gdf


def generate_spei_dict_from_dra(df):
    """
    Generate a dictionary of drought tolerance thresholds (SPEI) for each species 
    based on qualitative drought resistance descriptions.

    This function interprets values in the 'DRA' column (drought resistance assessment)
    and maps them to quantitative SPEI (Standardised Precipitation-Evapotranspiration Index)
    ranges indicating:
    - "ideal": range where the species performs best
    - "tolerable": wider range where the species can survive but may be stressed

    DRA labels are matched using keyword-based mapping to predefined SPEI ranges.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input dataframe with at least the following columns:
        - 'species': Name of the tree species
        - 'DRA': Drought resistance description (e.g., "moderate", "very poor")

    Returns:
    -------
    spei_dict : dict
        Dictionary where keys are species names and values are SPEI tolerance ranges:
        {
            "Species A": {
                "ideal": (min_spei, max_spei),
                "tolerable": (min_spei, max_spei)
            },
            ...
        }

    Notes:
    -----
    - Species with no recognizable DRA description are omitted.
    - If multiple labels are present for a species, the most drought-tolerant match is used.
    """

    # Define drought tolerance categories and their SPEI thresholds
    dra_map = {
        "very poor": {"ideal": (-0.2, 0), "tolerable": (-0.5, 0)},
        "poor": {"ideal": (-0.3, 0), "tolerable": (-0.7, 0)},
        "moderate": {"ideal": (-0.5, 0), "tolerable": (-1.0, 0)},
        "well (dry spells)": {"ideal": (-0.7, 0), "tolerable": (-1.5, 0)},
        "excessive": {"ideal": (-1.0, 0), "tolerable": (-2.0, 0)},
    }

    # Create a ranked list for determining the "most tolerant"
    tolerance_rank = list(dra_map.keys())[::-1]  # Most tolerant first

    spei_dict = {}
    for species, group in df.groupby("species"):
        dra_values = group["DRA"].dropna().astype(str).str.lower().unique()
        matched_keys = set()
        for val in dra_values:
            for key in dra_map:
                if key in val:
                    matched_keys.add(key)

        # Select the most tolerant matching description
        for key in tolerance_rank:
            if key in matched_keys:
                spei_dict[species] = dra_map[key]
                break

    return spei_dict


def plot_agroforest_vulnerability(df, spei_dict):
    """
    Visualises the climate vulnerability of agroforestry shade tree species 
    by comparing their ideal and tolerable ranges for temperature, precipitation, 
    and drought conditions.

    The plot contains three horizontal bar charts:
    1. Temperature (°C): Ideal and tolerable temperature range per species.
    2. Precipitation (mm/month): Ideal and tolerable precipitation range per species.
    3. Drought Tolerance (SPEI): Ideal and tolerable drought tolerance expressed as 
       a range of Standardised Precipitation-Evapotranspiration Index (SPEI) values.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the following columns:
        - 'species': species name
        - 'type': must include "1st tree" or "2nd tree" for shade trees
        - 'TOPMN', 'TOPMX': ideal temperature min and max
        - 'TMIN', 'TMAX': tolerable temperature min and max
        - 'ROPMN', 'ROPMX': ideal precipitation min and max
        - 'RMIN', 'RMAX': tolerable precipitation min and max

    spei_dict : dict
        Dictionary mapping species names to their drought tolerance ranges. 
        Format:
        {
            "Species A": {
                "ideal": (min_spei, max_spei),
                "tolerable": (min_spei, max_spei)
            },
            ...
        }

    Returns:
    -------
    None
        Displays a matplotlib figure with three subplots showing climate vulnerability.
    """

    shade_df = df[df["type"].isin(["1st tree", "2nd tree"])].copy()
    species_order = list(shade_df["species"].value_counts().index)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    # === 1. Temperature ===
    temp_df = (
        shade_df.groupby("species")[["TOPMN", "TOPMX", "TMIN", "TMAX"]]
        .mean(numeric_only=True)
        .dropna()
        .reindex(species_order)
    )
    for i, row in temp_df.iterrows():
        axs[0].barh(i, row["TOPMX"] - row["TOPMN"], left=row["TOPMN"], color="#ffcc99", edgecolor="black", label="Ideal" if i == temp_df.index[0] else "")
        axs[0].barh(i, row["TMAX"] - row["TMIN"], left=row["TMIN"], color="#ffe6cc", edgecolor="black", alpha=0.5, label="Tolerable" if i == temp_df.index[0] else "")
    axs[0].set_title("Temperature (°C)")
    axs[0].set_xlabel("°C")

    # === 2. Precipitation ===
    prec_df = (
        shade_df.groupby("species")[["ROPMN", "ROPMX", "RMIN", "RMAX"]]
        .mean(numeric_only=True)
        .dropna()
        .reindex(species_order)
    )
    for i, row in prec_df.iterrows():
        axs[1].barh(i, row["ROPMX"] - row["ROPMN"], left=row["ROPMN"], color="#99ccff", edgecolor="black", label="Ideal" if i == prec_df.index[0] else "")
        axs[1].barh(i, row["RMAX"] - row["RMIN"], left=row["RMIN"], color="#cce6ff", edgecolor="black", alpha=0.5, label="Tolerable" if i == prec_df.index[0] else "")
    axs[1].set_title("Precipitation (mm/month)")
    axs[1].set_xlabel("mm")

    # === 3. Drought Tolerance (SPEI) ===
    for i, species in enumerate(species_order):
        if species in spei_dict:
            ideal = spei_dict[species]["ideal"]
            tolerable = spei_dict[species]["tolerable"]

            # Only plot negative parts
            ideal_start = max(ideal[0], -2)
            ideal_end = min(ideal[1], 0)
            tolerable_start = max(tolerable[0], -2.5)
            tolerable_end = min(tolerable[1], 0)

            axs[2].barh(species, ideal_end - ideal_start, left=ideal_start, color="#a1dab4", edgecolor="black", label="Ideal" if i == 0 else "")
            axs[2].barh(species, tolerable_end - tolerable_start, left=tolerable_start, color="#e5f5f9", edgecolor="black", alpha=0.5, label="Tolerable" if i == 0 else "")
    axs[2].set_title("Drought Tolerance (SPEI)")
    axs[2].set_xlabel("SPEI (drought ≤ 0)")
    axs[2].set_xlim(-2.5, 0)

    # === Styling ===
    for ax in axs:
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()



def get_climate_at_point(tmean, ppt, lat, lon):
    t = tmean.sel(lat=lat, lon=lon, method="nearest").mean("year").values.item()
    p = ppt.sel(lat=lat, lon=lon, method="nearest").mean("year").values.item()
    return t, p

# === Per-hazard definitions ===
def make_trapezoidal_impact_func(haz_type, name, min_val, opt_min, opt_max, max_val, id_):
    impf = ImpactFunc()
    impf.haz_type = haz_type
    impf.id = id_
    impf.name = name

    if haz_type == "TM":
        intensity = np.array([opt_max, max_val, max_val + 5, max_val + 10])
        mdd = np.array([0.0, 0.5, 0.8, 1.0])
    elif haz_type == "PR":
        intensity = np.array([min_val - 500, min_val - 200, opt_min, opt_min + 1])
        mdd = np.array([1.0, 0.8, 0.3, 0.0])

    impf.intensity = intensity
    impf.mdd = mdd
    impf.paa = np.ones_like(mdd)
    impf.mdr = impf.mdd * impf.paa

    impf.check()
    return impf

# === Temperature and precipitation ===
def define_temp_prec_impfs(df, tmean, ppt, T_BUFFER=5, P_BUFFER=800):
    impf_set = ImpactFuncSet()
    species_to_temp_id, species_to_prec_id = {}, {}
    fallback_ids = {}
    counter = 1

    for _, row in df.iterrows():
        species = row["species"]
        lat, lon = row.geometry.y, row.geometry.x

        tmin = row.get("TMIN")
        tmax = row.get("TMAX")
        t_opt_min = row.get("TOPMN", tmin)
        t_opt_max = row.get("TOPMX", tmax)

        rmin = row.get("RMIN")
        rmax = row.get("RMAX")
        r_opt_min = row.get("ROPMN", rmin)
        r_opt_max = row.get("ROPMX", rmax)

        if species in species_to_temp_id and species in species_to_prec_id:
            continue

        if any(pd.isnull([tmin, tmax, rmin, rmax])):
            loc_key = (lat, lon)
            if loc_key in fallback_ids:
                temp_id, prec_id = fallback_ids[loc_key]
            else:
                t_loc, p_loc = get_climate_at_point(tmean, ppt, lat, lon)

                impf_temp = make_trapezoidal_impact_func("TM", f"fallback_{lat:.2f}_{lon:.2f}",
                                                         t_loc - T_BUFFER, t_loc - 1, t_loc + 1, t_loc + T_BUFFER, counter)
                impf_set.append(impf_temp)
                temp_id = counter
                counter += 1

                impf_prec = make_trapezoidal_impact_func("PR", f"fallback_{lat:.2f}_{lon:.2f}",
                                                         max(p_loc - P_BUFFER, 0), p_loc - 200, p_loc + 200, p_loc + P_BUFFER, counter)
                impf_set.append(impf_prec)
                prec_id = counter
                counter += 1

                fallback_ids[loc_key] = (temp_id, prec_id)

            species_to_temp_id[species] = temp_id
            species_to_prec_id[species] = prec_id

        else:
            impf_temp = make_trapezoidal_impact_func("TM", species, tmin, t_opt_min, t_opt_max, tmax, counter)
            impf_set.append(impf_temp)
            species_to_temp_id[species] = counter
            counter += 1

            impf_prec = make_trapezoidal_impact_func("PR", species, rmin, r_opt_min, r_opt_max, rmax, counter)
            impf_set.append(impf_prec)
            species_to_prec_id[species] = counter
            counter += 1

    return impf_set, species_to_temp_id, species_to_prec_id

# === Tropical cyclone impact functions ===
def define_tc_impfs():
    impf_set = ImpactFuncSet()
    intensity = np.linspace(0, 100, 15)

    def make_tc_curve(i, label, power):
        return ImpactFunc(
            haz_type="TC",
            id=i,
            name=label,
            intensity=intensity,
            mdd=np.linspace(0, 1, 15)**power,
            paa=np.ones(15),
        )

    impf_set.append(make_tc_curve(1, "High vulnerability (dense, tall canopy)", 1.5))
    impf_set.append(make_tc_curve(2, "Medium vulnerability (moderate canopy)", 2))
    impf_set.append(make_tc_curve(3, "Low vulnerability (sparse or short canopy)", 3))
    impf_set.append(ImpactFunc(haz_type="TC", id=0, name="Unknown / default",
                               intensity=intensity, mdd=np.zeros(15), paa=np.zeros(15)))
    return impf_set




def define_drought_impfs(spei_dict):
    impf_set = ImpactFuncSet()
    species_to_drought_id = {}

    # Default drought impact function with no impact before intensity 1.0
    default_intensity = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    default_mdd =      np.array([0.0, 0.0, 0.0, 0.2, 0.4, 0.6])

    default_impf = ImpactFunc(
        haz_type="DR",
        id=0,
        name="Default drought response",
        intensity=default_intensity,
        mdd=default_mdd,
        paa=np.ones_like(default_mdd)
    )
    default_impf.mdr = default_impf.mdd * default_impf.paa
    default_impf.check()
    impf_set.append(default_impf)

    counter = 1

    for species, spei_range in spei_dict.items():
        ideal_min, ideal_max = spei_range["ideal"]
        tol_min, tol_max = spei_range["tolerable"]

        # SPEI values inverted to positive intensity values
        spei_vals = [0.0, 0.5, 1.0, abs(ideal_min), abs(tol_min), abs(tol_min + 0.5)]
        intensity = np.array(sorted(set([round(abs(v), 2) for v in spei_vals if v <= 2.5])))

        # Match MDD: zero impact until intensity 1.0, then increase
        mdd = np.piecewise(intensity,
                           [intensity <= 1.0,
                            (intensity > 1.0) & (intensity <= 1.5),
                            (intensity > 1.5) & (intensity <= 2.0),
                            intensity > 2.0],
                           [0.0, 0.2, 0.4, 0.6])

        impf = ImpactFunc()
        impf.haz_type = "DR"
        impf.id = counter
        impf.name = f"Drought tolerance - {species}"
        impf.intensity = intensity
        impf.mdd = mdd
        impf.paa = np.ones_like(mdd)
        impf.mdr = impf.mdd * impf.paa
        impf.check()

        impf_set.append(impf)
        species_to_drought_id[species] = counter
        counter += 1

    return impf_set, species_to_drought_id




def assign_impact_function_ids(df, species_to_temp_id, species_to_prec_id, species_to_drought_id=None):
    df["impf_TM"] = df["species"].map(species_to_temp_id)
    df["impf_PR"] = df["species"].map(species_to_prec_id)

    if species_to_drought_id:
        df["impf_DR"] = df["species"].map(species_to_drought_id).fillna(0).astype(int)

    def assign_tc(row):
        hmin, hmax = row.get("height_min_m"), row.get("height_max_m")
        if pd.isnull(hmin) or pd.isnull(hmax): return 0
        h_mean = (hmin + hmax) / 2
        if h_mean >= 15: return 1
        elif h_mean >= 5: return 2
        else: return 3

    df["impf_TC"] = df.apply(assign_tc, axis=1)
    return df



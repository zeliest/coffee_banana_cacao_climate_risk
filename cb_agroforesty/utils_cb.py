from climada.entity import Exposures
from climada.util import rasterio
from climada.hazard import Hazard, Centroids
from climada.entity import ImpactFunc

import time
from scipy import sparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import copy

import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


import xarray as xr
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats

## DEFAULT PARAMETERS

# Define bounding box for Mexico and Guatemala
COUNTRY_BOUNDS = {'MEX': {# Define bounding box for Mexico and Guatemala
    "lon_min": -120,
    "lon_max": -85,
    "lat_min": 13,
    "lat_max": 32
    },
}

# Define file paths for crop production data
FILES_CROP_PROD = { "coffee": "Spam/spam2020V1r0_global_production/spam2020_v1r0_global_P_COFF_A.tif"}
FILES_CROP_HARVEST = { "coffee": "Spam/spam2020V1r0_global_harvested_area/spam2020_v1r0_global_H_COFF_A.tif"}

# Make a function that returns the exposure object given the bbox
def get_crop_prod_exposure(file, bbox, harvest_area_file=None):
    """
    Returns an Exposures object cropped to a bounding box, with optional harvested area added.
    
    Parameters:
    - file: Path to the main exposure raster file (e.g., yield).
    - bbox: Bounding box dictionary with keys lat_min, lat_max, lon_min, lon_max.
    - hazard_type: Hazard type string (e.g., 'VPD').
    - harvest_area_file: Optional path to harvested area raster to join by coordinates.
    - harvest_area_col: Name of the column for the added harvest area (default: 'harvest_area').

    Returns:
    - Exposures object.
    """
    # Load exposure
    exp = Exposures.from_raster(file)

    # Add coordinate column with rounding for matching
    exp.gdf["coord"] = list(zip(
        exp.gdf.geometry.x.round(5),
        exp.gdf.geometry.y.round(5)
    ))

    # Filter by bounding box
    gdf = exp.gdf[
        (exp.gdf.geometry.x >= bbox["lon_min"]) &
        (exp.gdf.geometry.x <= bbox["lon_max"]) &
        (exp.gdf.geometry.y >= bbox["lat_min"]) &
        (exp.gdf.geometry.y <= bbox["lat_max"])
    ].copy()

    # Filter out 0 values
    gdf = gdf[gdf["value"] > 0]

    # Optional: Add harvested area
    if harvest_area_file:
        area_exp = Exposures.from_raster(harvest_area_file)
        area_exp.gdf["coord"] = list(zip(
            area_exp.gdf.geometry.x.round(5),
            area_exp.gdf.geometry.y.round(5)
        ))

        area_df = area_exp.gdf[["coord", "value"]].rename(columns={"value": "harvest_area (ha)"})
        gdf = gdf.merge(area_df, on="coord", how="left")

    # Rebuild exposure object
    exp_out = Exposures(data=gdf.drop(columns=["coord"]))
    exp_out.value_unit = "tonnes"
    return exp_out

# # Make a function that returns the exposure object given the hazard type and bbox
# def get_crop_prod_exposure(file, bbox, hazard_type):
#     """
#     Returns the exposure object for the given file and bounding box.
#     """
#     exp = Exposures.from_raster(file)
#     # Set the impact function id
#     exp.gdf[f'impf_{hazard_type}'] = 1
#     # Select only exposures inside the bounding box
#     exp_central_america = exp.gdf[
#         (exp.gdf.geometry.x >= bbox["lon_min"]) &
#         (exp.gdf.geometry.x <= bbox["lon_max"]) &
#         (exp.gdf.geometry.y >= bbox["lat_min"]) &
#         (exp.gdf.geometry.y <= bbox["lat_max"])
#     ].copy()
#     # Remove all the rows with zero values
#     exp_central_america = exp_central_america[exp_central_america['value'] > 0]
#     # Make an Exposures object
#     exp_central_america = Exposures(data=exp_central_america)
#     # Set the value unit
#     exp_central_america.value_unit = 'tonnes'

#     return exp_central_america


# Make a function that merges all rasters in a given folder
def merge_rasters(input_dir: Path, output_path: Path, file_pattern: str = "*.tif") -> Path:
    """
    Merge all raster files in a given folder and save the result.

    Parameters:
    - input_dir (Path): Directory containing input .tif files.
    - output_path (Path): Path to save the merged raster.
    - file_pattern (str): Glob pattern to find raster files (default: '*.tif').

    Returns:
    - Path: The path to the merged output file.
    """
    tif_files = list(input_dir.glob(file_pattern))
    if not tif_files:
        raise FileNotFoundError(f"No files found in {input_dir} matching pattern {file_pattern}")

    print(f"Merging {len(tif_files)} raster tiles...")

    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]
    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": 1
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged raster saved at: {output_path}")
    return output_path



# That downsamples a raster given a factor
def downsample_raster(raster_path: Path, factor: int | None = 10) -> np.ndarray:
    """
    Load a raster and optionally downsample it.

    Parameters:
    - raster_path (Path): Path to the raster (.tif) file.
    - factor (int or None): Downsampling factor. Downsampling factor (e.g., 10 = 1/10th resolution). If None, loads at full resolution.

    Returns:
    - np.ndarray: Raster array (2D), optionally downsampled.
    """
    with rasterio.open(raster_path) as src:
        if factor is None:
            downsampled = src.read(1)
        else:
            downsampled = src.read(
                1,
                out_shape=(
                    int(src.height / factor),
                    int(src.width / factor)
                )
            )
        return downsampled

def add_canopy_cover_to_exposure(exp: Exposures, canopy_raster_path: Path, buffer_m: float = 1000) -> Exposures:
    """
    Add mean canopy cover from raster to an Exposures object using zonal stats.

    Parameters:
    - exp: Exposures object (CLIMADA)
    - canopy_raster_path: Path to merged canopy raster (.tif)
    - buffer_m: Radius of buffer (in meters) to compute zonal stats (default: 1000)

    Returns:
    - Updated Exposures object with 'canopy_cover_mean_1km' column
    """
    # Step 1: Reproject to metric CRS and buffer
    gdf = exp.gdf.to_crs(epsg=3857)
    gdf["buffer_geom"] = gdf.geometry.buffer(buffer_m)

    # Step 2: Reproject buffer to WGS84
    buffered_gdf = gdf.set_geometry("buffer_geom").to_crs("EPSG:4326")

    # Step 3: Zonal stats on tree cover
    stats = zonal_stats(
        buffered_gdf,
        canopy_raster_path,
        stats=["mean"],
        nodata=0,
        all_touched=True
    )

    # Step 4: Assign and clean
    gdf["canopy_cover"] = [s["mean"] for s in stats]
    gdf = gdf[~gdf["canopy_cover"].isna()]  # Drop NaNs

    # Step 5: Restore original geometry and CRS
    gdf = gdf.set_geometry("geometry").to_crs("EPSG:4326")

    # Step 6: Return new exposure
    new_exp = Exposures(data=gdf)
    new_exp.value_unit = exp.value_unit
    return new_exp

def load_isimip_multi_and_subset(
    tas_files: List[Path],
    hurs_files: List[Path],
    bbox: dict,
    padding: float = 1.0
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Load, merge, subset, and filter ISIMIP tas and hurs files over a bounding box and growing season.

    Parameters:
    - tas_files: List of Path objects to tas NetCDF files.
    - hurs_files: List of Path objects to hurs NetCDF files.
    - bbox: dict with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'.
    - padding: float, degrees of spatial buffer (default 1.0).

    Returns:
    - Tuple of DataArrays: (tas_growing_season_C, hurs_growing_season)
    """
    # Merge NetCDF files
    tas_all = xr.open_mfdataset(tas_files, combine='by_coords')
    hurs_all = xr.open_mfdataset(hurs_files, combine='by_coords')

    # Define spatial subset
    lat_slice = slice(bbox["lat_max"] + padding, bbox["lat_min"] - padding)
    lon_slice = slice(bbox["lon_min"] - padding, bbox["lon_max"] + padding)

    # Subset and convert tas to Celsius
    tas_celsius = tas_all['tas'].sel(lat=lat_slice, lon=lon_slice) - 273.15
    hurs = hurs_all['hurs'].sel(lat=lat_slice, lon=lon_slice)

    # Select growing season (May–Nov)
    growing = tas_celsius['time.month'].isin([5, 6, 7, 8, 9, 10, 11])
    tas_gs = tas_celsius.sel(time=growing)
    hurs_gs = hurs.sel(time=growing)

    return tas_gs, hurs_gs

def canopy_cooling(canopy_cover_percent, max_cooling=2.5, plateau_at=50):
    """
    General cooling function: linear increase until a plateau.
    
    Parameters:
    - canopy_cover_percent : float or np.array
        Canopy cover in percent (0 to 100).
    - max_cooling : float
        Maximum cooling in °C (positive value, e.g., 2.5°C).
    - plateau_at : float
        Canopy cover (%) at which max_cooling is reached.

    Returns:
    - cooling : np.array
        Cooling values in °C (negative).
    """
    canopy_cover_percent = np.clip(canopy_cover_percent, 0, 100)
    slope = -max_cooling / plateau_at
    cooling = np.where(
        canopy_cover_percent <= plateau_at,
        slope * canopy_cover_percent,
        -max_cooling
    )
    return cooling

def compute_vpd_hazard(exp, tas_gs, hurs_gs, canopy_scenario="current", cooling_fn=None, return_timings=False):
    """
    Compute VPD hazard with canopy cooling effect, with timing for each step.
    Parameters:
    - exp: Exposures object with 'canopy_cover' column.
    - tas_gs: xarray DataArray of temperature (tas) for the growing season.
    - hurs_gs: xarray DataArray of relative humidity (hurs) for the growing season.
    - canopy_scenario: Canopy cover scenario (e.g., "current", None, or a numeric value).
    - cooling_fn: Function to compute cooling effect based on canopy cover.
    - return_timings: If True, return a dictionary with timings for each step.

    Returns:
    - hazard: Hazard object with VPD intensity.
    - timings: Dictionary with timings for each step (if return_timings is True).
    """


    timings = {}

    # Step 1: Get canopy cover
    t0 = time.time()
    if canopy_scenario in [None, "none"]:
        canopy_cover = np.zeros(len(exp.gdf))
    else:
        current_cover = exp.gdf["canopy_cover"].values
        if canopy_scenario == "current":
            canopy_cover = current_cover
        elif isinstance(canopy_scenario, (int, float)):
            # Apply only where current canopy is below the scenario threshold
            canopy_cover = np.where(current_cover < canopy_scenario, canopy_scenario, current_cover)
        else:
            raise ValueError("Invalid canopy_scenario.")
    timings["canopy_cover"] = time.time() - t0

    # Step 2: Interpolate
    t0 = time.time()
    tas_pts = tas_gs.interp(lon=("site", exp.gdf.geometry.x.values),
                            lat=("site", exp.gdf.geometry.y.values),
                            method="nearest")
    hurs_pts = hurs_gs.interp(lon=("site", exp.gdf.geometry.x.values),
                              lat=("site", exp.gdf.geometry.y.values),
                              method="nearest")
    timings["interpolation"] = time.time() - t0

    # Step 3: Cooling
    t0 = time.time()
    cooling = cooling_fn(canopy_cover)
    tas_shaded = tas_pts + cooling[np.newaxis, :]
    timings["cooling"] = time.time() - t0

    # Step 4: Compute VPD
    t0 = time.time()
    es = 0.6108 * np.exp((17.27 * tas_shaded) / (tas_shaded + 237.3))
    ea = (hurs_pts / 100) * es
    vpd = es - ea
    timings["vpd_computation"] = time.time() - t0

    # Step 5: Aggregate (faster version without .to_pandas())
    t0 = time.time()
    vpd_yr = vpd.groupby("time.year").mean("time")  # DataArray: [year x site]
    years = vpd_yr.year.values
    timings["aggregation"] = time.time() - t0

    # Step 6: Package
    t0 = time.time()
    # Optional — Zero out small values (helps sparsity)
    vpd_array = vpd_yr.values
    vpd_array[np.abs(vpd_array) < 0.4] = 0 # Convert small values to zero to speed up packaging
    # Use faster constructor
    intensity = sparse.csr_matrix(vpd_array)
    if np.isnan(intensity.data).any():
        raise ValueError("NaNs in VPD intensity matrix.")
    timings["packaging"] = time.time() - t0
    
    # Step 7: Create Hazard object
    t0 = time.time()
    hazard = Hazard(
        haz_type="VPD",
        intensity=intensity,
        fraction=intensity.copy().astype(bool),
        centroids=Centroids(lat=exp.gdf.geometry.y.values, lon=exp.gdf.geometry.x.values),
        units="kPa",
        event_id=np.arange(len(years)),
        frequency=np.ones(len(years)) / len(years),
        date=np.array([pd.Timestamp(f"{y}-01-01").toordinal() for y in years]),
        event_name=[f"year_{y}" for y in years]
    )
    timings["hazard"] = time.time() - t0

    # Step 7: Return hazard and timings
    if return_timings:
        return hazard, timings

    return hazard

def impact_func_from_kath(
    file_path: str,
    intensity_col: str,
    yield_col: str,
    only_yield_losses: bool = True,
    haz_type: str = "VPD",
    unit: str = "kPa",
    if_id: int = 1,
    name: str = "Impact Function from CSV"
) -> ImpactFunc:
    """
    Create a CLIMADA ImpactFunc directly from a CSV file with intensity and yield(-log) data.

    Parameters:
    - file_path: path to CSV
    - intensity_col: column name for hazard intensity (e.g. 'Mean_VPD_kPa')
    - yield_col: column name for yield or log-yield (e.g. 'Effect_Central')
    - is_log: True if yield_col is in log scale
    - only_yield_losses: whether to filter for only negative yield impacts
    - haz_type: CLIMADA hazard type string (e.g. "VPD")
    - unit: unit for intensity (e.g. "kPa", "°C")
    - if_id: Impact function ID
    - name: Name for the impact function

    Returns:
    - ImpactFunc
    """
    df = pd.read_csv(file_path)

    if yield_col not in df.columns or intensity_col not in df.columns:
        raise ValueError("Specified columns not found in the CSV.")

    x = df[intensity_col].values
    y = df[yield_col].values

    # Optional filtering
    if only_yield_losses:
        mask = y < 0
        x, y = x[mask], y[mask]

    # Convert to linear scale if needed
    yield_vals = np.exp(y)

    # Normalize to max yield (best-case baseline)
    baseline = np.max(yield_vals)
    mdd = 1 - (yield_vals / baseline)
    mdd = np.clip(mdd, 0, 1)

    paa = np.ones_like(mdd)

    # Build and return
    impf = ImpactFunc(
        id=if_id,
        name=name,
        haz_type=haz_type,
        intensity=x,
        mdd=mdd,
        paa=paa,
        intensity_unit=unit
    )
    impf.check()
    return impf

def compute_additional_canopy_cost(
    exp,
    min_canopy_percent=50,
    setup_cost_per_ha=300,
    annual_maint_cost_per_ha=30,
    duration_years=10,
    default_hektar_per_site=5
):
    """
    Compute cost and canopy area needed to reach a minimum canopy threshold.

    Parameters:
    - exp: Exposures object with 'canopy_cover' column.
    - min_canopy_percent: Target minimum canopy cover (%)
    - setup_cost_per_ha: Setup cost per hectare (USD)
    - annual_maint_cost_per_ha: Annual maintenance cost per hectare (USD)
    - duration_years: Maintenance period (years)
    - default_hektar_per_site: Default area per site if missing

    Returns:
    - total_cost: float, total cost across all sites
    - site_costs: pd.Series of site-specific costs
    - total_planted_hectares: float, total new canopy hectares needed
    """
    import pandas as pd

    current_cover = exp.gdf["canopy_cover"]
    current_frac = current_cover / 100
    target_frac = min_canopy_percent / 100

    # Only plant if under target
    deficit_frac = (target_frac - current_frac).clip(lower=0)

    if "harvest_area (ha)" in exp.gdf.columns:
        area_ha = exp.gdf["harvest_area (ha)"]
    else:
        area_ha = pd.Series(default_hektar_per_site, index=exp.gdf.index)

    # Hectares of canopy to be planted per site
    planted_ha = area_ha * deficit_frac

    # Cost calculation
    site_costs = planted_ha * (setup_cost_per_ha + annual_maint_cost_per_ha * duration_years)

    total_cost = site_costs.sum()
    total_planted_hectares = planted_ha.sum()

    return total_cost, site_costs, total_planted_hectares


def generate_random_impact_func_from_ci(
    file_path: str,
    intensity_col: str = "Mean_VPD_kPa",
    lower_col: str = "Lower_Bound",
    central_col: str = "Effect_Central",
    upper_col: str = "Upper_Bound",
    haz_type: str = "VPD",
    unit: str = "kPa",
    if_id: int = 1,
    name: str = "Sampled IF from CI",
    seed: int = None
) -> ImpactFunc:
    """
    Generate a random ImpactFunc using triangular sampling from confidence bounds.

    Parameters:
    - file_path: Path to CSV with intensity, lower/central/upper effect
    - *_col: Column names for intensity and bounds
    - haz_type: CLIMADA hazard type
    - unit: Unit of intensity (e.g. 'kPa')
    - if_id: Impact function ID
    - name: Name of the function
    - seed: Optional random seed

    Returns:
    - ImpactFunc
    """
    if seed is not None:
        np.random.seed(seed)

    df = pd.read_csv(file_path)
    df = df[df[central_col] < 0]  # focus on yield losses only

    x = df[intensity_col].values
    lower = df[lower_col].values
    central = df[central_col].values
    upper = df[upper_col].values

    # Triangular sampling: lower, mode, upper
    sampled_effect = np.random.triangular(lower, central, upper)

    # Convert to yield ratio and normalize
    yield_ratio = np.exp(sampled_effect)
    baseline_yield = yield_ratio.max()
    mdd = np.clip(1 - (yield_ratio / baseline_yield), 0, 1)
    paa = np.ones_like(mdd)

    impf = ImpactFunc(
        id=if_id,
        name=name if seed is None else f"{name} (seed={seed})",
        haz_type=haz_type,
        intensity=x,
        mdd=mdd,
        paa=paa,
        intensity_unit=unit
    )
    impf.check()
    return impf

def plot_uncertainty_curves(output_dict, ax, *, y_label="Impact", title="", value_scale=1.0, percent=False):
    def get_ci_bounds(unc_output):
        df = unc_output.freq_curve_unc_df * value_scale
        return {
            'p05': df.quantile(0.05),
            'p50': df.quantile(0.5),
            'p95': df.quantile(0.95)
        }

    for label, unc_output in output_dict.items():
        rps = [int(col[2:]) for col in unc_output.freq_curve_unc_df.columns]
        ci = get_ci_bounds(unc_output)
        ax.plot(rps, ci['p50'], label=f'{label} Median')
        ax.fill_between(rps, ci['p05'], ci['p95'], alpha=0.2, label=f'{label} [90% CI]')

    ax.set_xscale("log")
    ax.set_xlim(1, max(rps))
    ax.set_xticks([1, 2, 5, 10, 20, 30])
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}"))

    ax.set_xlabel('Return Period [years]')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    elif "USD" in y_label or "Value" in title:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1e6:.0f}M" if abs(x) >= 1e6 else f"${x:,.0f}"))
    elif "tonnes" in y_label:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if abs(x) >= 1e3 else f"{x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))


def plot_aai_kde_colored_means(output_dict, ax, *, to_percent=False, y_label="Density", title=""):
    """
    Plot KDEs of AAI aggregated uncertainty values with matching colored vertical mean lines into a given axis.
    """
    palette = sns.color_palette(n_colors=len(output_dict))
    
    for (label, unc_output), color in zip(output_dict.items(), palette):
        aai_values = unc_output.aai_agg_unc_df.iloc[:, 0].dropna()
        if to_percent:
            aai_values *= 100
        mean_val = aai_values.mean()

        sns.kdeplot(
            aai_values,
            label=f"{label} (mean: {mean_val:,.1f}{'%' if to_percent else ''})",
            fill=True,
            alpha=0.3,
            linewidth=2,
            ax=ax,
            color=color
        )

        ax.axvline(
            mean_val,
            color=color,
            linestyle='--',
            linewidth=2,
            zorder=5
        )
    

    ax.set_xlabel('Aggregated AAI [{}]'.format('%' if to_percent else 'units'))
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if to_percent:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    elif 'USD' in y_label or 'Value' in title:
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
    else:
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    
    ax.legend()
    ax.grid(True)

    
# def plot_kde_and_rp_panels(output_imp, scaled_price_imp):

#     kde_configs = [
#         {"data": output_imp["unscaled"], "title": "Absolute AAI (Production Volume)"},
#         {"data": output_imp["scaled"], "title": "Relative AAI (Share of Total Production)", "to_percent": True},
#         {"data": scaled_price_imp, "title": "Economic AAI (Market Value in USD)"}
#     ]

#     rp_configs = [
#         {"data": output_imp["unscaled"], "title": "Absolute RP Curve", "y_label": "Impact [k tonnes]"},
#         {"data": output_imp["scaled"], "title": "Relative RP Curve", "y_label": "Impact [% of Total]", "value_scale": 100, "percent": True},
#         {"data": scaled_price_imp, "title": "Economic RP Curve", "y_label": "Impact [USD]"}
#     ]

#     fig_kde, axs_kde = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
#     for ax, cfg in zip(axs_kde, kde_configs):
#         plot_aai_kde_colored_means(cfg["data"], ax=ax, title=cfg["title"], to_percent=cfg.get("to_percent", False))
#     fig_kde.tight_layout()

#     fig_rp, axs_rp = plt.subplots(1, 3, figsize=(18, 5))
#     for ax, cfg in zip(axs_rp, rp_configs):
#         plot_uncertainty_curves(
#             cfg["data"],
#             ax=ax,
#             title=cfg["title"],
#             y_label=cfg["y_label"],
#             value_scale=cfg.get("value_scale", 1.0),
#             percent=cfg.get("percent", False)
#         )
#     fig_rp.tight_layout()
#     plt.show()

def scale_outputs_by_price(output_dict, price_per_ton):
    """
    Scale AAI and frequency curve values by a unit price.
    """
    scaled_output = {
        label: copy.deepcopy(output) for label, output in output_dict.items()
    }
    for out in scaled_output.values():
        out.aai_agg_unc_df.iloc[:, 0] *= price_per_ton
        out.freq_curve_unc_df *= price_per_ton
    return scaled_output

def plot_kde_and_rp_panels(output_imp, price_per_ton=None):

    # Compute price-scaled version if requested
    if price_per_ton:
        scaled_price_imp = scale_outputs_by_price(output_imp["unscaled"], price_per_ton)
    else:
        scaled_price_imp = None

    # --- KDE Plots ---
    kde_configs = [
        {"data": output_imp["unscaled"], "title": "Absolute AAI (Production Volume)"},
        {"data": output_imp["scaled"], "title": "Relative AAI (Share of Total Production)", "to_percent": True},
    ]
    if scaled_price_imp:
        kde_configs.append({"data": scaled_price_imp, "title": "Economic AAI (Market Value in USD)"})

    fig_kde, axs_kde = plt.subplots(1, len(kde_configs), figsize=(6 * len(kde_configs), 5), sharey=False)
    for ax, cfg in zip(axs_kde, kde_configs):
        plot_aai_kde_colored_means(cfg["data"], ax=ax, title=cfg["title"], to_percent=cfg.get("to_percent", False))
    fig_kde.tight_layout()

    # --- RP Plots ---
    rp_configs = [
        {"data": output_imp["unscaled"], "title": "Absolute RP Curve", "y_label": "Impact [k tonnes]"},
        {"data": output_imp["scaled"], "title": "Relative RP Curve", "y_label": "Impact [% of Total Production]", "value_scale": 100, "percent": True},
    ]
    if scaled_price_imp:
        rp_configs.append({"data": scaled_price_imp, "title": "Economic RP Curve", "y_label": "Impact [USD]"})

    fig_rp, axs_rp = plt.subplots(1, len(rp_configs), figsize=(6 * len(rp_configs), 5))
    for ax, cfg in zip(axs_rp, rp_configs):
        plot_uncertainty_curves(
            cfg["data"],
            ax=ax,
            title=cfg["title"],
            y_label=cfg["y_label"],
            value_scale=cfg.get("value_scale", 1.0),
            percent=cfg.get("percent", False)
        )
    fig_rp.tight_layout()
    plt.show()


    import xarray as xr
import numpy as np
import pandas as pd
from scipy import sparse
from climada.hazard import Hazard, Centroids

def create_hazard_from_nc_files(
    nc_files,
    varname,
    haz_type="HAZARD",
    units="unit",
    agg="max",
    months=None,             # List of integers, e.g., [4,5,6,7,8,9] for April–September
    nan_replacement=None     # Value to replace NaNs, e.g., 0.0 or np.nan to skip
):
    """
    Create a CLIMADA Hazard object from NetCDF files.

    Parameters:
    - nc_files (List[Path or str]): List of NetCDF file paths
    - varname (str): Variable name to extract (e.g. "tas", "vpd", "tmax")
    - haz_type (str): Hazard type label
    - units (str): Units of the variable
    - agg (str): Aggregation method over time ('max', 'sum', 'mean')
    - months (List[int], optional): Months (1–12) to include in annual aggregation
    - nan_replacement (float or None): Value to replace NaNs in the intensity matrix

    Returns:
    - climada.hazard.Hazard object
    """
    print(f"📂 Loading {len(nc_files)} files for variable '{varname}'...")
    ds_list = [xr.open_dataset(str(f))[varname] for f in nc_files]
    data = xr.concat(ds_list, dim="time")

    if months:
        print(f"📆 Filtering to months: {months}")
        data = data.sel(time=data['time.month'].isin(months))

    print(f"🧮 Aggregating by year using '{agg}'...")
    grouped = data.groupby("time.year")
    if agg == "max":
        data_yr = grouped.max(dim="time")
    elif agg == "sum":
        data_yr = grouped.sum(dim="time")
    elif agg == "mean":
        data_yr = grouped.mean(dim="time")
    else:
        raise ValueError(f"Unsupported aggregation method: {agg}")

    print("📌 Reshaping data to [year, lat*lon] array...")
    data_flat = data_yr.stack(site=("lat", "lon")).transpose("year", "site")
    data_arr = data_flat.values

    if nan_replacement is not None:
        print(f"🔄 Replacing NaNs with {nan_replacement}...")
        data_arr = np.nan_to_num(data_arr, nan=nan_replacement)

    print("🧱 Converting to sparse matrix...")
    intensity = sparse.csr_matrix(data_arr)

    print("🗺️ Extracting lat/lon centroids...")
    latlon = data_flat.site.to_index().to_frame(index=False)
    centroids = Centroids(lat=latlon["lat"].values, lon=latlon["lon"].values)

    years = data_yr.year.values
    event_name = [f"year_{y}" for y in years]

    print("📦 Creating Hazard object...")
    hazard = Hazard(
        haz_type=haz_type,
        intensity=intensity,
        fraction=intensity.copy().astype(bool),
        centroids=centroids,
        units=units,
        event_id=np.arange(len(years)),
        frequency=np.ones(len(years)) / len(years),
        date=np.array([pd.Timestamp(f"{y}-01-01").toordinal() for y in years]),
        event_name=event_name
    )

    return hazard


def reproject_hazard_to_exposures(hazard, exp):
    """
    Interpolate a gridded Hazard object to exposure site locations,
    and return a new Hazard object with site-level columns.

    Parameters:
    - hazard (Hazard): CLIMADA Hazard object with gridded data
    - exp (Exposures): CLIMADA Exposures object with site geometries
    - value_threshold (float): Threshold to zero out small values for sparsity

    Returns:
    - site_hazard (Hazard): New Hazard object [years x sites]
    """
    print("🔁 Interpolating hazard values to exposure coordinates...")

    # Get hazard info
    n_events, n_grid = hazard.intensity.shape
    years = np.array([pd.Timestamp.fromordinal(d).year for d in hazard.date])
    lat = hazard.centroids.lat
    lon = hazard.centroids.lon

    # Create DataArray from intensity matrix [year x grid]
    intensity_dense = hazard.intensity.toarray()  # [n_events x n_grid]
    df = pd.DataFrame({
        'lat': np.repeat(lat, n_events),
        'lon': np.repeat(lon, n_events),
        'year': np.tile(years, n_grid),
        'value': intensity_dense.T.flatten()
    })

    # Create 3D DataArray: [year, lat, lon]
    da = df.set_index(['year', 'lat', 'lon']).to_xarray().value

    # Interpolate to exposure points
    interp_vals = da.interp(
        lon=("site", exp.gdf.geometry.x.values),
        lat=("site", exp.gdf.geometry.y.values),
        method="nearest"
    )

    # Prepare intensity matrix: [n_years x n_sites]
    data_arr = interp_vals.values
    intensity = sparse.csr_matrix(data_arr)

    # Create site-level Hazard
    site_hazard = Hazard(
        haz_type=hazard.haz_type,
        intensity=intensity,
        fraction=intensity.copy().astype(bool),
        centroids=Centroids(lat=exp.gdf.geometry.y.values, lon=exp.gdf.geometry.x.values),
        units=hazard.units,
        event_id=np.arange(len(years)),
        frequency=np.ones(len(years)) / len(years),
        date=np.array([pd.Timestamp(f"{y}-01-01").toordinal() for y in years]),
        event_name=[f"year_{y}" for y in years]
    )

    return site_hazard


def apply_canopy_scenario(exp, canopy_scenario="current"):
    """
    Return a new exposure object with canopy adjusted to the specified scenario.
    """
    from copy import deepcopy
    new_exp = deepcopy(exp)
    current_cover = exp.gdf.get("canopy_cover", np.zeros(len(exp.gdf)))

    if canopy_scenario in [None, "none"]:
        new_exp.gdf["canopy_cover"] = 0
    elif canopy_scenario == "current":
        new_exp.gdf["canopy_cover"] = current_cover
    elif isinstance(canopy_scenario, (int, float)):
        new_exp.gdf["canopy_cover"] = np.where(current_cover < canopy_scenario, canopy_scenario, current_cover)
    else:
        raise ValueError("Invalid canopy_scenario.")

    return new_exp

def adjust_hazard_with_exposure_modifier(
    exposure,
    hazard,
    adjustment_fn,
    adjustment_kwargs=None,
    return_timings=False
):
    """
    Apply a cell-wise reduction to hazard intensity values based on exposure info.

    The `adjustment_fn` must return the REDUCTION (not the final value).

    Parameters:
    - exposure: Exposures object
    - hazard: Hazard object
    - adjustment_fn: Function returning an adjustment (subtracted from hazard values)
        Signature:
        def adjustment_fn(hazard_matrix, exposure_gdf, **kwargs) -> np.ndarray
        Output shape must be:
            - same as hazard_matrix (n_events x n_sites), or
            - (n_sites,) — broadcasted across years
    - adjustment_kwargs: Additional keyword arguments passed to `adjustment_fn`
    - return_timings: If True, return timing info

    Returns:
    - adjusted_hazard: Hazard object
    - timings (optional)
    """
    import time
    from copy import deepcopy
    from scipy import sparse

    if adjustment_kwargs is None:
        adjustment_kwargs = {}

    timings = {}
    t0 = time.time()

    exposure_gdf = exposure.gdf
    hazard_matrix = hazard.intensity.toarray()
    timings["prepare_data"] = time.time() - t0

    t0 = time.time()
    reduction = adjustment_fn(
        hazard_matrix=hazard_matrix,
        exposure_gdf=exposure_gdf,
        **adjustment_kwargs
    )

    if reduction.shape != hazard_matrix.shape and reduction.shape != (hazard_matrix.shape[1],):
        raise ValueError("adjustment_fn must return shape [n_events x n_sites] or [n_sites]")

    if reduction.ndim == 1:
        reduction = np.tile(reduction, (hazard_matrix.shape[0], 1))

    adjusted = hazard_matrix - reduction
    adjusted[adjusted < 0] = 0
    timings["adjustment_calc"] = time.time() - t0

    t0 = time.time()
    adjusted_hazard = deepcopy(hazard)
    adjusted_hazard.intensity = sparse.csr_matrix(adjusted)
    timings["packaging"] = time.time() - t0

    if return_timings:
        return adjusted_hazard, timings
    return adjusted_hazard

def helper_adj_fcn_canopy_vpd(
    hazard_matrix,
    exposure_gdf,
    max_cooling=3.0,
    plateau_at=80,
    vpd_sensitivity=0.15
):
    """
    Estimate VPD REDUCTION from canopy-based cooling.

    Parameters:
    - hazard_matrix: [n_events x n_sites] np.array (original VPD values)
    - exposure_gdf: GeoDataFrame with 'canopy_cover' column (0–100%)
    - max_cooling: °C, maximum temperature cooling (positive value)
    - plateau_at: % canopy cover where max cooling is reached
    - vpd_sensitivity: kPa per °C, e.g. 0.15 means 1°C → 0.15 kPa VPD drop

    Returns:
    - reduction: np.array of shape [n_sites] or [n_events x n_sites]
    """
    canopy = np.clip(exposure_gdf.get("canopy_cover", np.zeros(len(exposure_gdf))), 0, 100)

    # Temperature reduction from canopy
    slope = -max_cooling / plateau_at
    temp_reduction = np.where(
        canopy <= plateau_at,
        slope * canopy,
        -max_cooling
    )  # shape: [n_sites]

    # Convert to VPD reduction
    vpd_reduction = -vpd_sensitivity * temp_reduction  # still positive
    return vpd_reduction  # shape: [n_sites]

def helper_adj_fcn_canopy_tmax(
    hazard_matrix,
    exposure_gdf,
    max_cooling=2.5,
    plateau_at=50
):
    """
    Return REDUCTION in Tmax hazard intensity based on canopy cover,
    using a linear-to-plateau cooling function.

    Cooling is applied as:
        - Linear up to `plateau_at` % canopy
        - Constant `-max_cooling` beyond that

    Parameters:
    - hazard_matrix: np.array of shape [n_events x n_sites]
    - exposure_gdf: GeoDataFrame with 'canopy_cover' column (0–100%)
    - max_cooling: Maximum cooling in °C (positive, e.g. 2.5)
    - plateau_at: Canopy cover % at which max_cooling is reached (e.g. 50)

    Returns:
    - reduction: np.array of shape [n_sites] or [n_events x n_sites]
    """
    canopy = exposure_gdf.get("canopy_cover", np.zeros(len(exposure_gdf)))
    canopy = np.clip(canopy, 0, 100)

    slope = -max_cooling / plateau_at
    reduction = np.where(
        canopy <= plateau_at,
        slope * canopy,
        -max_cooling
    )

    return reduction  # shape: [n_sites]


import os
from pathlib import Path
import requests

def download_if_missing(url, dest_folder, filename=None):
    """
    Download a file from `url` into `dest_folder` if not already present.

    Parameters:
    - url (str): Direct download URL to the file
    - dest_folder (str or Path): Folder to save the file
    - filename (str, optional): Custom filename (otherwise inferred from URL)

    Returns:
    - Path to the downloaded file
    """
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]  # Clean URL artifacts
    dest_path = dest_folder / filename

    if dest_path.exists():
        print(f"✅ File already exists: {dest_path}")
    else:
        print(f"⬇️ Downloading {filename} to {dest_path} ...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Download completed.")
        else:
            raise Exception(f"❌ Failed to download: HTTP {response.status_code}")

    return dest_path

def generate_hansen_tile_urls_from_bounds(lat_min, lat_max, lon_min, lon_max):
    """
    Generate Hansen tile filenames and URLs based on bounding box.

    Parameters:
    - base_url (str): URL prefix ending with '/' (e.g., Dropbox folder, GEE repo)
    - lat_min, lat_max (int): Latitude bounds in degrees (e.g., 10, 30)
    - lon_min, lon_max (int): Longitude bounds in degrees (e.g., -110, -90)

    Returns:
    - dict: {filename: full_url}
    """
    tile_urls = {}
    base_url = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2019-v1.7/"

    # Loop over 10-degree tiles (based on naming convention)
    for lat in range(lat_min, lat_max + 1, 10):
        lat_dir = "N" if lat >= 0 else "S"
        lat_str = f"{abs(lat):02d}{lat_dir}"

        for lon in range(lon_min, lon_max + 1, 10):
            lon_dir = "E" if lon >= 0 else "W"
            lon_str = f"{abs(lon):03d}{lon_dir}"

            filename = f"Hansen_GFC-2019-v1.7_treecover2000_{lat_str}_{lon_str}.tif"
            tile_urls[filename] = f"{base_url}{filename}"

    return tile_urls


import xarray as xr
import os
from pathlib import Path
import requests

def download_terraclimate_data(
    output_dir,
    country_code="MEX",
    lat_bounds=(32, 13),
    lon_bounds=(-120, -85),
    variables=("tmax", "vpd"),
    years=range(1985, 2023),
    download_type="historical",  # or "scenario"
    scenario="plus2C",
    scenario_prefix="2c",
    skip_existing=True
):
    """
    Download and crop TerraClimate data (historical or scenario).

    Parameters:
    - output_dir (str or Path): Destination folder
    - country_code (str): Used in saved file names
    - lat_bounds (tuple): (max_lat, min_lat)
    - lon_bounds (tuple): (min_lon, max_lon)
    - variables (iterable): List of variable names to download
    - years (iterable): Range/list of years
    - download_type (str): "historical" or "scenario"
    - scenario (str): Scenario name for future data
    - scenario_prefix (str): Prefix in filename (e.g., "2c" for plus2C)
    - skip_existing (bool): If True, skip downloading files that already exist
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        for year in years:
            if download_type == "historical":
                url = f"http://thredds.northwestknowledge.net:8080/thredds/dodsC/TERRACLIMATE_ALL/data/TerraClimate_{var}_{year}.nc"
                filename = f"TerraClimate_{var}_{year}_{country_code}.nc"
            elif download_type == "scenario":
                url = f"http://thredds.northwestknowledge.net:8080/thredds/fileServer/TERRACLIMATE_ALL/data_{scenario}/TerraClimate_{scenario_prefix}_{var}_{year}.nc"
                filename = f"TerraClimate_{scenario}_{var}_{year}_{country_code}.nc"
            else:
                raise ValueError("Invalid download_type: must be 'historical' or 'scenario'.")

            out_path = output_dir / filename
            if skip_existing and out_path.exists():
                print(f"✅ Skipping existing: {out_path.name}")
                continue

            print(f"🔄 Downloading {var} {year} for {country_code}...")

            try:
                if download_type == "historical":
                    ds = xr.open_dataset(url)
                else:
                    tmp_path = output_dir / f"_tmp_{var}_{year}.nc"
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        f.write(r.content)
                    ds = xr.open_dataset(tmp_path)

                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))  # Normalize
                ds_crop = ds[var].sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
                ds_crop.to_netcdf(out_path)
                print(f"✅ Saved: {out_path.name}")

                if download_type == "scenario":
                    tmp_path.unlink()

            except Exception as e:
                print(f"❌ Failed for {var} {year}: {e}")

def print_summary_statistics(df, measure_names):
    """
    Print mean and 90% confidence intervals for AAI (present, future),
    total climate risk, and cost-benefit metrics for each measure.
    """
    from tabulate import tabulate

    def summarize(col):
        data = df[col].dropna()
        mean = data.mean()
        ci_lower = data.quantile(0.05)
        ci_upper = data.quantile(0.95)
        return mean, ci_lower, ci_upper

    rows = []

    # --- AAI Risk Summary ---
    base_present = "no measure - risk - present"
    base_future = "no measure - risk - future"
    rows.append(["AAI (Present)", "No measure", *summarize(base_present)])
    rows.append(["AAI (Future)", "No measure", *summarize(base_future)])

    for meas in measure_names:
        rows.append(["AAI (Present)", meas, *summarize(f"{meas} - risk - present")])
        rows.append(["AAI (Future)", meas, *summarize(f"{meas} - risk - future")])

    # --- Total Climate Risk ---
    rows.append(["Total Climate Risk", "All", *summarize("tot_climate_risk")])

    # --- Cost, Benefit, Cost-Benefit Ratio ---
    for meas in measure_names:
        rows.append(["Cost (NPV)", meas, *summarize(f"{meas} - cost_meas - future")])
        rows.append(["Benefit (NPV)", meas, *summarize(f"{meas} Benef")])
        rows.append(["Cost-Benefit", meas, *summarize(f"{meas} CostBen")])

    # --- Display table ---
    print(tabulate(rows, headers=["Metric", "Scenario", "Mean", "5th Percentile", "95th Percentile"], floatfmt=".2f"))


def plot_kde_panel(df, column_map, *, suptitle=None, value_format="${:,.0f}", figsize=(18, 5)):
    """
    General function for plotting 1xN KDE subplots from a dict of label → [columns].

    Parameters:
    - df: DataFrame with data
    - column_map: OrderedDict or dict: subplot_title -> [(label, column_name)]
    - value_format: formatter for mean values in legend
    - suptitle: optional string
    - figsize: tuple for figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_subplots = len(column_map)
    fig, axs = plt.subplots(1, n_subplots, figsize=figsize, sharey=False)
    if n_subplots == 1:
        axs = [axs]

    palette = sns.color_palette("Set1", n_colors=10)

    for ax, (title, entries) in zip(axs, column_map.items()):
        for i, (label, col) in enumerate(entries):
            if col in df.columns:
                val = df[col].dropna()
                if val.var() > 0:
                    mean_val = val.mean()
                    label_with_mean = f"{label} (mean: {value_format.format(mean_val)})"
                    sns.kdeplot(val, ax=ax, label=label_with_mean,
                                fill=True, color=palette[i], linewidth=2, alpha=0.3)
                    ax.axvline(mean_val, color=palette[i], linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.show()

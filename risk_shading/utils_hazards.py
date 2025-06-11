import os
import glob
import xarray as xr
import numpy as np
from scipy import sparse
from scipy.stats import genextreme
from climada.hazard import Hazard, Centroids
from climada.util.constants import DEF_CRS
from climate_indices.indices import spei, Distribution
from climate_indices.compute import Periodicity


def clip_to_region(ds, lat_bounds, lon_bounds):
    return ds.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))

def get_aggregates(path):
    def _load(varname):
        files = sorted(glob.glob(os.path.join(path, f"TerraClimate*{varname}*CA.nc")))
        print(f"📂 Loading {len(files)} files for {varname}")
        if len(files) == 0:
            raise FileNotFoundError(f"No files found for variable '{varname}' in {path}")
        return xr.concat([xr.open_dataset(f)[varname] for f in files], dim="time")

    tmin = _load("tmin")
    tmax = _load("tmax")
    ppt = _load("ppt")
    pet = _load("pet")

    tmean = (tmin + tmax) / 2
    tmean_annual = tmean.groupby("time.year").max("time")
    ppt_annual = ppt.groupby("time.year").sum("time")

    return (
        clip_to_region(tmean_annual, (20.0, 17.5), (-72.0, -68.0)),
        clip_to_region(ppt_annual, (20.0, 17.5), (-72.0, -68.0)),
        clip_to_region(ppt, (20.0, 17.5), (-72.0, -68.0)),
        clip_to_region(pet, (20.0, 17.5), (-72.0, -68.0))
    )

def compute_spei_3(ppt, pet):
    clim = ppt - pet
    out = np.full(clim.shape, np.nan)
    start_year = int(clim.time.dt.year.values[0])
    n_months = clim.sizes["time"]

    for i in range(clim.lat.size):
        for j in range(clim.lon.size):
            ppt_series = ppt[:, i, j].values.astype(float)
            pet_series = pet[:, i, j].values.astype(float)

            if np.isnan(ppt_series).all() or np.isnan(pet_series).all():
                continue
            if np.isnan(ppt_series).any():
                ppt_series[np.isnan(ppt_series)] = np.nanmean(ppt_series)
            if np.isnan(pet_series).any():
                pet_series[np.isnan(pet_series)] = np.nanmean(pet_series)

            try:
                spei_series = spei(
                    precips_mm=ppt_series,
                    pet_mm=pet_series,
                    scale=3,
                    distribution=Distribution.gamma,
                    periodicity=Periodicity.monthly,
                    data_start_year=start_year,
                    calibration_year_initial=start_year,
                    calibration_year_final=start_year + (n_months // 12) - 1
                )
                out[:, i, j] = spei_series
            except Exception as e:
                print(f"⚠️ Grid cell ({i},{j}) failed: {e}")
                continue

    return xr.DataArray(
        out,
        coords={"time": clim.time, "lat": clim.lat, "lon": clim.lon},
        dims=["time", "lat", "lon"]
    )

def generate_gev_sample_field(dataarray, n_years=100, start_year=2101, invert=False):
    synthetic_years = np.arange(start_year, start_year + n_years)

    def _fit_and_sample(ts):
        ts = ts[~np.isnan(ts)]
        if len(ts) < 10:
            return np.full(n_years, np.nan)
        try:
            ts = -ts if invert else ts
            shape, loc, scale = genextreme.fit(ts)
            sampled = genextreme.rvs(shape, loc=loc, scale=scale, size=n_years)
            return -sampled if invert else sampled
        except Exception:
            return np.full(n_years, np.nan)

    stacked = dataarray.stack(grid=("lat", "lon"))
    sampled_values = np.array([
        _fit_and_sample(stacked.isel(grid=i).values)
        for i in range(stacked.sizes["grid"])
    ])
    reshaped = sampled_values.reshape((dataarray.sizes["lat"], dataarray.sizes["lon"], n_years))
    reshaped = np.transpose(reshaped, (2, 0, 1))

    return xr.DataArray(
        reshaped,
        coords={"year": synthetic_years, "lat": dataarray.lat, "lon": dataarray.lon},
        dims=["year", "lat", "lon"]
    )

def create_hazard_from_array(dataarray, haz_type, units, start_year=2101):
    years = np.arange(start_year, start_year + dataarray.sizes["year"])
    lat_vals = dataarray.lat.values
    lon_vals = dataarray.lon.values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    intensity_data = np.vstack([
        dataarray.sel(year=yr).values.reshape(1, -1)
        for yr in years
    ])
    intensity_data = np.nan_to_num(intensity_data, nan=0)

    intensity = sparse.csr_matrix(intensity_data)
    fraction = intensity.copy()
    fraction.data.fill(1)

    centroids = Centroids(
        lat=lat_grid.ravel(),
        lon=lon_grid.ravel(),
        crs=DEF_CRS
    )

    return Hazard(
        haz_type=haz_type,
        intensity=intensity,
        fraction=fraction,
        centroids=centroids,
        units=units,
        event_id=years.tolist(),
        event_name=[f"{haz_type}_{y}" for y in years],
        date=np.array([np.datetime64(f"{y}-07-01").astype("datetime64[D]").astype(int) for y in years]),
        orig=np.zeros(len(years), dtype=bool),
        frequency=np.ones(len(years)) / len(years)
    )

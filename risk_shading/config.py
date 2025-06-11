from pathlib import Path

DATA_DIR = Path("/Users/szelie/data/unu")

RASTER_PATHS = {
    "dominican_republic": {
        # 🌲 Canopy height: ETH Zurich Global Canopy Height dataset (10m)
        # Download from: https://planetarycomputer.microsoft.com/dataset/global-forest-canopy-height
        "canopy_height": [
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N18W069_Map.tif",
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N18W072_Map.tif",
        ],

        # 🌳 Forest cover: Hansen Global Forest Change / Tree Cover (2000 baseline, updated annually)
        # Download from: https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.7.html
        "forest_cover": [DATA_DIR / "tree_cover/20N_070W.tif",
        DATA_DIR /"tree_cover/20N_080W.tif",
        DATA_DIR / "tree_cover/20N_090W.tif"
    ],

        # 🌍 Land cover: ESA WorldCover 10m v2.0 (2021)
        # Download from: https://esa-worldcover.org/en
        "land_cover": [
            DATA_DIR / "WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP/ESA_WorldCover_10m_2021_v200_N18W075_Map/ESA_WorldCover_10m_2021_v200_N18W075_Map.tif",
            DATA_DIR / "WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP/ESA_WorldCover_10m_2021_v200_N15W072_Map/ESA_WorldCover_10m_2021_v200_N15W072_Map.tif",
            DATA_DIR / "WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP/ESA_WorldCover_10m_2021_v200_N18W072_Map/ESA_WorldCover_10m_2021_v200_N18W072_Map.tif", 
    DATA_DIR / "WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP/ESA_WorldCover_10m_2021_v200_N18W069_Map/ESA_WorldCover_10m_2021_v200_N18W069_Map.tif"    ],
    },

    "guatemala": {
        # 🌲 Canopy height: ETH Global Canopy Height
        "canopy_height": [
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N14W091_Map.tif",
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N15W090_Map.tif",
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N16W090_Map.tif",
        ],

        # 🌳 Forest cover: Hansen Tree Cover
        "forest_cover": [
            DATA_DIR / "tree_cover/20N_090W.tif",
            DATA_DIR / "tree_cover/20N_080W.tif",
        ],

        # 🌍 Land cover: ESA WorldCover
        "land_cover": [
            DATA_DIR / "WORLDCOVER/ESA_WorldCover_10m_2021_v200_N14W091_Map.tif",
            DATA_DIR / "WORLDCOVER/ESA_WorldCover_10m_2021_v200_N15W090_Map.tif",
            DATA_DIR / "WORLDCOVER/ESA_WorldCover_10m_2021_v200_N16W090_Map.tif",
        ],
    },

    "mexico": {
        # 🌲 Canopy height: ETH Global Canopy Height
        "canopy_height": [
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N16W096_Map.tif",
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N17W096_Map.tif",
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N18W096_Map.tif",
            DATA_DIR / "canopy/ETH_GlobalCanopyHeight_10m_2020_N19W096_Map.tif",
        ],

        # 🌳 Forest cover: Hansen Tree Cover
        "forest_cover": [
            DATA_DIR / "tree_cover/20N_100W.tif",
            DATA_DIR / "tree_cover/20N_090W.tif",
        ],

        # 🌍 Land cover: ESA WorldCover
        "land_cover": [
            DATA_DIR / "WORLDCOVER/ESA_WorldCover_10m_2021_v200_N16W096_Map.tif",
            DATA_DIR / "WORLDCOVER/ESA_WorldCover_10m_2021_v200_N17W096_Map.tif",
            DATA_DIR / "WORLDCOVER/ESA_WorldCover_10m_2021_v200_N18W096_Map.tif",
            DATA_DIR / "WORLDCOVER/ESA_WorldCover_10m_2021_v200_N19W096_Map.tif",
        ],
    },
}

import os

# Base data directory
BASE_DATA_DIR = "/Users/szelie/data/unu"

# Path to EcoCrop database
ECOCROP_PATH = os.path.join(BASE_DATA_DIR, "EcoCrop_DB.csv")


#########################################################################################


from pathlib import Path

# Bounding boxes for countries
REGION_BOUNDS = {
    "dominican_republic": {"lat": (20.0, 17.5), "lon": (-72.0, -68.0)},
    "guatemala": {"lat": (18.0, 13.5), "lon": (-92.5, -88.0)},
    "mexico": {"lat": (20.5, 14.5), "lon": (-98.0, -90.0)}
}

# TerraClimate paths by scenario
TERRACLIMATE_PATHS = {
    "plus2C": Path("/Users/szelie/data/unu/terra_climate_scenarios_2c"),
    "historical": Path("/Users/szelie/data/unu/terra_climate/")
}

# Default parameters for hazard simulation
DEFAULT_START_YEAR = 2000
DEFAULT_SAMPLE_YEARS = 100


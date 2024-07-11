"""
**Data Bridge**

The data_bridge module contains all the classes and functions used to load and process the input data for the model.
"""

from pathlib import Path

import pandas as pd

# Default file paths
FILE_PATH = Path(__file__).parent
PACKAGE_PATH = FILE_PATH.parent
DATA_PATH = PACKAGE_PATH / "data"

# Default file names
FILE_OCCUPATIONS = DATA_PATH / "nat5d_6d_M2018_dl.xlsx"
FILE_MOBILITY_NETWORK = DATA_PATH / "edgelist_cc_mobility_merge.csv"
FILE_TECHNOLOGIES = DATA_PATH / "6digitNAICS_tech.csv"
NODE_OCC = DATA_PATH / "node_occ_name.csv"

CANADA_LABOUR_FORCE = {
    "Alberta.a": 2.59e6,
    "British Columbia.a": 2.93e6,
    "Manitoba.a": 0.73e6,
    "New Brunswick.a": 0.411e6,
    "Newfoundland and Labrador.a": 0.26e6,
    "Nova Scotia.a": 0.526e6,
    "Ontario.a": 4.2e6,
    "Ontario.b": 4.2e6,
    "Prince Edward Island.a": 0.096e6,
    "Quebec.a": 2.35e6,
    "Quebec.b": 2.35e6,
    "Saskatchewan.a": 0.613e6,
    "National": 21e6,
}

df_nodes = pd.read_csv(NODE_OCC)
OCC_NAME_DICT = dict(zip(df_nodes["node_id"], df_nodes["OCC_title"]))


def get_occupation_name(occ_id: int) -> str:
    """
    Get the name of an occupation given its ID.
    """
    return OCC_NAME_DICT.get(occ_id, "Unknown")


def generic_loader(file_path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Generic loader for data files. It uses the file extension to determine the appropriate loader.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix == ".xlsx":
        return pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {file_path.suffix}")

import os

import geopandas as gpd


def load_roi_gdf(path: os.PathLike) -> gpd.GeoDataFrame:
    return (
        gpd.read_file(path)
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
        .dropna(subset=["hex_id"])
        .rename(columns={"hex_id": "region_id"})[["region_id", "geometry"]]
    )

import os
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import pandas as pd

from sea_surface_temp.zonal import aggregate_roi_temperatures


class SspPaths(NamedTuple):
    temperature_path: Path
    mapping_path: Path


def output_paths(root_path: os.PathLike, source_id: str, member_id: str) -> SspPaths:
    root_path = Path(root_path)

    name = Path(f"{source_id}", f"{member_id}.parquet")
    temperature_path = root_path / "temperatures" / name
    mapping_path = root_path / "roi_pixels" / name
    temperature_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    return SspPaths(temperature_path, mapping_path)


# Esta función lee todos los archivos de temperaturas que calculamos, así como
# las tablas que nos dicen qué polígonos caen dentro de cada píxel. Si
# un polígono cae dentro de un solo píxel, se le asigna su valor directamente
# (a menos que dicho píxel no tenga temperatura, en cuyo caso se busca un
# píxel cercano). Si un polígono cae dentro de varios píxeles, se calcula
# un promedio ponderado por el área de intersección entre el polígono y cada
# píxel.
def generate_roi_monthly_temperatures(
    root_path: os.PathLike,
    *,
    source_id: str,
    member_id: str,
    year: int,
    month: int | None = None,
) -> pd.Series:
    ssp_paths = output_paths(root_path, source_id, member_id)
    filters = [("year", "==", year)]
    if month is not None:
        filters.append(("month", "==", month))
    temperatures = pd.read_parquet(ssp_paths.temperature_path, filters=filters)
    matches = pd.read_parquet(ssp_paths.mapping_path).merge(
        temperatures, on=["grid_id", "pixel_id"]
    )
    return aggregate_roi_temperatures(matches)


def generate_roi_temperature_df(  # noqa: PLR0913
    root_path: os.PathLike,
    roi_gdf: gpd.GeoDataFrame,
    *,
    source_id: str,
    member_id: str,
    year: int,
    month: int,
) -> gpd.GeoDataFrame:
    temperatures = generate_roi_monthly_temperatures(
        root_path, source_id=source_id, member_id=member_id, year=year, month=month
    ).droplevel("month")
    return roi_gdf.assign(temperature_c=roi_gdf["region_id"].map(temperatures))


def generate_roi_hottest_month_temperature_df(
    root_path: os.PathLike,
    roi_gdf: gpd.GeoDataFrame,
    *,
    source_id: str,
    member_id: str,
    year: int,
) -> gpd.GeoDataFrame:
    monthly_temperatures = generate_roi_monthly_temperatures(
        root_path, source_id=source_id, member_id=member_id, year=year
    )
    hottest_temperatures = monthly_temperatures.groupby(level="region_id").max()
    return roi_gdf.assign(temperature_c=roi_gdf["region_id"].map(hottest_temperatures))

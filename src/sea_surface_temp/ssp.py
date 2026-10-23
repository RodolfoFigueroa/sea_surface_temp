import os
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import pandas as pd


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
    matches = (
        pd.read_parquet(ssp_paths.mapping_path)
        .merge(temperatures, on=["grid_id", "pixel_id"])
        .dropna(subset=["temperature_c"])
    )
    group_columns = ["region_id", "month"]
    weighted_temperatures = (
        matches.loc[lambda frame: frame["intersection_area_m2"] > 0]
        .assign(
            weighted_temperature=lambda frame: (
                frame["temperature_c"] * frame["intersection_area_m2"]
            )
        )
        .groupby(group_columns)
        .agg(
            weighted_temperature=("weighted_temperature", "sum"),
            overlap_area=("intersection_area_m2", "sum"),
        )
        .assign(
            temperature_c=lambda frame: (
                frame["weighted_temperature"] / frame["overlap_area"]
            )
        )["temperature_c"]
    )
    nearest_temperatures = (
        matches.loc[lambda frame: frame["within_cutoff"]]
        .sort_values([*group_columns, "distance_km", "pixel_id"])
        .drop_duplicates(group_columns)
        .set_index(group_columns)["temperature_c"]
    )
    return weighted_temperatures.combine_first(nearest_temperatures).rename(
        "temperature_c"
    )


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

import os
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.windows import Window

from sea_surface_temp.zonal import aggregate_roi_temperatures

AREA_CRS = "EPSG:6933"
DISTANCE_CRS = "EPSG:6372"
CUTOFF_KM = 50.0


class _GridMapping(NamedTuple):
    window: Window
    relationships: pd.DataFrame


def _candidate_window(
    dataset: rasterio.io.DatasetReader,
    rois: gpd.GeoDataFrame,
    points: gpd.GeoSeries,
) -> Window:
    # Square buffers conservatively enclose the circular distance search.
    search = points.buffer(CUTOFF_KM * 1000).envelope.to_crs(dataset.crs)
    bounds = np.stack([rois.to_crs(dataset.crs).total_bounds, search.total_bounds])
    xmin, ymin = bounds[:, :2].min(axis=0)
    xmax, ymax = bounds[:, 2:].max(axis=0)
    cols, rows = ~dataset.transform * (
        np.array([xmin, xmin, xmax, xmax]),
        np.array([ymin, ymax, ymin, ymax]),
    )
    col_start, col_stop = np.clip(
        [np.floor(cols.min()), np.ceil(cols.max())], 0, dataset.width
    ).astype(int)
    row_start, row_stop = np.clip(
        [np.floor(rows.min()), np.ceil(rows.max())], 0, dataset.height
    ).astype(int)
    return Window.from_slices((row_start, row_stop), (col_start, col_stop))


def _grid_cells(dataset: rasterio.io.DatasetReader, window: Window) -> gpd.GeoDataFrame:
    rows, cols = np.indices((int(window.height), int(window.width)))
    rows = rows.ravel() + int(window.row_off)
    cols = cols.ravel() + int(window.col_off)
    x, y = dataset.transform * (
        cols[:, None] + np.array([0, 1, 1, 0]),
        rows[:, None] + np.array([0, 0, 1, 1]),
    )
    center_x, center_y = dataset.transform * (cols + 0.5, rows + 0.5)
    return gpd.GeoDataFrame(
        {
            "pixel_id": rows * dataset.width + cols,
            "row": rows - int(window.row_off),
            "col": cols - int(window.col_off),
            "center_x": center_x,
            "center_y": center_y,
        },
        geometry=shapely.polygons(np.stack([x, y], axis=-1)),
        crs=dataset.crs,
    )


def _relationships(
    rois: gpd.GeoDataFrame, points: gpd.GeoSeries, cells: gpd.GeoDataFrame
) -> pd.DataFrame:
    roi_geometry = rois.to_crs(str(cells.crs)).geometry
    intersections = (
        gpd.GeoDataFrame(geometry=roi_geometry)
        .rename_axis("roi_index")
        .sjoin(cells[["geometry"]].rename_axis("cell_index"))
        .reset_index()[["roi_index", "cell_index"]]
    )
    centers = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(cells["center_x"], cells["center_y"]),
        crs=cells.crs,
    ).to_crs(DISTANCE_CRS)
    nearby = (
        gpd.GeoDataFrame(geometry=points)
        .rename_axis("roi_index")
        .sjoin(
            centers.rename_axis("cell_index"),
            predicate="dwithin",
            distance=CUTOFF_KM * 1000,
        )
        .reset_index()[["roi_index", "cell_index"]]
    )
    pairs = pd.concat([intersections, nearby], ignore_index=True).drop_duplicates()
    roi_indices = pairs["roi_index"].to_numpy()
    cell_indices = pairs["cell_index"].to_numpy()
    distances = (
        points.iloc[roi_indices]
        .reset_index(drop=True)
        .distance(
            centers.geometry.iloc[cell_indices].reset_index(drop=True), align=False
        )
        .to_numpy()
        / 1000
    )
    overlap = shapely.intersection(
        roi_geometry.to_numpy()[roi_indices], cells.geometry.to_numpy()[cell_indices]
    )
    areas = gpd.GeoSeries(overlap, crs=cells.crs).to_crs(AREA_CRS).area.to_numpy()
    return (
        pd.DataFrame(cells[["pixel_id", "row", "col"]].iloc[cell_indices])
        .reset_index(drop=True)
        .assign(
            region_id=rois["region_id"].to_numpy()[roi_indices],
            intersection_area_m2=areas,
            distance_km=distances,
            within_cutoff=distances <= CUTOFF_KM,
        )
    )


def generate_roi_monthly_temperatures(
    root_path: os.PathLike,
    roi_gdf: gpd.GeoDataFrame,
    *,
    year: int,
    month: int | None = None,
) -> pd.Series:
    root_path = Path(root_path)
    paths = (
        [root_path / f"{year:04d}-{month:02d}.tif"]
        if month is not None
        else sorted(root_path.glob(f"{year:04d}-[0-9][0-9].tif"))
    )
    if not paths:
        message = f"No NOAA TIFF files found for year {year} in {root_path}"
        raise FileNotFoundError(message)
    rois = roi_gdf[["region_id", roi_gdf.geometry.name]].to_crs("EPSG:4326")
    rois = rois.reset_index(drop=True)
    points = rois.geometry.representative_point().to_crs(DISTANCE_CRS)
    cache: dict[str, _GridMapping] = {}
    monthly: list[pd.Series] = []
    for path in paths:
        with rasterio.open(path) as dataset:
            if rois.empty:
                continue
            key = str((dataset.crs, dataset.transform, dataset.width, dataset.height))
            if key not in cache:
                window = _candidate_window(dataset, rois, points)
                cells = _grid_cells(dataset, window)
                cache[key] = _GridMapping(window, _relationships(rois, points, cells))
            mapping = cache[key]
            matches = mapping.relationships
            if matches.empty:
                continue
            values = dataset.read(1, window=mapping.window, masked=True)
            temperatures = values[matches["row"].to_numpy(), matches["col"].to_numpy()]
            monthly.append(
                aggregate_roi_temperatures(
                    matches.assign(
                        month=int(path.stem.split("-")[1]),
                        temperature_c=temperatures.astype("float64").filled(np.nan),
                    )
                )
            )
    if not monthly:
        return pd.Series(
            index=pd.MultiIndex.from_arrays([[], []], names=["region_id", "month"]),
            dtype="float64",
            name="temperature_c",
        )
    return pd.concat(monthly).sort_index()


def generate_roi_temperature_df(
    root_path: os.PathLike,
    roi_gdf: gpd.GeoDataFrame,
    *,
    year: int,
    month: int,
) -> gpd.GeoDataFrame:
    temperatures = generate_roi_monthly_temperatures(
        root_path, roi_gdf, year=year, month=month
    ).droplevel("month")
    return roi_gdf.assign(temperature_c=roi_gdf["region_id"].map(temperatures))


def generate_roi_hottest_month_temperature_df(
    root_path: os.PathLike,
    roi_gdf: gpd.GeoDataFrame,
    *,
    year: int,
) -> gpd.GeoDataFrame:
    temperatures = generate_roi_monthly_temperatures(root_path, roi_gdf, year=year)
    hottest = temperatures.groupby(level="region_id").max()
    return roi_gdf.assign(temperature_c=roi_gdf["region_id"].map(hottest))

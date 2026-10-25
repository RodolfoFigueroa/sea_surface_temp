import pandas as pd


def aggregate_roi_temperatures(matches: pd.DataFrame) -> pd.Series:
    matches = matches.dropna(subset=["temperature_c"])
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

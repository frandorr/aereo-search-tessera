"""Function-based Hamilton nodes for GeoTessera search.

These nodes replace the class-based :class:`TesseraSearchPlugin` with plain
functions that Hamilton can compose into a DAG.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
from aereo.schemas import AssetSchema
from pandera.typing.geopandas import GeoDataFrame
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from structlog import get_logger

from aereo.search_tessera.core import (
    DEFAULT_BASE_URL,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_TESSERA_VERSION,
    check_href_exists,
    load_tiles_for_region,
    tile_to_bounds,
    tile_utm_info,
    _ensure_registry,
)

logger = get_logger()

# Module-level variable consumed by the plugin discovery machinery.
supported_collections = ("geotessera",)


def _empty_result() -> GeoDataFrame:
    """Return an empty validated GeoDataFrame with AssetSchema columns."""
    columns = list(AssetSchema.to_schema().columns.keys())
    if "geometry" not in columns:
        columns.append("geometry")
    gdf = gpd.GeoDataFrame(columns=columns, geometry="geometry")
    return cast(GeoDataFrame, AssetSchema.validate(gdf))


def _normalize_datetime(dt: datetime | str | None) -> datetime | None:
    """Ensure datetime is timezone-aware UTC."""
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def search_assets(
    aoi: BaseGeometry | None = None,
    start_datetime: datetime | str | None = None,
    end_datetime: datetime | str | None = None,
    collections: Sequence[str] | None = None,
    search_params: dict[str, Any] | None = None,
) -> GeoDataFrame:
    """Search GeoTessera registry and return AssetSchema-compliant GeoDataFrame.

    Args:
        aoi: Optional geometry for spatial filtering.
        start_datetime: Start of temporal range.
        end_datetime: End of temporal range.
        collections: Sequence of collection names. Must contain ``"geotessera"``
            to return results.
        search_params: Additional parameters including ``base_url``,
            ``tessera_version``, ``registry_path``, ``refresh_registry``,
            and ``check_href``.

    Returns:
        Validated GeoDataFrame containing matched tile metadata.
    """
    if search_params is None:
        search_params = {}

    # If collections are explicitly provided and don't include geotessera, return empty
    if collections is not None and "geotessera" not in collections:
        return _empty_result()

    # Bounding box for region
    if aoi is not None:
        bbox = aoi.bounds
    else:
        bbox = (-180.0, -90.0, 180.0, 90.0)

    # Normalize datetimes
    q_start = _normalize_datetime(start_datetime)
    q_end = _normalize_datetime(end_datetime)

    # Determine which years intersect the query range
    matching_years: list[int] = []
    for yr in range(2017, 2026):
        yr_start = datetime(yr, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        yr_end = datetime(yr, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        if q_start is not None and yr_end < q_start:
            continue
        if q_end is not None and yr_start > q_end:
            continue
        matching_years.append(yr)

    if not matching_years:
        return _empty_result()

    # Resolve paths and options
    base_url = search_params.get("base_url", DEFAULT_BASE_URL)
    tessera_version = search_params.get("tessera_version", DEFAULT_TESSERA_VERSION)
    refresh_registry = search_params.get("refresh_registry", False)

    # Handle registry path and bootstrapping
    if "registry_path" in search_params:
        registry_path = Path(search_params["registry_path"]).expanduser().resolve()
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_path}")
    else:
        registry_path = DEFAULT_REGISTRY_PATH.expanduser().resolve()
        url = f"{base_url}/{tessera_version}/registry.parquet"
        _ensure_registry(registry_path, url, refresh=refresh_registry)

    # Load tiles from parquet
    tiles_dfs: list[pd.DataFrame] = []
    for year in matching_years:
        df = load_tiles_for_region(bbox, year, registry_path)
        if not df.empty:
            tiles_dfs.append(df)

    if not tiles_dfs:
        return _empty_result()

    tiles_df = pd.concat(tiles_dfs, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for _, tile in tiles_df.iterrows():
        lon = float(cast(Any, tile["lon"]))
        lat = float(cast(Any, tile["lat"]))
        year = int(cast(Any, tile["year"]))
        utm = tile_utm_info(lon, lat)
        grid_name = f"grid_{lon:.2f}_{lat:.2f}"
        wgs_bounds = tile_to_bounds(lon, lat)

        rows.append(
            {
                "id": f"{grid_name}_{year}",
                "collection": "geotessera",
                "geometry": box(*wgs_bounds),
                "start_time": datetime(year, 1, 1, tzinfo=timezone.utc),
                "end_time": datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                "href": f"{base_url}/{tessera_version}/global_0.1_degree_representation/{year}/{grid_name}/{grid_name}.npy",
                "tile_lon": lon,
                "tile_lat": lat,
                "tile_year": year,
                "tile_utm_crs": utm["crs"],
                "tile_utm_epsg": utm["epsg"],
                "tile_utm_bbox": utm["utm_bbox"],
                "tile_shape": utm["shape"],
                "tile_hash": tile["hash"],
            }
        )

    if not rows:
        return _empty_result()

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    # Apply high-precision geometry intersection if a shape was provided
    if aoi is not None:
        gdf = gdf[gdf.intersects(aoi)]

    # Check if href exists if check_href is True
    if search_params.get("check_href", False) and not gdf.empty:
        from concurrent.futures import ThreadPoolExecutor

        hrefs = gdf["href"].tolist()
        max_workers = min(32, len(hrefs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            exists_flags = list(executor.map(check_href_exists, hrefs))
        gdf = gdf[exists_flags]

    if gdf.empty:
        return _empty_result()

    validated = AssetSchema.validate(gdf)
    return cast(GeoDataFrame, cast(object, validated))


def search_results(search_assets: GeoDataFrame) -> GeoDataFrame:
    """Return validated search results.

    This is the output boundary of the search stage. Downstream Hamilton
    nodes depend on ``search_results`` so that the plugin can be swapped
    without changing the DAG contract.
    """
    return search_assets

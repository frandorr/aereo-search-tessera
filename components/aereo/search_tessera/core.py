"""Search provider for GeoTessera satellite embeddings.

Queries the registry parquet directly to locate tiles within the AOI and time range.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from aereo.interfaces import SearchProvider
from aereo.schemas import AssetSchema
from pandera.typing.geopandas import GeoDataFrame
from pyproj import Transformer
from shapely.geometry import box
from structlog import get_logger

logger = get_logger()

DEFAULT_REGISTRY_PATH = Path.home() / ".cache" / "geotessera" / "registry.parquet"


def tile_to_bounds(lon: float, lat: float) -> tuple[float, float, float, float]:
    """WGS84 bounds for a 0.1° tile centered at (lon, lat)."""
    return (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)


def tile_utm_info(tile_lon: float, tile_lat: float, pixel_size: float = 10.0) -> dict[str, Any]:
    """Compute UTM EPSG, pixel-aligned bbox, and shape for a tile.

    Verified identical to GeoTessera's from_origin(xmin, ymax, 10.0, 10.0).
    """
    west, south, east, north = tile_to_bounds(tile_lon, tile_lat)
    zone = int((tile_lon + 180) / 6) + 1
    epsg = 32600 + zone if tile_lat >= 0 else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = transformer.transform([west, east, west, east], [north, north, south, south])
    xmin, ymax = min(xs), max(ys)
    width = int(round((max(xs) - xmin) / pixel_size))
    height = int(round((ymax - min(ys)) / pixel_size))
    return {
        "epsg": epsg,
        "crs": f"EPSG:{epsg}",
        "utm_bbox": (xmin, ymax - height * pixel_size, xmin + width * pixel_size, ymax),
        "shape": (height, width),
    }


def load_tiles_for_region(
    bbox: tuple[float, float, float, float],
    year: int,
    registry_path: Path,
) -> pd.DataFrame:
    """Query tiles from parquet with predicate pushdown (~40ms)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    expansion = 0.05
    table = pq.read_table(
        registry_path,
        columns=["lon", "lat", "year", "hash", "file_size"],
        filters=[
            ("year", "=", year),
            ("lon", ">=", min_lon - expansion),
            ("lon", "<=", max_lon + expansion),
            ("lat", ">=", min_lat - expansion),
            ("lat", "<=", max_lat + expansion),
        ],
    )
    return table.to_pandas().drop_duplicates(subset=["year", "lon", "lat"])


def _ensure_registry(cache_path: Path, url: str, refresh: bool = False) -> Path:
    """Download registry.parquet if not cached, or if refresh is True."""
    if cache_path.exists() and not refresh:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")

    logger.info(f"Downloading GeoTessera registry ({url}) ...")
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "aereo-search-tessera"})
    resp = urlopen(req, timeout=120)
    total = int(resp.headers.get("Content-Length", 0))

    with open(tmp_path, "wb") as f:
        downloaded = 0
        while chunk := resp.read(8 * 1024 * 1024):  # 8 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                logger.info(f"  {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB")

    tmp_path.rename(cache_path)
    logger.info(f"Registry cached at {cache_path}")
    return cache_path


def check_href_exists(href: str, timeout: float = 5.0) -> bool:
    """Check if the href (URL or local path) actually exists."""
    if href.startswith("http://") or href.startswith("https://"):
        from urllib.request import Request, urlopen
        try:
            req = Request(href, method="HEAD", headers={"User-Agent": "aereo-search-tessera"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False
    else:
        from urllib.parse import urlparse
        if href.startswith("file://"):
            path_str = urlparse(href).path
        else:
            path_str = href
        try:
            return Path(path_str).exists()
        except Exception:
            return False


class SearchTessera(SearchProvider):
    """Search provider for GeoTessera satellite embeddings."""

    supported_collections: Sequence[str] = ["geotessera"]
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    base_url: str = "https://dl2.geotessera.org"
    tessera_version: str = "v1"
    registry_path: Path | None = None
    refresh_registry: bool = False
    check_href: bool = False

    def __call__(self) -> GeoDataFrame[AssetSchema]:
        """Search GeoTessera registry and return AssetSchema-compliant GeoDataFrame.

        Returns:
            Validated GeoDataFrame containing matched tile metadata.
        """
        # Bounding box for region
        if self.intersects is not None:
            bbox = self.intersects.bounds
        else:
            bbox = (-180.0, -90.0, 180.0, 90.0)

        # Normalize datetimes
        q_start = self._normalize_datetime(self.start_datetime)
        q_end = self._normalize_datetime(self.end_datetime)

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
            return self.empty_result()

        # Resolve paths and options
        base_url = self.base_url
        tessera_version = self.tessera_version
        refresh_registry = self.refresh_registry

        # Handle registry path and bootstrapping
        if self.registry_path is not None:
            registry_path = self.registry_path.expanduser().resolve()
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
            return self.empty_result()

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
                    "crs": utm["crs"],
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
            return self.empty_result()

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        # Apply high-precision geometry intersection if a shape was provided
        if self.intersects is not None:
            gdf = gdf[gdf.intersects(self.intersects)]

        # Check if href exists if check_href is True
        if self.check_href and not gdf.empty:
            hrefs = gdf["href"].tolist()
            max_workers = min(32, len(hrefs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                exists_flags = list(executor.map(check_href_exists, hrefs))
            gdf = gdf[exists_flags]

        if gdf.empty:
            return self.empty_result()

        return cast(GeoDataFrame[AssetSchema], AssetSchema.validate(gdf))

    @staticmethod
    def _normalize_datetime(dt: datetime | None) -> datetime | None:
        """Ensure datetime is timezone-aware UTC."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

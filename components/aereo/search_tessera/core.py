"""Search provider for GeoTessera satellite embeddings.

Queries the registry parquet directly to locate tiles within the AOI and time range.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from aereo.interfaces import SearchProvider
from aereo.schemas import AssetSchema
from pandera.typing.geopandas import GeoDataFrame
from pyproj import Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from structlog import get_logger

logger = get_logger()

# Default public S3 bucket published by GeoTessera.
TESSERA_BASE_URL = "https://s3.us-west-2.amazonaws.com/tessera-embeddings"

# Directory name used for the default ``vultr`` variant on S3.
EMBEDDINGS_DIR_NAME = "global_0.1_degree_representation"

# Default dataset variant. The bare ``global_0.1_degree_representation`` dir on
# S3 corresponds to this variant; named variants get a ``.<name>`` suffix.
DEFAULT_VARIANT = "vultr"

# Spatial resolution of a GeoTessera tile in decimal degrees.
TESSERA_TILE_RESOLUTION = 0.1

# Half the tile resolution; used to expand a tile center into a bounding box
# or to widen a query bbox to include neighbouring tiles.
TESSERA_TILE_HALF_RESOLUTION = TESSERA_TILE_RESOLUTION / 2

# Inclusive year range currently covered by the public GeoTessera registry.
MIN_TESSERA_YEAR = 2017
MAX_TESSERA_YEAR = 2025

# Chunk size for downloading the registry parquet (8 MiB).
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

# Maximum parallel workers used when validating asset hrefs.
MAX_HREF_CHECK_WORKERS = 32

# User-Agent sent with HTTP requests to the GeoTessera CDN.
USER_AGENT = "aereo-search-tessera"


def _parse_dataset_version(spec: str) -> tuple[str, str]:
    """Parse a flexible dataset-version spec.

    Returns ``(s3_path_component, normalized_version)``. Accepts inputs like
    ``"v1"``, ``"1"``, ``"1.0"``, ``"v1.0"`` (all → ``("v1", "1.0")``) and
    ``"v1.1"``, ``"1.1"`` (→ ``("v1.1", "1.1")``). The S3 layout uses ``v1/``
    for the 1.0 series.

    Args:
        spec: Version string such as ``"v1"``, ``"1.0"``, ``"v1.1"``.

    Returns:
        Tuple of ``(path_component, normalized_version)``.
    """
    s = spec.strip()
    if s.startswith("v"):
        s = s[1:]
    parts = s.split(".")
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else "0"
    norm = f"{major}.{minor}"
    path = f"v{major}" if minor == "0" else f"v{major}.{minor}"
    return path, norm


def _variant_subdir(variant: str) -> str:
    """Map a variant name to its embeddings-dir name on S3.

    Args:
        variant: Dataset variant, e.g. ``"vultr"`` or ``"cambridge"``.

    Returns:
        Embeddings subdirectory name.
    """
    if variant == DEFAULT_VARIANT:
        return EMBEDDINGS_DIR_NAME
    return f"{EMBEDDINGS_DIR_NAME}.{variant}"


def _default_registry_cache_path(version_path: str) -> Path:
    """Return the default local cache path for a versioned manifest.

    Args:
        version_path: Version path component from :func:`_parse_dataset_version`.

    Returns:
        Cache path such as ``~/.cache/geotessera/v1.1/manifest.parquet``.
    """
    return Path.home() / ".cache" / "geotessera" / version_path / "manifest.parquet"


def tile_to_bounds(lon: float, lat: float) -> tuple[float, float, float, float]:
    """WGS84 bounds for a GeoTessera tile centered at (lon, lat).

    Args:
        lon: Tile center longitude.
        lat: Tile center latitude.

    Returns:
        Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
    """
    half = TESSERA_TILE_HALF_RESOLUTION
    return (lon - half, lat - half, lon + half, lat + half)


def tile_utm_info(tile_lon: float, tile_lat: float, pixel_size: float = 10.0) -> dict[str, Any]:
    """Compute UTM EPSG, pixel-aligned bbox, and shape for a tile.

    Verified identical to GeoTessera's from_origin(xmin, ymax, 10.0, 10.0).

    Args:
        tile_lon: Tile center longitude.
        tile_lat: Tile center latitude.
        pixel_size: Pixel size in meters. Defaults to 10.0.

    Returns:
        Dictionary with keys ``epsg``, ``crs``, ``utm_bbox``, and ``shape``.
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


def _available_columns(registry_path: Path) -> list[str]:
    """Return the column names available in a parquet file.

    Args:
        registry_path: Path to the parquet registry.

    Returns:
        List of column names in the parquet schema.
    """
    return pq.ParquetFile(registry_path).schema.names


def load_tiles_for_region(
    bbox: tuple[float, float, float, float],
    year: int,
    registry_path: Path,
) -> pd.DataFrame:
    """Query tiles from parquet with predicate pushdown (~40ms).

    Args:
        bbox: Region bounds as ``(min_lon, min_lat, max_lon, max_lat)``.
        year: Year to filter.
        registry_path: Path to the parquet registry.

    Returns:
        DataFrame of matching tile metadata with normalized ``file_size``.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    expansion = TESSERA_TILE_HALF_RESOLUTION

    available_columns = _available_columns(registry_path)
    desired_columns = ["lon", "lat", "year", "hash", "file_size", "grid_size"]
    columns = [c for c in desired_columns if c in available_columns]

    table = pq.read_table(
        registry_path,
        columns=columns,
        filters=[
            ("year", "=", year),
            ("lon", ">=", min_lon - expansion),
            ("lon", "<=", max_lon + expansion),
            ("lat", ">=", min_lat - expansion),
            ("lat", "<=", max_lat + expansion),
        ],
    )
    df = table.to_pandas()

    # New manifests use ``grid_size`` for the embedding file size; old ones used
    # ``file_size``. Expose ``file_size`` consistently for downstream metadata.
    if "file_size" not in df.columns and "grid_size" in df.columns:
        df["file_size"] = df["grid_size"]

    return df.drop_duplicates(subset=["year", "lon", "lat"])


def _ensure_registry(cache_path: Path, url: str, refresh: bool = False) -> Path:
    """Download registry parquet if not cached, or if refresh is True.

    Args:
        cache_path: Local path where the registry should be stored.
        url: Remote URL to download.
        refresh: Force re-download even if cached. Defaults to False.

    Returns:
        Path to the cached registry file.

    Raises:
        urllib.error.URLError: If the remote request fails.
    """
    if cache_path.exists() and not refresh:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")

    logger.info(f"Downloading GeoTessera registry ({url}) ...")

    req = Request(url, headers={"User-Agent": USER_AGENT})
    resp = urlopen(req, timeout=120)
    total = int(resp.headers.get("Content-Length", 0))

    with open(tmp_path, "wb") as f:
        downloaded = 0
        while chunk := resp.read(DOWNLOAD_CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                logger.info(
                    f"  {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB"
                )

    tmp_path.rename(cache_path)
    logger.info(f"Registry cached at {cache_path}")
    return cache_path


def check_href_exists(href: str, timeout: float = 5.0) -> bool:
    """Check if the href (URL or local path) actually exists.

    Args:
        href: URL, ``file://`` URL, or local filesystem path.
        timeout: Timeout in seconds for HTTP checks. Defaults to 5.0.

    Returns:
        True if the resource exists, False otherwise.

    Raises:
        Does not raise; all exceptions are swallowed and return False.
    """
    if href.startswith("http://") or href.startswith("https://"):
        try:
            req = Request(href, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False
    else:
        if href.startswith("file://"):
            path_str = urlparse(href).path
        else:
            path_str = href
        try:
            return Path(path_str).exists()
        except Exception:
            return False


class SearchTessera(SearchProvider):
    """Search provider for GeoTessera satellite embeddings.

    Queries the GeoTessera registry parquet for tiles intersecting a given
    geometry and time range, then returns an ``AssetSchema``-validated
    GeoDataFrame with per-tile metadata and asset hrefs.
    """

    supported_collections: Sequence[str] = ["geotessera"]
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    base_url: str = TESSERA_BASE_URL
    tessera_version: str = "v1"
    tessera_variant: str = DEFAULT_VARIANT
    registry_path: Path | None = None
    registry_filename: str = "manifest.parquet"
    refresh_registry: bool = False
    check_href: bool = False

    def _intersects_geometry(self) -> BaseGeometry | None:
        """Return the normalized AOI geometry, or None for a global search.

        ``SearchProvider`` validates ``intersects`` into a ``BaseGeometry``
        at construction time, but its declared type is wider. This accessor
        provides a typed view and raises if normalization failed.

        Returns:
            Normalized AOI geometry, or ``None`` if no spatial filter.

        Raises:
            TypeError: If ``intersects`` was not normalized to a geometry.
        """
        geom = self.intersects
        if geom is None:
            return None
        if not isinstance(geom, BaseGeometry):
            raise TypeError(
                f"Expected intersects to be normalized to BaseGeometry, got {type(geom)}"
            )
        return geom

    def _matching_years(
        self, q_start: datetime | None, q_end: datetime | None
    ) -> list[int]:
        """Return the years that overlap the query temporal range.

        Args:
            q_start: Inclusive query start datetime, or None for open start.
            q_end: Inclusive query end datetime, or None for open end.

        Returns:
            Years between ``MIN_TESSERA_YEAR`` and ``MAX_TESSERA_YEAR`` that
            intersect the query range.
        """
        years: list[int] = []
        for yr in range(MIN_TESSERA_YEAR, MAX_TESSERA_YEAR + 1):
            yr_start = datetime(yr, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            yr_end = datetime(yr, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            if q_start is not None and yr_end < q_start:
                continue
            if q_end is not None and yr_start > q_end:
                continue
            years.append(yr)
        return years

    def _resolve_registry(self, version_path: str) -> Path:
        """Resolve the registry parquet path, downloading the default if needed.

        Args:
            version_path: Version path component from :func:`_parse_dataset_version`.

        Returns:
            Path to the registry parquet file.

        Raises:
            FileNotFoundError: If an explicit ``registry_path`` does not exist.
        """
        if self.registry_path is not None:
            registry_path = self.registry_path.expanduser().resolve()
            if not registry_path.exists():
                raise FileNotFoundError(f"Registry file not found: {registry_path}")
            return registry_path

        registry_path = _default_registry_cache_path(version_path)
        url = f"{self.base_url}/{version_path}/{self.registry_filename}"
        _ensure_registry(registry_path, url, refresh=self.refresh_registry)
        return registry_path

    def _build_asset_rows(
        self,
        tiles_df: pd.DataFrame,
        version_path: str,
        version_norm: str,
        embeddings_subdir: str,
    ) -> list[dict[str, Any]]:
        """Build AssetSchema rows from tile metadata.

        Args:
            tiles_df: DataFrame of tile metadata from :func:`load_tiles_for_region`.
            version_path: Version path component used in asset hrefs.
            version_norm: Normalized version string stored in row metadata.
            embeddings_subdir: Variant-aware embeddings subdirectory.

        Returns:
            List of AssetSchema-compatible row dictionaries.
        """
        base_url = self.base_url
        variant = self.tessera_variant
        rows: list[dict[str, Any]] = []
        for raw in tiles_df.itertuples(index=False):
            tile = cast(Any, raw)
            lon = float(tile.lon)
            lat = float(tile.lat)
            year = int(tile.year)
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
                    "href": f"{base_url}/{version_path}/{embeddings_subdir}/{year}/{grid_name}/{grid_name}.npy",
                    "crs": utm["crs"],
                    "tile_lon": lon,
                    "tile_lat": lat,
                    "tile_year": year,
                    "tile_utm_crs": utm["crs"],
                    "tile_utm_epsg": utm["epsg"],
                    "tile_utm_bbox": utm["utm_bbox"],
                    "tile_shape": utm["shape"],
                    "tile_hash": getattr(tile, "hash", None),
                    "tile_file_size": getattr(tile, "file_size", None),
                    "tessera_version": version_norm,
                    "tessera_variant": variant,
                }
            )
        return rows

    def _filter_existing_hrefs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Drop rows whose asset href does not exist.

        Args:
            gdf: GeoDataFrame with an ``href`` column.

        Returns:
            GeoDataFrame containing only rows with reachable hrefs.
        """
        if gdf.empty:
            return gdf
        hrefs = gdf["href"].tolist()
        max_workers = min(MAX_HREF_CHECK_WORKERS, len(hrefs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            exists_flags = list(executor.map(check_href_exists, hrefs))
        return cast(gpd.GeoDataFrame, gdf[exists_flags])

    def __call__(self) -> GeoDataFrame[AssetSchema]:
        """Search GeoTessera registry and return AssetSchema-compliant GeoDataFrame.

        Returns:
            Validated GeoDataFrame containing matched tile metadata.

        Raises:
            FileNotFoundError: If an explicit ``registry_path`` does not exist.
            TypeError: If ``intersects`` could not be normalized to a geometry.
        """
        # Bounding box for region
        geom = self._intersects_geometry()
        if geom is not None:
            bbox = geom.bounds
        else:
            bbox = (-180.0, -90.0, 180.0, 90.0)

        # Normalize datetimes
        q_start = self._normalize_datetime(self.start_datetime)
        q_end = self._normalize_datetime(self.end_datetime)

        # Determine which years intersect the query range
        matching_years = self._matching_years(q_start, q_end)
        if not matching_years:
            return self.empty_result()

        # Resolve version, variant and paths
        version_path, version_norm = _parse_dataset_version(self.tessera_version)
        embeddings_subdir = _variant_subdir(self.tessera_variant)

        # Handle registry path and bootstrapping
        registry_path = self._resolve_registry(version_path)

        # Load tiles from parquet
        tiles_dfs: list[pd.DataFrame] = []
        for year in matching_years:
            df = load_tiles_for_region(bbox, year, registry_path)
            if not df.empty:
                tiles_dfs.append(df)

        if not tiles_dfs:
            return self.empty_result()

        tiles_df = pd.concat(tiles_dfs, ignore_index=True)
        rows = self._build_asset_rows(tiles_df, version_path, version_norm, embeddings_subdir)

        if not rows:
            return self.empty_result()

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        # Apply high-precision geometry intersection if a shape was provided
        if geom is not None:
            gdf = cast(gpd.GeoDataFrame, gdf[gdf.intersects(geom)])

        # Check if href exists if check_href is True
        if self.check_href and not gdf.empty:
            gdf = self._filter_existing_hrefs(gdf)

        if gdf.empty:
            return self.empty_result()

        return cast(GeoDataFrame[AssetSchema], AssetSchema.validate(cast(pd.DataFrame, gdf)))

    @staticmethod
    def _normalize_datetime(dt: datetime | None) -> datetime | None:
        """Ensure datetime is timezone-aware UTC.

        Args:
            dt: Datetime to normalize, or None.

        Returns:
            Timezone-aware UTC datetime, or None if input was None.
        """
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

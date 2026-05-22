"""Unit tests for the aer-search-tessera plugin."""

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aer.interfaces import AerProfile
from aer.schemas import AssetSchema
from shapely.geometry import box

from aer.search_tessera.core import (
    DEFAULT_REGISTRY_PATH,
    TesseraSearchPlugin,
    tile_to_bounds,
    tile_utm_info,
)


@pytest.fixture
def mock_registry_file(tmp_path: Path) -> Path:
    """Create a temporary mock registry parquet file."""
    # We define 3 mock tiles:
    # 1. London area, 2025
    # 2. London area, 2024
    # 3. Buenos Aires area, 2025
    df = pd.DataFrame(
        [
            {
                "lon": -0.05,
                "lat": 51.45,
                "year": 2025,
                "hash": "london2025hash",
                "file_size": 104857600,
            },
            {
                "lon": -0.05,
                "lat": 51.45,
                "year": 2024,
                "hash": "london2024hash",
                "file_size": 104857600,
            },
            {
                "lon": -58.45,
                "lat": -34.65,
                "year": 2025,
                "hash": "ba2025hash",
                "file_size": 104857600,
            },
        ]
    )
    table = pa.Table.from_pandas(df)
    file_path = tmp_path / "registry.parquet"
    pq.write_table(table, file_path)
    return file_path


@pytest.fixture
def plugin() -> TesseraSearchPlugin:
    return TesseraSearchPlugin()


@pytest.fixture
def profile_tessera() -> AerProfile:
    return AerProfile(
        name="tessera_test",
        resolution=10.0,
        collections={"geotessera": ["all"]},
        plugin_hints={"search": "tessera"},
    )


def test_tile_to_bounds() -> None:
    """Verify coordinate bounding box math."""
    # Center at (0, 0)
    assert tile_to_bounds(0.0, 0.0) == pytest.approx((-0.05, -0.05, 0.05, 0.05))
    # Center at (-0.05, 51.45)
    assert tile_to_bounds(-0.05, 51.45) == pytest.approx((-0.1, 51.4, 0.0, 51.5))


def test_tile_utm_info() -> None:
    """Verify UTM zone selection and CRS selection."""
    # London is in UTM zone 30 (lon between -6 and 0)
    info_london = tile_utm_info(-0.05, 51.45)
    assert info_london["epsg"] == 32630
    assert info_london["crs"] == "EPSG:32630"

    # Southern hemisphere, zone 21 (lon between -60 and -54)
    info_ba = tile_utm_info(-58.45, -34.65)
    assert info_ba["epsg"] == 32721
    assert info_ba["crs"] == "EPSG:32721"


def test_tile_utm_info_pixel_alignment() -> None:
    """Verify UTM bbox width/height matches shape * pixel_size."""
    info = tile_utm_info(-0.05, 51.45)
    xmin, ymin, xmax, ymax = info["utm_bbox"]
    h, w = info["shape"]
    assert abs((xmax - xmin) - w * 10.0) < 1e-5
    assert abs((ymax - ymin) - h * 10.0) < 1e-5


def test_search_empty_region(
    plugin: TesseraSearchPlugin, profile_tessera: AerProfile, mock_registry_file: Path
) -> None:
    """Verify search in a region with no tiles returns empty GeoDataFrame with correct schema."""
    # Search somewhere in the ocean with no tiles in mock registry
    intersects = box(10.0, 10.0, 10.1, 10.1)
    result = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        search_params={"registry_path": mock_registry_file},
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
    validated = AssetSchema.validate(result)
    assert len(validated) == 0


def test_search_schema_validation(
    plugin: TesseraSearchPlugin, profile_tessera: AerProfile, mock_registry_file: Path
) -> None:
    """Verify search result passes AssetSchema validation."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result) == 1
    # Check that it validates
    validated = AssetSchema.validate(result)
    assert len(validated) == 1


def test_search_href_is_real_url(
    plugin: TesseraSearchPlugin, profile_tessera: AerProfile, mock_registry_file: Path
) -> None:
    """Verify that href URLs are generated correctly."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result) == 1
    href = result.iloc[0]["href"]
    assert href.startswith("https://dl2.geotessera.org/v1/global_0.1_degree_representation/2025/grid_-0.05_51.45/")
    assert href.endswith("grid_-0.05_51.45.npy")


def test_search_temporal_filter(
    plugin: TesseraSearchPlugin, profile_tessera: AerProfile, mock_registry_file: Path
) -> None:
    """Verify temporal range filtering works correctly."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)

    # Query 2025
    result_2025 = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_2025) == 1
    assert result_2025.iloc[0]["tile_year"] == 2025

    # Query 2024
    result_2024 = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        start_datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2024, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_2024) == 1
    assert result_2024.iloc[0]["tile_year"] == 2024


def test_search_spatial_filter(
    plugin: TesseraSearchPlugin, profile_tessera: AerProfile, mock_registry_file: Path
) -> None:
    """Verify spatial bounding box filtering works correctly."""
    # Search London
    intersects_london = box(-0.1, 51.4, 0.0, 51.5)
    result_london = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects_london,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_london) == 1
    assert result_london.iloc[0]["id"] == "grid_-0.05_51.45_2025"

    # Search Buenos Aires
    intersects_ba = box(-58.5, -34.7, -58.4, -34.6)
    result_ba = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects_ba,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_ba) == 1
    assert result_ba.iloc[0]["id"] == "grid_-58.45_-34.65_2025"


def test_search_params_override(
    plugin: TesseraSearchPlugin, profile_tessera: AerProfile, mock_registry_file: Path
) -> None:
    """Verify that override params are respected."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        search_params={
            "registry_path": mock_registry_file,
            "base_url": "https://my-custom-cdn.org",
            "tessera_version": "v99",
        },
    )
    assert len(result) == 1
    href = result.iloc[0]["href"]
    assert href.startswith("https://my-custom-cdn.org/v99/global_0.1_degree_representation/2025/")


def test_search_registry_path_not_exists(plugin: TesseraSearchPlugin, profile_tessera: AerProfile) -> None:
    """Verify that search raises FileNotFoundError if explicit registry_path is missing."""
    with pytest.raises(FileNotFoundError):
        plugin.search(
            profiles=[profile_tessera],
            search_params={"registry_path": "/path/does/not/exist/registry.parquet"},
        )


@pytest.mark.integration
def test_search_real_registry(plugin: TesseraSearchPlugin, profile_tessera: AerProfile) -> None:
    """Integration test using the real registry cached on system."""
    if not DEFAULT_REGISTRY_PATH.exists():
        pytest.skip("Real registry.parquet not found under default path.")

    # London bbox
    intersects = box(-0.2, 51.4, 0.1, 51.6)
    result = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
    )
    # The real registry should return some tiles for London area in 2025 (typically ~20)
    assert len(result) > 0
    validated = AssetSchema.validate(result)
    assert len(validated) == len(result)


@pytest.mark.integration
def test_download_from_href(plugin: TesseraSearchPlugin, profile_tessera: AerProfile, tmp_path: Path) -> None:
    """Integration test that downloads a tile from CDN to verify correctness."""
    if not DEFAULT_REGISTRY_PATH.exists():
        pytest.skip("Real registry.parquet not found under default path.")

    # Query a single tile in London
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = plugin.search(
        profiles=[profile_tessera],
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
    )
    assert len(result) > 0
    href = result.iloc[0]["href"]

    # Download it
    import urllib.request

    dest_file = tmp_path / "tile.npy"
    req = urllib.request.Request(href, headers={"User-Agent": "aer-search-tessera-test"})
    with urllib.request.urlopen(req, timeout=120) as response, open(dest_file, "wb") as out_file:
        out_file.write(response.read())

    # Load it
    arr = np.load(dest_file)
    assert arr.ndim == 3  # (H, W, 128)
    assert arr.shape[2] == 128

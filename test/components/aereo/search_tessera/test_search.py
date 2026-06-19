"""Unit tests for the aereo-search-tessera plugin."""

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aereo.schemas import AssetSchema
from shapely.geometry import box

from aereo.search_tessera.core import (
    SearchTessera,
    TESSERA_BASE_URL,
    _default_registry_cache_path,
    _parse_dataset_version,
    _variant_subdir,
    tile_to_bounds,
    tile_utm_info,
)


@pytest.fixture
def mock_registry_file_v1(tmp_path: Path) -> Path:
    """Create a temporary mock v1-style registry parquet file."""
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
def mock_registry_file_v1_1(tmp_path: Path) -> Path:
    """Create a temporary mock v1.1-style registry parquet file."""
    df = pd.DataFrame(
        [
            {
                "lon": -0.05,
                "lat": 51.45,
                "year": 2025,
                "grid_size": 104857600,
                "scales_size": 20971520,
                "version": "1.1",
                "variant": "cambridge",
            },
            {
                "lon": -58.45,
                "lat": -34.65,
                "year": 2025,
                "grid_size": 104857600,
                "scales_size": 20971520,
                "version": "1.1",
                "variant": "cambridge",
            },
        ]
    )
    table = pa.Table.from_pandas(df)
    file_path = tmp_path / "manifest.parquet"
    pq.write_table(table, file_path)
    return file_path


@pytest.fixture
def plugin() -> SearchTessera:
    return SearchTessera()


def test_parse_dataset_version() -> None:
    """Verify flexible version parsing."""
    assert _parse_dataset_version("v1") == ("v1", "1.0")
    assert _parse_dataset_version("1") == ("v1", "1.0")
    assert _parse_dataset_version("1.0") == ("v1", "1.0")
    assert _parse_dataset_version("v1.0") == ("v1", "1.0")
    assert _parse_dataset_version("v1.1") == ("v1.1", "1.1")
    assert _parse_dataset_version("1.1") == ("v1.1", "1.1")


def test_variant_subdir() -> None:
    """Verify variant-aware embeddings subdirectory names."""
    assert _variant_subdir("vultr") == "global_0.1_degree_representation"
    assert _variant_subdir("cambridge") == "global_0.1_degree_representation.cambridge"


def test_default_registry_cache_path_is_versioned() -> None:
    """Verify default cache path includes the version directory."""
    path = _default_registry_cache_path("v1.1")
    assert path.name == "manifest.parquet"
    assert path.parent.name == "v1.1"


def test_tile_to_bounds() -> None:
    """Verify coordinate bounding box math."""
    assert tile_to_bounds(0.0, 0.0) == pytest.approx((-0.05, -0.05, 0.05, 0.05))
    assert tile_to_bounds(-0.05, 51.45) == pytest.approx((-0.1, 51.4, 0.0, 51.5))


def test_tile_utm_info() -> None:
    """Verify UTM zone selection and CRS selection."""
    info_london = tile_utm_info(-0.05, 51.45)
    assert info_london["epsg"] == 32630
    assert info_london["crs"] == "EPSG:32630"

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


def test_search_empty_region(plugin: SearchTessera, mock_registry_file_v1: Path) -> None:
    """Verify search in a region with no tiles returns empty GeoDataFrame with correct schema."""
    intersects = box(10.0, 10.0, 10.1, 10.1)
    searcher = SearchTessera(intersects=intersects, registry_path=mock_registry_file_v1)
    result = searcher()
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
    validated = AssetSchema.validate(result)
    assert len(validated) == 0


def test_search_schema_validation(plugin: SearchTessera, mock_registry_file_v1: Path) -> None:
    """Verify search result passes AssetSchema validation."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    searcher = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        registry_path=mock_registry_file_v1,
    )
    result = searcher()
    assert len(result) == 1
    validated = AssetSchema.validate(result)
    assert len(validated) == 1
    assert result.iloc[0]["crs"] == "EPSG:32630"


def test_search_href_is_real_url(plugin: SearchTessera, mock_registry_file_v1: Path) -> None:
    """Verify that href URLs are generated correctly for the default v1 dataset."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    searcher = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        registry_path=mock_registry_file_v1,
    )
    result = searcher()
    assert len(result) == 1
    href = result.iloc[0]["href"]
    assert href.startswith(
        f"{TESSERA_BASE_URL}/v1/global_0.1_degree_representation/2025/grid_-0.05_51.45/"
    )
    assert href.endswith("grid_-0.05_51.45.npy")


def test_search_temporal_filter(plugin: SearchTessera, mock_registry_file_v1: Path) -> None:
    """Verify temporal range filtering works correctly."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)

    searcher_2025 = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        registry_path=mock_registry_file_v1,
    )
    result_2025 = searcher_2025()
    assert len(result_2025) == 1
    assert result_2025.iloc[0]["tile_year"] == 2025

    searcher_2024 = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2024, 12, 31, tzinfo=timezone.utc),
        registry_path=mock_registry_file_v1,
    )
    result_2024 = searcher_2024()
    assert len(result_2024) == 1
    assert result_2024.iloc[0]["tile_year"] == 2024


def test_search_spatial_filter(plugin: SearchTessera, mock_registry_file_v1: Path) -> None:
    """Verify spatial bounding box filtering works correctly."""
    intersects_london = box(-0.1, 51.4, 0.0, 51.5)
    searcher_london = SearchTessera(
        intersects=intersects_london,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        registry_path=mock_registry_file_v1,
    )
    result_london = searcher_london()
    assert len(result_london) == 1
    assert result_london.iloc[0]["id"] == "grid_-0.05_51.45_2025"

    intersects_ba = box(-58.5, -34.7, -58.4, -34.6)
    searcher_ba = SearchTessera(
        intersects=intersects_ba,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        registry_path=mock_registry_file_v1,
    )
    result_ba = searcher_ba()
    assert len(result_ba) == 1
    assert result_ba.iloc[0]["id"] == "grid_-58.45_-34.65_2025"
    assert result_ba.iloc[0]["crs"] == "EPSG:32721"


def test_search_params_override(mock_registry_file_v1: Path) -> None:
    """Verify that override params are respected."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    searcher = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        base_url="https://my-custom-cdn.org",
        tessera_version="v99",
        registry_path=mock_registry_file_v1,
    )
    result = searcher()
    assert len(result) == 1
    href = result.iloc[0]["href"]
    assert href.startswith("https://my-custom-cdn.org/v99/global_0.1_degree_representation/2025/")


def test_search_v1_1_with_cambridge_variant(mock_registry_file_v1_1: Path) -> None:
    """Verify search against a v1.1-style manifest uses the variant-aware path."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    searcher = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        tessera_version="v1.1",
        tessera_variant="cambridge",
        registry_path=mock_registry_file_v1_1,
    )
    result = searcher()
    assert len(result) == 1
    href = result.iloc[0]["href"]
    assert href.startswith(
        f"{TESSERA_BASE_URL}/v1.1/global_0.1_degree_representation.cambridge/2025/grid_-0.05_51.45/"
    )
    assert href.endswith("grid_-0.05_51.45.npy")
    assert result.iloc[0]["tessera_version"] == "1.1"
    assert result.iloc[0]["tessera_variant"] == "cambridge"
    assert result.iloc[0]["tile_file_size"] == 104857600
    assert pd.isna(result.iloc[0]["tile_hash"])


def test_search_v1_1_normalized_version(mock_registry_file_v1_1: Path) -> None:
    """Verify that ``1.1`` is normalized to the same path as ``v1.1``."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    searcher = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        tessera_version="1.1",
        tessera_variant="cambridge",
        registry_path=mock_registry_file_v1_1,
    )
    result = searcher()
    assert len(result) == 1
    assert result.iloc[0]["tessera_version"] == "1.1"
    assert "/v1.1/" in result.iloc[0]["href"]


def test_search_registry_path_not_exists() -> None:
    """Verify that search raises FileNotFoundError if explicit registry_path is missing."""
    searcher = SearchTessera(registry_path=Path("/path/does/not/exist/registry.parquet"))
    with pytest.raises(FileNotFoundError):
        searcher()


@pytest.mark.integration
def test_search_real_registry() -> None:
    """Integration test using the real registry cached on system."""
    if not _default_registry_cache_path("v1").exists():
        pytest.skip("Real registry.parquet not found under default path.")

    intersects = box(-0.2, 51.4, 0.1, 51.6)
    searcher = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
    )
    result = searcher()
    assert len(result) > 0
    validated = AssetSchema.validate(result)
    assert len(validated) == len(result)


@pytest.mark.integration
def test_download_from_href(tmp_path: Path) -> None:
    """Integration test that downloads a tile from CDN to verify correctness."""
    if not _default_registry_cache_path("v1").exists():
        pytest.skip("Real registry.parquet not found under default path.")

    intersects = box(-0.1, 51.4, 0.0, 51.5)
    searcher = SearchTessera(
        intersects=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
    )
    result = searcher()
    assert len(result) > 0
    href = result.iloc[0]["href"]

    import urllib.request

    dest_file = tmp_path / "tile.npy"
    req = urllib.request.Request(href, headers={"User-Agent": "aereo-search-tessera-test"})
    with urllib.request.urlopen(req, timeout=120) as response, open(dest_file, "wb") as out_file:
        out_file.write(response.read())

    arr = np.load(dest_file)
    assert arr.ndim == 3
    assert arr.shape[2] == 128

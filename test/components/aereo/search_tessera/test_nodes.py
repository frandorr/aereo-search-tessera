"""Unit tests for the function-based GeoTessera search nodes."""

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aereo.schemas import AssetSchema
from hamilton import driver
from shapely.geometry import box

from aereo.search_tessera.nodes import (
    _empty_result,
    _normalize_datetime,
    search_assets,
    search_results,
    supported_collections,
)


@pytest.fixture
def mock_registry_file(tmp_path: Path) -> Path:
    """Create a temporary mock registry parquet file."""
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


def test_supported_collections() -> None:
    assert supported_collections == ("geotessera",)


def test_empty_result() -> None:
    result = _empty_result()
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0
    validated = AssetSchema.validate(result)
    assert len(validated) == 0


def test_normalize_datetime_with_datetime() -> None:
    dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _normalize_datetime(dt) == dt


def test_normalize_datetime_naive() -> None:
    dt = datetime(2024, 6, 15, 12, 0, 0)
    result = _normalize_datetime(dt)
    assert result.tzinfo is not None
    assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_normalize_datetime_with_string() -> None:
    result = _normalize_datetime("2024-06-15T12:00:00Z")
    assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_normalize_datetime_none() -> None:
    assert _normalize_datetime(None) is None


def test_search_assets_empty_collections() -> None:
    """Explicit non-geotessera collections return empty."""
    result = search_assets(collections=["S3OLCI"])
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_search_assets_none_collections(mock_registry_file: Path) -> None:
    """None collections proceeds with search (plugin explicitly chosen)."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = search_assets(
        aoi=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1


def test_search_assets_no_results(mock_registry_file: Path) -> None:
    """Search in a region with no tiles returns empty GeoDataFrame."""
    intersects = box(10.0, 10.0, 10.1, 10.1)
    result = search_assets(
        aoi=intersects,
        search_params={"registry_path": mock_registry_file},
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_search_assets_returns_geodataframe(mock_registry_file: Path) -> None:
    """Verify search result passes AssetSchema validation."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = search_assets(
        aoi=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        collections=["geotessera"],
        search_params={"registry_path": mock_registry_file},
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert "collection" in result.columns
    assert result.iloc[0]["collection"] == "geotessera"
    validated = AssetSchema.validate(result)
    assert len(validated) == 1


def test_search_assets_href_generation(mock_registry_file: Path) -> None:
    """Verify that href URLs are generated correctly."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = search_assets(
        aoi=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result) == 1
    href = result.iloc[0]["href"]
    assert href.startswith(
        "https://dl2.geotessera.org/v1/global_0.1_degree_representation/2025/grid_-0.05_51.45/"
    )
    assert href.endswith("grid_-0.05_51.45.npy")


def test_search_assets_temporal_filter(mock_registry_file: Path) -> None:
    """Verify temporal range filtering works correctly."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)

    result_2025 = search_assets(
        aoi=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_2025) == 1
    assert result_2025.iloc[0]["tile_year"] == 2025

    result_2024 = search_assets(
        aoi=intersects,
        start_datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2024, 12, 31, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_2024) == 1
    assert result_2024.iloc[0]["tile_year"] == 2024


def test_search_assets_spatial_filter(mock_registry_file: Path) -> None:
    """Verify spatial bounding box filtering works correctly."""
    intersects_london = box(-0.1, 51.4, 0.0, 51.5)
    result_london = search_assets(
        aoi=intersects_london,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_london) == 1
    assert result_london.iloc[0]["id"] == "grid_-0.05_51.45_2025"

    intersects_ba = box(-58.5, -34.7, -58.4, -34.6)
    result_ba = search_assets(
        aoi=intersects_ba,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        search_params={"registry_path": mock_registry_file},
    )
    assert len(result_ba) == 1
    assert result_ba.iloc[0]["id"] == "grid_-58.45_-34.65_2025"


def test_search_assets_params_override(mock_registry_file: Path) -> None:
    """Verify that override params are respected."""
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = search_assets(
        aoi=intersects,
        start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        search_params={
            "registry_path": mock_registry_file,
            "base_url": "https://my-custom-cdn.org",
            "tessera_version": "v99",
        },
    )
    assert len(result) == 1
    href = result.iloc[0]["href"]
    assert href.startswith(
        "https://my-custom-cdn.org/v99/global_0.1_degree_representation/2025/"
    )


def test_search_assets_registry_path_not_exists() -> None:
    """Verify that search raises FileNotFoundError if explicit registry_path is missing."""
    with pytest.raises(FileNotFoundError):
        search_assets(
            search_params={"registry_path": "/path/does/not/exist/registry.parquet"},
        )


def test_search_results_passthrough() -> None:
    """search_results is a passthrough of search_assets."""
    empty = _empty_result()
    assert search_results(empty) is empty


def test_search_pipeline_runs(mock_registry_file: Path) -> None:
    """Build a real Hamilton driver from the nodes module and execute search_results."""
    from aereo.search_tessera import nodes as search_module

    dr = driver.Builder().with_modules(search_module).build()
    intersects = box(-0.1, 51.4, 0.0, 51.5)
    result = dr.execute(
        ["search_results"],
        inputs={
            "aoi": intersects,
            "start_datetime": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "end_datetime": datetime(2025, 12, 31, tzinfo=timezone.utc),
            "search_params": {"registry_path": mock_registry_file},
        },
    )
    assert "search_results" in result
    gdf = result["search_results"]
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.iloc[0]["collection"] == "geotessera"

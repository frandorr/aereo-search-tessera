"""Search implementation for GeoTessera satellite embeddings.

Provides both the legacy class-based API (:class:`TesseraSearchPlugin`)
and the new function-based Hamilton nodes.
"""

from aereo.search_tessera.core import TesseraSearchPlugin
from aereo.search_tessera.nodes import (
    search_assets,
    search_results,
    supported_collections,
)

__all__ = [
    "TesseraSearchPlugin",
    "search_assets",
    "search_results",
    "supported_collections",
]

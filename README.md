# aereo-search-tessera

An `aereo` ecosystem **Search** plugin for [GeoTessera](https://geotessera.io) satellite embeddings.

This plugin queries GeoTessera's parquet registries directly to discover embedding tiles that intersect a given area of interest and time range. The returned assets can then be read by the companion reader plugin, `aereo-read-tessera`, or any other AEREO reader.

Supports dataset versions ``v1`` / ``1.0`` and ``v1.1`` / ``1.1``, and dataset variants such as ``vultr`` and ``cambridge``.

---

## Installation

Add the plugin to your AEREO project with `uv`:

```bash
uv add aereo-search-tessera
```

Or with `pip`:

```bash
pip install aereo-search-tessera
```

Once installed, `aereo` automatically discovers the `search_tessera` plugin through Python entry points.

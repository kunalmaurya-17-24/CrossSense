"""Small, reproducible metadata-only searches of the official CDSE catalogue."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

LOG = logging.getLogger(__name__)
ODATA_PRODUCTS = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
INDIA_BBOX = (73.80, 18.45, 73.90, 18.55)


def catalogue_params(
    sensor: str,
    bbox: tuple[float, float, float, float] = INDIA_BBOX,
    start: str = "2024-01-01T00:00:00Z",
    end: str = "2024-01-16T00:00:00Z",
    limit: int = 4,
) -> dict[str, str | int]:
    """Construct a bounded AOI/time/product query; dates use [start, end)."""
    if sensor not in {"sar", "optical"}:
        raise ValueError("sensor must be sar or optical")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    minx, miny, maxx, maxy = bbox
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError("Invalid WGS84 bbox")
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    if start_dt.tzinfo is None or end_dt.tzinfo is None or start_dt >= end_dt:
        raise ValueError("Acquisition dates must be timezone-aware with start before end")
    collection = "SENTINEL-1" if sensor == "sar" else "SENTINEL-2"
    polygon = f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))"
    clauses = [f"Collection/Name eq '{collection}'",
               f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
               f"ContentDate/Start ge {start_dt.isoformat().replace('+00:00', 'Z')}",
               f"ContentDate/Start lt {end_dt.isoformat().replace('+00:00', 'Z')}"]
    if sensor == "sar":
        # Dual VV/VH IW GRD product name; no RAW/SLC or single-pol substitution.
        clauses += ["contains(Name,'_IW_GRDH_1SDV_')"]
    else:
        clauses += ["Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/OData.CSC.StringAttribute/Value eq 'S2MSI2A')",
                    "Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and a/OData.CSC.DoubleAttribute/Value le 20)"]
    return {"$filter": " and ".join(clauses), "$expand": "Attributes", "$top": limit,
            "$orderby": "ContentDate/Start asc"}


def normalize_product(product: dict[str, Any], sensor: str) -> dict[str, Any]:
    """Retain retrieval-relevant metadata and explicitly mark absent fields."""
    attrs = {item["Name"]: item.get("Value") for item in product.get("Attributes", [])}
    name = product["Name"]
    tile = re.search(r"_T(\d{2}[A-Z]{3})(?:_|\.)", name)
    tile_id = attrs.get("tileId") or (tile.group(1) if tile else None)
    # Product footprints are WGS84 metadata; delivered raster CRS must be read
    # from the local raster, not inferred from a footprint or tile name alone.
    return {
        "product_id": product["Id"], "name": name, "sensor": sensor,
        "acquisition_start": product.get("ContentDate", {}).get("Start"),
        "acquisition_end": product.get("ContentDate", {}).get("End"),
        "orbit_direction": attrs.get("orbitDirection"),
        "relative_orbit_number": attrs.get("relativeOrbitNumber"),
        "orbit_number": attrs.get("orbitNumber"), "tile_id": tile_id,
        "processing_level": attrs.get("processingLevel") or ("GRD" if sensor == "sar" else "L2A"),
        "processing_baseline": attrs.get("processingBaseline"),
        "cloud_cover_percent": attrs.get("cloudCover"),
        "polarisation_channels": attrs.get("polarisationChannels"),
        "footprint": product.get("GeoFootprint"), "footprint_crs": "EPSG:4326",
        "raster_crs": None, "raster_crs_status": "inspect after local preparation",
        "content_length_bytes": product.get("ContentLength"),
        "checksums": product.get("Checksum", []), "online": product.get("Online"),
        "source": "Copernicus Data Space Ecosystem",
        "license_url": "https://dataspace.copernicus.eu/terms-and-conditions",
        "attribution": "Contains Copernicus Sentinel data (2024); modified by CrossSense after preparation",
        "attributes": attrs,
    }


def query_india_catalogue(
    output_path: str | Path,
    *,
    bbox: tuple[float, float, float, float] = INDIA_BBOX,
    start: str = "2024-01-01T00:00:00Z",
    end: str = "2024-01-16T00:00:00Z",
    limit: int = 4,
    timeout_s: float = 20,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Save at most four products/sensor by default; no raster downloads.

    This is a candidate list, not a paired or evaluated dataset. Each failed
    request is recorded once without retries. Unknown metadata remains null.
    """
    # Validate both calls before making any request or writing a success record.
    queries = {sensor: catalogue_params(sensor, bbox, start, end, limit) for sensor in ("sar", "optical")}
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    client = session or requests.Session()
    result: dict[str, Any] = {
        "source": "Copernicus Data Space Ecosystem OData", "endpoint": ODATA_PRODUCTS,
        "queried_at": datetime.now(UTC).isoformat(), "aoi_name": "Pune, India",
        "bbox_wgs84": list(bbox), "start_inclusive": start, "end_exclusive": end,
        "target_grid_crs": "EPSG:32643", "target_resolution_m": 10,
        "metadata_only": True, "raster_downloaded": False, "products": [], "queries": [],
    }
    try:
        for sensor, params in queries.items():
            query = {"sensor": sensor, "params": params, "status": "started"}
            try:
                LOG.info("CDSE metadata query sensor=%s bbox=%s interval=%s/%s limit=%d", sensor, bbox, start, end, limit)
                with client.get(ODATA_PRODUCTS, params=params, timeout=timeout_s) as response:
                    response.raise_for_status()
                    products = response.json().get("value", [])
                result["products"].extend(normalize_product(product, sensor) for product in products[:limit])
                query.update(status="ok", count=min(len(products), limit))
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                query.update(status="failed", error=f"HTTP {status}; no retry")
                LOG.error("CDSE metadata query sensor=%s failed HTTP=%s", sensor, status)
            except (requests.RequestException, ValueError, KeyError) as exc:
                query.update(status="failed", error=f"Metadata request/response failure ({type(exc).__name__}); no retry")
                LOG.error("CDSE metadata query sensor=%s failed type=%s", sensor, type(exc).__name__)
            result["queries"].append(query)
    finally:
        if session is None:
            client.close()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

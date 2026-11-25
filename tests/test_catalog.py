import json

import pytest

from crosssense.data.catalog import catalogue_params, normalize_product, query_india_catalogue


def test_catalogue_is_bounded_and_source_specific():
    sar = catalogue_params("sar")
    optical = catalogue_params("optical")
    assert sar["$top"] == 4
    assert "_IW_GRDH_1SDV_" in sar["$filter"]
    assert "SRID=4326" in sar["$filter"]
    assert "S2MSI2A" in optical["$filter"]
    assert "cloudCover" in optical["$filter"]


@pytest.mark.parametrize("kwargs", [{"sensor": "bad"}, {"sensor": "sar", "limit": 100},
                                        {"sensor": "sar", "bbox": (5, 5, 1, 1)},
                                        {"sensor": "sar", "start": "2024-01-01", "end": "2024-02-01"}])
def test_invalid_catalogue_queries(kwargs):
    with pytest.raises(ValueError):
        catalogue_params(**kwargs)


def test_product_metadata_does_not_invent_raster_projection():
    product = normalize_product({"Id": "example", "Name": "S2A_MSIL2A_20240101T000000_N0510_R000_T43QDA_20240101T000000.SAFE",
                                 "Attributes": [{"Name": "cloudCover", "Value": 2.5}],
                                 "ContentDate": {"Start": "2024-01-01T00:00:00Z"}}, "optical")
    assert product["tile_id"] == "43QDA"
    assert product["raster_crs"] is None
    assert product["orbit_direction"] is None
    assert product["cloud_cover_percent"] == 2.5


def test_metadata_query_writes_failures_once(tmp_path):
    import requests

    class Client:
        calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            raise requests.ConnectionError("token=secret")

    client = Client()
    output = tmp_path / "catalogue.json"
    result = query_india_catalogue(output, session=client)
    assert client.calls == 2
    assert len(result["queries"]) == 2
    assert all(query["status"] == "failed" for query in result["queries"])
    assert result["raster_downloaded"] is False
    assert "secret" not in output.read_text()
    assert json.loads(output.read_text())["metadata_only"] is True

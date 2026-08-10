import polars as pl
import pytest

from nyc_demand.data.zones import normalize_zone_lookup, validate_zone_lookup


def _lookup() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "LocationID": [1, 2],
            "Borough": ["EWR", "Queens"],
            "Zone": ["Newark Airport", "Jamaica Bay"],
            "service_zone": ["EWR", "Boro Zone"],
        }
    )


def test_zone_lookup_is_normalized_to_model_contract() -> None:
    result = normalize_zone_lookup(_lookup())

    assert result.columns == ["zone_id", "borough", "zone_name", "service_zone"]
    assert result["zone_id"].to_list() == [1, 2]
    assert result["borough"].to_list() == ["EWR", "Queens"]


def test_zone_lookup_rejects_duplicate_location_ids() -> None:
    frame = pl.concat([_lookup(), _lookup().head(1)])

    with pytest.raises(ValueError, match="duplicate LocationID"):
        validate_zone_lookup(frame)


def test_zone_lookup_rejects_missing_schema() -> None:
    with pytest.raises(ValueError, match="Missing taxi-zone columns"):
        validate_zone_lookup(pl.DataFrame({"LocationID": [1]}))

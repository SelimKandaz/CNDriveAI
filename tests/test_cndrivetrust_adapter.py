import pytest

from cndriveai.cndrivetrust import adapt_artifact


def test_health_summary_preserves_zero_and_strips_identifiers_and_raw_data():
    result = adapt_artifact(
        {
            "artifact_type": "CNDRIVETRUST_HEALTH_USAGE_SUMMARY",
            "schema_version": "2.1",
            "workflow_version": "2.3.0",
            "run_id": "RUN-SYNTHETIC-01",
            "observed_utc": "2026-01-01T00:00:00Z",
            "target": {
                "model": "Synthetic SSD",
                "serial": "SYNTHETIC-SECRET",
                "device_path": "/dev/fake",
            },
            "current_values": {
                "media_errors": {
                    "value": 0,
                    "status": "0",
                    "source": "synthetic",
                    "time": "t0",
                    "unit": "count",
                },
                "bad": "not-an-envelope",
            },
            "dimensions": {
                "media_health": {"state": "GOOD"},
                "history_trust": {"state": "PARTIAL"},
            },
            "overall": "PARTIAL",
            "notes": "private operator note",
            "raw": {"stdout": "private raw output"},
            "assessment_limits": ["Synthetic limit"],
        }
    ).context
    assert result["evidence_fields"]["media_errors"]["value"] == 0
    assert result["evidence_fields"]["media_errors"]["status"] == "0"
    assert result["target_context"] == {"model": "Synthetic SSD"}
    assert "serial" not in str(result)
    assert "device_path" not in str(result)
    assert "private operator note" not in str(result)
    assert "private raw output" not in str(result)
    assert set(result["health_dimensions"]) == {"media_health", "history_trust"}


def test_evidence_artifact_keeps_query_failure_and_drops_raw_command_records():
    result = adapt_artifact(
        {
            "artifact_type": "CNDRIVETRUST_SSD_NVME_EVIDENCE",
            "schema_version": "2.0",
            "run_id": "RUN-SYNTHETIC-02",
            "values": {
                "temperature": {
                    "value": None,
                    "status": "QUERY_FAILED",
                    "source": "synthetic",
                    "time": "t1",
                }
            },
            "analysis": {"disposition": "INSUFFICIENT_EVIDENCE", "findings": []},
            "raw_records": {"stderr": "ignored"},
        }
    ).context
    assert result["evidence_fields"]["temperature"]["status"] == "QUERY_FAILED"
    assert "raw_records" not in result


def test_rejects_unknown_artifact_type():
    with pytest.raises(ValueError):
        adapt_artifact({"artifact_type": "UNKNOWN"})

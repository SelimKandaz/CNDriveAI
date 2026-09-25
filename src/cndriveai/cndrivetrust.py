"""Read-only normalizer for the public CNDriveTrust v2.3 artifact contract.

This module does not import CNDriveTrust code, execute commands, write to its
result tree, or determine/override a product verdict. It emits a privacy-minimal
ephemeral context package for an optional advisory model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_ARTIFACTS = {
    "CNDRIVETRUST_SSD_NVME_EVIDENCE",
    "CNDRIVETRUST_HEALTH_USAGE_SUMMARY",
}
VALUE_ENVELOPE_KEYS = {"value", "status", "source", "time", "unit"}
HEALTH_DIMENSIONS = {
    "media_health",
    "endurance",
    "error_state",
    "usage",
    "performance",
    "history_trust",
}
PRIVATE_KEYS = {
    "serial",
    "serial_number",
    "target_serial",
    "serial_directory",
    "po_batch",
    "batch_id",
    "device_path",
    "device_paths",
    "path",
    "host",
    "hostname",
    "notes",
    "mac",
    "device_id",
    "namespace_id",
    "namespace_nguid",
    "namespace_eui64",
    "account_id",
    "token",
    "authorization",
    "credential",
    "raw",
    "stdout",
    "stderr",
}


@dataclass(frozen=True)
class AdaptedRun:
    """Model-facing data only; never an authority or execution request."""

    context: dict[str, Any]

    def to_prompt_json(self) -> str:
        import json

        return json.dumps(self.context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sanitize(value: Any, key: str = "") -> Any:
    folded = key.casefold()
    if (
        folded in PRIVATE_KEYS
        or folded.endswith("_path")
        or folded.endswith("_paths")
        or "serial" in folded
        or any(part in folded for part in ("password", "secret", "token"))
    ):
        return None
    if isinstance(value, dict):
        return {
            str(child_key): clean
            for child_key, child_value in value.items()
            if (clean := _sanitize(child_value, str(child_key))) is not None
        }
    if isinstance(value, list):
        return [clean for item in value if (clean := _sanitize(item)) is not None]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalized_values(values: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(values, dict):
        raise ValueError("normalized values must be an object")
    normalized: dict[str, dict[str, Any]] = {}
    for name, item in values.items():
        if not isinstance(item, dict):
            continue
        # Preserve explicit zero and the source/status/time envelope exactly.
        field_name = str(name).casefold()
        if (
            field_name in PRIVATE_KEYS
            or "serial" in field_name
            or field_name.endswith("_path")
            or field_name.endswith("_paths")
        ):
            continue
        normalized[str(name)] = {
            key: _sanitize(item.get(key), key) for key in VALUE_ENVELOPE_KEYS if key in item
        }
    return normalized


def adapt_artifact(artifact: dict[str, Any]) -> AdaptedRun:
    """Convert one normalized evidence or health artifact to advisory context.

    Raw command output, operator notes, batch/serial identifiers, local paths,
    and network/credential material are excluded. Caller must supply the
    normalized artifact from an approved local handoff location.
    """
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a JSON object")
    artifact_type = artifact.get("artifact_type")
    if artifact_type not in SUPPORTED_ARTIFACTS:
        raise ValueError(f"unsupported artifact_type: {artifact_type!r}")

    result: dict[str, Any] = {
        "context_contract_version": "1.0",
        "source_system": "CNDriveTrust",
        "artifact_type": artifact_type,
        "schema_version": artifact.get("schema_version"),
        "workflow_version": artifact.get("workflow_version"),
        # The model does not need customer/run identifiers to reason about a
        # single supplied record. The host can retain the true ID out of band.
        "run_reference": "current supplied run",
        "observed_utc": artifact.get("observed_utc") or artifact.get("completed_utc"),
        "authority": "advisory_only",
        "network_access_required": False,
        "raw_evidence_included": False,
    }

    target = artifact.get("target")
    if isinstance(target, dict):
        allowed_target = {
            name: target[name]
            for name in ("manufacturer", "model", "firmware", "transport", "capacity_bytes")
            if target.get(name) is not None
        }
        result["target_context"] = _sanitize(allowed_target)

    if artifact_type == "CNDRIVETRUST_SSD_NVME_EVIDENCE":
        values = artifact.get("values")
        result["evidence_fields"] = _normalized_values(values)
        result["analysis"] = _sanitize(artifact.get("analysis", {}))
        result["read_verification"] = _sanitize(artifact.get("read_verification", {}))
        result["command_status"] = _sanitize(artifact.get("command_status", {}))
        result["vendor_specific"] = _sanitize(artifact.get("vendor_specific", {}))
    else:
        result["evidence_fields"] = _normalized_values(artifact.get("current_values", {}))
        dimensions = artifact.get("dimensions", {})
        if not isinstance(dimensions, dict):
            raise ValueError("health summary dimensions must be an object")
        result["health_dimensions"] = {
            name: _sanitize(dimensions[name]) for name in HEALTH_DIMENSIONS if name in dimensions
        }
        result["overall"] = artifact.get("overall")
        result["assessment_limits"] = _sanitize(artifact.get("assessment_limits", []))

    result["context_priority"] = [
        "this current run's normalized evidence",
        "current product rules/config supplied separately by the host",
        "local project documentation",
        "historical runs explicitly supplied for comparison",
        "vendor/reference documentation",
        "general pretrained knowledge",
    ]
    result["safety_notice"] = (
        "Do not replace product verdicts, select targets, execute commands, "
        "or authorize erase, sanitize, format, firmware, or deployment actions."
    )
    return AdaptedRun(result)

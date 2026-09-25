import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def test_all_json_schemas_are_valid_and_samples_validate():
    schema_files = list((ROOT / "schemas").glob("*.schema.json"))
    assert len(schema_files) == 9
    schemas = {
        path.stem.removesuffix(".schema"): json.loads(path.read_text(encoding="utf-8"))
        for path in schema_files
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    samples = [
        ("system_context", ROOT / "examples/context/synthetic_system_context.json"),
        ("current_run", ROOT / "examples/context/synthetic_health_summary.json"),
    ]
    for key, path in samples:
        Draft202012Validator(schemas[key]).validate(json.loads(path.read_text(encoding="utf-8")))

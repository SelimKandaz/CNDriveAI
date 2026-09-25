"""Local allowlist-style privacy, secret, license and large-file preflight."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".local_runs",
    "build",
    "dist",
}
SECRET_PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_secret_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{20,}"
    ),
}
PRIVATE_PATHS = [
    "".join(("C:", "\\", "Users", "\\", "selim", "\\")),
    "/".join(("", "home", "selim")) + "/",
    "/".join(("", "opt", "cngpu-drive-evidence")),
]
PRIVATE_NETWORK = re.compile(r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3})\b")


def _tracked_or_source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.is_file() and path.name != "publication_audit.json":
            files.append(path)
    return files


def _private_path_hits(text: str) -> list[str]:
    normalized = text.casefold()
    return [value for value in PRIVATE_PATHS if value.casefold() in normalized]


def run_audit(max_bytes: int) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    files = _tracked_or_source_files()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        if size > max_bytes:
            findings.append({"type": "oversized_file", "path": relative, "detail": str(size)})
        if path.suffix.lower() in {".pyc", ".gguf", ".safetensors", ".pt", ".pth", ".bin"}:
            findings.append(
                {"type": "forbidden_artifact_extension", "path": relative, "detail": path.suffix}
            )
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"type": "secret_pattern", "path": relative, "detail": label})
        for _value in _private_path_hits(text):
            findings.append(
                {"type": "machine_or_company_path", "path": relative, "detail": "path pattern"}
            )
        if PRIVATE_NETWORK.search(text):
            findings.append(
                {"type": "private_ipv4", "path": relative, "detail": "private address pattern"}
            )
    if not (ROOT / "LICENSE").is_file() or "Apache License" not in (ROOT / "LICENSE").read_text(
        encoding="utf-8"
    ):
        findings.append(
            {"type": "license_missing", "path": "LICENSE", "detail": "Apache-2.0 text missing"}
        )
    try:
        files_changed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        files_changed = []
    return {
        "audit_version": "1.0",
        "files_scanned": len(files),
        "max_file_bytes": max_bytes,
        "findings": findings,
        "git_status_lines": len(files_changed),
        "result": "PASS" if not findings else "FAIL",
        "scope_note": (
            "Static heuristic preflight only; review staged diff and "
            "upstream licenses before publishing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-mib", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "publication_audit.json")
    args = parser.parse_args()
    result = run_audit(args.max_file_mib * 1024 * 1024)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

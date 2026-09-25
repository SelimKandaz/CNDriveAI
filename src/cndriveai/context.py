"""Generic context contracts and deterministic source-priority resolution.

The release-specific normalization of CNDriveTrust v2.3 evidence artifacts is
implemented in :mod:`cndriveai.cndrivetrust`; this module remains product-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class ContextPriority(IntEnum):
    CURRENT_RUN = 0
    PRODUCTION_RULES = 1
    PROJECT_DOCUMENTATION = 2
    HISTORICAL_RUN = 3
    VENDOR_REFERENCE = 4
    GENERAL_KNOWLEDGE = 5


CONTEXT_PRIORITY_LABELS = {
    ContextPriority.CURRENT_RUN: "current_run_evidence",
    ContextPriority.PRODUCTION_RULES: "current_production_rules",
    ContextPriority.PROJECT_DOCUMENTATION: "local_project_documentation",
    ContextPriority.HISTORICAL_RUN: "historical_run",
    ContextPriority.VENDOR_REFERENCE: "vendor_reference",
    ContextPriority.GENERAL_KNOWLEDGE: "general_pretrained_knowledge",
}


@dataclass(frozen=True)
class ContextFact:
    fact_id: str
    kind: str
    value: Any
    source: str
    priority: ContextPriority
    observed_at: str | None = None
    status: str = "observed"
    sensitive: bool = False


def resolve_facts(facts: list[ContextFact]) -> list[ContextFact]:
    """Keep conflicting facts visible, ordered by authority then recency."""
    seen_ids: set[str] = set()
    for fact in facts:
        if not fact.fact_id or not fact.kind or not fact.source:
            raise ValueError("fact_id, kind, and source are required")
        if fact.fact_id in seen_ids:
            raise ValueError(f"duplicate fact_id: {fact.fact_id}")
        seen_ids.add(fact.fact_id)
    return sorted(
        facts,
        key=lambda fact: (fact.priority, fact.observed_at or "", fact.fact_id),
        reverse=False,
    )


def context_packet(facts: list[ContextFact]) -> dict[str, Any]:
    ordered = resolve_facts(facts)
    return {
        "context_version": "1.0",
        "facts": [
            {
                "fact_id": fact.fact_id,
                "kind": fact.kind,
                "value": fact.value,
                "source": fact.source,
                "priority": CONTEXT_PRIORITY_LABELS[fact.priority],
                "observed_at": fact.observed_at,
                "status": fact.status,
                "sensitive": fact.sensitive,
            }
            for fact in ordered
        ],
        "policy": "Higher-authority supplied evidence wins; conflicts remain explicit.",
    }

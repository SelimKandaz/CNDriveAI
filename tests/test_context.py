from cndriveai.context import ContextFact, ContextPriority, context_packet


def test_priority_keeps_current_evidence_ahead_of_knowledge():
    packet = context_packet(
        [
            ContextFact("base", "counter", 0, "model", ContextPriority.GENERAL_KNOWLEDGE),
            ContextFact("run", "counter", None, "run", ContextPriority.CURRENT_RUN),
        ]
    )
    assert [item["fact_id"] for item in packet["facts"]] == ["run", "base"]


def test_conflicting_facts_remain_visible():
    packet = context_packet(
        [
            ContextFact("a", "state", "FAIL", "current", ContextPriority.CURRENT_RUN),
            ContextFact("b", "state", "PASS", "history", ContextPriority.HISTORICAL_RUN),
        ]
    )
    assert [item["value"] for item in packet["facts"]] == ["FAIL", "PASS"]

import pytest

from cndriveai.dataset import heldout_records
from scripts.compare_evaluations import build_comparison


def _row(task, answer, truncated=False):
    return {
        "id": task["id"],
        "category": task["category"],
        "answer": answer,
        "scores": {"truncated_output": truncated, "latency_seconds": 1.0},
    }


def test_comparison_uses_only_paired_nontruncated_tasks():
    tasks = heldout_records()[:2]
    base = {
        task["id"]: _row(task, "unknown", truncated=index == 0) for index, task in enumerate(tasks)
    }
    adapter = {task["id"]: _row(task, "unknown not established") for task in tasks}

    result = build_comparison(base, adapter, {task["id"]: task for task in tasks})

    paired = result["paired_complete_comparison"]
    assert paired["paired_complete_task_count"] == 1
    assert paired["incorrect_task_changes"]["improved"]["count"] == 0
    assert result["base"]["truncated_count"] == 1
    assert result["adapter"]["truncated_count"] == 0


def test_comparison_rejects_different_task_sets():
    tasks = heldout_records()[:2]
    base = {task["id"]: _row(task, "unknown") for task in tasks}
    adapter = {tasks[0]["id"]: _row(tasks[0], "unknown")}

    with pytest.raises(ValueError, match="identical task IDs"):
        build_comparison(base, adapter, {task["id"]: task for task in tasks})

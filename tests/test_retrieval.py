from cndriveai.retrieval import LocalEvidenceIndex


def test_bm25_search_is_namespace_scoped(tmp_path):
    index = LocalEvidenceIndex(tmp_path / "evidence.sqlite3")
    index.upsert("doc-a", "project-a", "SMART query", "query failure means unknown", "synthetic")
    index.upsert("doc-b", "project-b", "SMART query", "query failure means unknown", "synthetic")
    hits = index.search("query failure", "project-a")
    assert [hit.doc_id for hit in hits] == ["doc-a"]


def test_empty_query_returns_no_hits(tmp_path):
    index = LocalEvidenceIndex(tmp_path / "evidence.sqlite3")
    assert index.search("!!!", "project-a") == []

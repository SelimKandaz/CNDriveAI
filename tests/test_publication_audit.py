from scripts.audit_publication import _private_path_hits


def test_audit_detects_owner_paths_without_matching_its_own_patterns():
    windows_path = "".join(("C:", "\\", "Users", "\\", "selim", "\\", "private"))
    linux_path = "/".join(("", "home", "selim", "cache"))
    assert _private_path_hits(f"local cache: {windows_path}")
    assert _private_path_hits(f"runtime at {linux_path}")
    assert not _private_path_hits('PRIVATE_PATHS = ["C:" + "\\\\" + "Users"]')

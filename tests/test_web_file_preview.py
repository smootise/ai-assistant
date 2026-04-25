"""Tests for the whitelisted file preview module."""

import json
import pytest

from jarvis.web.file_preview import PreviewError, _resolve_and_validate, read_for_fragment
from jarvis.store import SummaryStore
from tests.fixtures.web_seed import seed


def make_file(path, content="hello"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --- _resolve_and_validate ---

def test_allowed_path_under_root(tmp_path):
    root = tmp_path / "OUTPUTS"
    root.mkdir()
    f = make_file(root / "subdir" / "file.json", "{}")
    resolved = _resolve_and_validate(f, root)
    assert resolved == f.resolve()


def test_rejects_traversal_via_dotdot(tmp_path):
    root = tmp_path / "OUTPUTS"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    # construct a path that looks like it's under root but resolves outside
    traversal = root / ".." / "secret.txt"
    with pytest.raises(PreviewError):
        _resolve_and_validate(traversal, root)


def test_rejects_bad_extension(tmp_path):
    root = tmp_path / "OUTPUTS"
    root.mkdir()
    f = make_file(root / "script.py", "print('hi')")
    with pytest.raises(PreviewError, match="not previewable"):
        _resolve_and_validate(f, root)


def test_rejects_missing_file(tmp_path):
    root = tmp_path / "OUTPUTS"
    root.mkdir()
    ghost = root / "ghost.json"
    with pytest.raises(PreviewError, match="not found"):
        _resolve_and_validate(ghost, root)


def test_rejects_file_outside_all_roots(tmp_path):
    root = tmp_path / "OUTPUTS"
    root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()
    f = make_file(other_root / "file.json", "{}")
    with pytest.raises(PreviewError):
        _resolve_and_validate(f, root)


def test_allowed_under_second_root(tmp_path):
    root1 = tmp_path / "OUTPUTS"
    root1.mkdir()
    root2 = tmp_path / "inbox"
    root2.mkdir()
    f = make_file(root2 / "file.json", "{}")
    resolved = _resolve_and_validate(f, root1, root2)
    assert resolved == f.resolve()


# --- size cap ---

def test_size_cap_enforced(tmp_path):
    from jarvis.web import file_preview as fp
    old = fp._MAX_BYTES
    fp._MAX_BYTES = 10
    try:
        root = tmp_path / "OUTPUTS"
        root.mkdir()
        f = make_file(root / "big.json", "x" * 100)
        with pytest.raises(PreviewError, match="too large"):
            fp._read_file(f)
    finally:
        fp._MAX_BYTES = old


# --- JSON pretty-printing ---

def test_json_is_pretty_printed(tmp_path):
    from jarvis.web import file_preview as fp
    root = tmp_path / "OUTPUTS"
    root.mkdir()
    raw = '{"a":1,"b":2}'
    f = make_file(root / "data.json", raw)
    content, kind = fp._read_file(f)
    assert kind == "json"
    parsed = json.loads(content)
    assert parsed == {"a": 1, "b": 2}
    assert "\n" in content  # pretty-printed


# --- read_for_fragment ---

def test_read_for_fragment_not_found_in_db(tmp_path):
    store = SummaryStore(db_path=str(tmp_path / "test.db"))
    output_root = tmp_path / "OUTPUTS"
    output_root.mkdir()
    with pytest.raises(PreviewError, match="not found"):
        read_for_fragment(store, "ghost-frag-id", output_root)


def test_read_for_fragment_file_missing_on_disk(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = SummaryStore(db_path=db_path)
    ids = seed(store)
    output_root = tmp_path / "OUTPUTS"
    output_root.mkdir()
    # file not created on disk — should raise PreviewError, not crash
    with pytest.raises(PreviewError):
        read_for_fragment(store, ids["frag0_id"], output_root)


def test_read_for_fragment_success(tmp_path):
    from jarvis.web.file_preview import read_for_fragment

    db_path = str(tmp_path / "test.db")
    store = SummaryStore(db_path=db_path)
    ids = seed(store)

    output_root = tmp_path / "OUTPUTS"
    # Reproduce the expected path: OUTPUTS/<conv_id>/fragments/segment_000/fragment_000.json
    frag_dir = output_root / ids["conv_id"] / "fragments" / "segment_000"
    frag_dir.mkdir(parents=True)
    frag_file = frag_dir / "fragment_000.json"
    frag_file.write_text('{"fragment_id": "test"}', encoding="utf-8")

    content, kind, filename = read_for_fragment(store, ids["frag0_id"], output_root)
    assert kind == "json"
    assert "fragment_id" in content

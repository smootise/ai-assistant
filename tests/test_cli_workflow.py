"""Public CLI workflows with real temporary SQLite and a stateful fake Qdrant server."""

import copy
import json
import math
import re
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis import cli
from jarvis.clients import build_memory_layer, create_summary_store
from jarvis.config import load_config
from jarvis.recovery import CleanupService
from jarvis.vector_store import fragment_point_id


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.dimensions: dict[str, int] = {}
        self.fail_delete = False

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        assert collection_name not in self.collections
        self.collections[collection_name] = {}
        self.dimensions[collection_name] = vectors_config.size

    def get_collection(self, collection_name: str) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=self.dimensions[collection_name])
                )
            )
        )

    def upsert(self, collection_name: str, points: list[Any], wait: bool) -> None:
        assert wait
        for point in points:
            assert len(point.vector) == self.dimensions[collection_name]
            self.collections[collection_name][str(point.id)] = copy.deepcopy(point)

    def query_points(
        self, collection_name: str, query: list[float], limit: int, with_payload: bool
    ) -> Any:
        assert with_payload and len(query) == self.dimensions[collection_name]
        hits = []
        for point in self.collections[collection_name].values():
            dot = sum(a * b for a, b in zip(query, point.vector))
            norm = math.sqrt(sum(a * a for a in query) * sum(b * b for b in point.vector))
            hits.append(
                SimpleNamespace(id=point.id, payload=point.payload, score=dot / norm if norm else 0)
            )
        return SimpleNamespace(points=sorted(hits, key=lambda hit: -hit.score)[:limit])

    def delete(self, collection_name: str, points_selector: Any, wait: bool) -> None:
        assert wait
        if self.fail_delete:
            raise RuntimeError("Synthetic Qdrant deletion outage")
        for point_id in points_selector.points:
            self.collections[collection_name].pop(str(point_id), None)


class SyntheticOllama:
    model = "synthetic"

    def __init__(self) -> None:
        self.stage = ""
        self.calls = 0
        self.answer_prompts: list[str] = []

    @staticmethod
    def statements(text: str) -> list[dict]:
        return [
            {"speaker": speaker, "text": sentence}
            for speaker, sentence in re.findall(r"^(user|assistant): (.+)$", text, re.MULTILINE)
        ]

    def chat(self, system: str, user: str, **kwargs: Any) -> tuple[str, bool, str]:
        self.calls += 1
        if self.stage == "answer":
            self.answer_prompts.append(system)
            statements = self.statements(system)
            assert statements
            return " ".join(item["text"] for item in statements), False, ""
        if self.stage == "fragment-extracts":
            statements = self.statements(user)
            assert statements
            result = {"fragments": [{"title": statements[0]["text"], "statements": statements}]}
        else:
            transcript = user.split("---BEGIN SEGMENT TRANSCRIPT---")[-1]
            sentences = self.statements(transcript)
            summary = sentences[0]["text"] if sentences else user
            result = {"summary": summary, "bullets": [], "action_items": [], "confidence": 0.9}
        return json.dumps(result), False, ""

    def generate_json(self, prompt: str, schema: dict) -> tuple[str, bool, str]:
        self.calls += 1
        document = prompt.split("<document>")[-1].split("</document>")[0]
        statements = self.statements(document)
        assert statements
        return json.dumps({"statements": statements}), False, ""

    @staticmethod
    def parse_json_response(raw: str, **kwargs: Any) -> tuple[dict, bool, str]:
        return json.loads(raw), False, ""


class SyntheticEmbedder:
    model = "synthetic-embedding"

    @staticmethod
    def embed(text: str) -> list[float]:
        text = text.lower()
        return [
            (
                float(text.count(topic))
                if marker in text or not any(name in text for name in ("cedar", "orchid"))
                else 0.0
            )
            for marker in ("cedar", "orchid")
            for topic in ("atlas", "harbor", "meadow")
        ]


def export(path: Path, marker: str) -> None:
    messages = [
        ("user", f"{marker} atlas budget is 120 credits."),
        ("assistant", f"{marker} atlas launches on Monday."),
        ("user", f"{marker} harbor rollout needs two checks."),
        ("assistant", f"{marker} harbor checks cover latency and storage."),
        ("user", f"{marker} meadow archive retains seven reports."),
        ("assistant", f"{marker} meadow reports are reviewed every Friday."),
    ]
    mapping = {"root": {"id": "root", "parent": None, "children": ["m0"], "message": None}}
    for index, (speaker, text) in enumerate(messages):
        node = f"m{index}"
        mapping[node] = {
            "id": node,
            "parent": f"m{index - 1}" if index else "root",
            "children": [f"m{index + 1}"] if index + 1 < len(messages) else [],
            "message": {
                "id": node,
                "author": {"role": speaker},
                "content": {"content_type": "text", "parts": [text]},
                "create_time": 1700000000 + index,
            },
        }
    path.write_text(
        json.dumps(
            {
                "conversation_id": "shared",
                "title": marker,
                "mapping": mapping,
                "current_node": "m5",
                "create_time": 1700000000,
            }
        ),
        encoding="utf-8",
    )


def database_snapshot(config: dict) -> tuple[str, ...]:
    with closing(create_summary_store(config, read_only=True)._connect()) as connection:
        return tuple(connection.iterdump())


def artifact_snapshot(config: dict) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for key in ("inbox_root", "output_root")
        for path in Path(config[key]).rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("cleaned", ["personal", "mom"])
def test_complete_cli_workflow_and_profile_isolation(
    profile_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cleaned: str,
) -> None:
    prompts = Path(__file__).parents[1] / "prompts"
    monkeypatch.setenv("JARVIS_PROMPTS_DIR", str(prompts))
    backend, ollama = FakeQdrant(), SyntheticOllama()
    monkeypatch.setattr("jarvis.vector_store.QdrantClient", lambda **kwargs: backend)
    monkeypatch.setattr("jarvis.clients.OllamaClient", lambda **kwargs: ollama)
    monkeypatch.setattr("jarvis.clients.EmbeddingClient", lambda **kwargs: SyntheticEmbedder())

    def dispatch(profile: str, *args: str, expected: int = 0) -> str:
        ollama.stage = args[0]
        monkeypatch.setattr(sys, "argv", ["jarvis", "--profile", profile, *args])
        capsys.readouterr()
        assert cli.main() == expected
        return capsys.readouterr().out

    configs = {name: load_config(name) for name in ("personal", "mom")}
    identities = {}
    for name, marker in (("personal", "cedar"), ("mom", "orchid")):
        config = configs[name]
        other = configs["mom" if name == "personal" else "personal"]
        source = profile_repo / f"{name}.json"
        export(source, marker)
        original = source.read_bytes()
        other_files = artifact_snapshot(other)
        other_database = database_snapshot(other) if Path(other["db_path"]).exists() else None
        dispatch(name, "ingest", "chatgpt", "--file", str(source), "--persist")
        for stage in (
            "summarize-segments",
            "detect-topics",
            "extract-segments",
            "fragment-extracts",
        ):
            args = [stage, "chatgpt", "--conversation-id", "shared", "--persist"]
            if stage == "fragment-extracts":
                args.append("--embed")
            dispatch(name, *args)
        assert source.read_bytes() == original
        assert artifact_snapshot(other) == other_files
        if other_database is None:
            assert not Path(other["db_path"]).exists()
        else:
            assert database_snapshot(other) == other_database
        store = create_summary_store(config, read_only=True)
        assert store.get_conversation("shared")["title"] == marker
        segments = store.list_segments_for_conversation("shared")
        assert len(segments) == 3
        fragments = store.get_fragments_for_conversation("shared")
        assert len(fragments) == 3
        identities[name] = [row["fragment_id"] for row in fragments]
        for segment in segments:
            sid = segment["segment_id"]
            assert marker in segment["segment_text"]
            assert marker in store.get_segment_summary(sid)["summary"]
            extract = store.get_extract_by_segment(sid)
            assert extract["extract_id"] == f"{sid}_x"
            assert extract["parent_conversation_id"] == "shared"
            statements = store.get_statements_for_extract(extract["extract_id"])
            assert len(statements) >= 2 and all(marker in item["text"] for item in statements)
            assert [item["statement_id"] for item in statements] == [
                f"{sid}_x_st{index:04d}" for index in range(len(statements))
            ]
            fragment = store.list_fragments_for_extract(extract["extract_id"])[0]
            assert store.get_link_count(fragment["fragment_id"]) == len(statements)
            point = backend.collections[config["qdrant_collection"]][fragment["qdrant_point_id"]]
            assert str(point.id) == fragment_point_id(fragment["fragment_id"])
            assert point.payload["segment_id"] == sid
            assert point.payload["parent_conversation_id"] == "shared"
            assert point.payload["fragment_id"] == fragment["fragment_id"]
            hydrated = store.get_fragments_with_statements([fragment["fragment_id"]])[0]
            assert [item["text"] for item in hydrated["statements"]] == [
                item["text"] for item in statements
            ]
            assert fragment["embedded_at"] and fragment["embedding_model"] == "synthetic-embedding"
        with closing(store._connect()) as connection:
            assert connection.execute("SELECT COUNT(*) FROM topic_segments").fetchone()[0] == 3
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (Path(config["chatgpt_inbox_dir"]) / "shared/normalized.json").exists()
        assert (
            Path(config["output_root"]) / "shared/fragments/segment_002/fragment_000.md"
        ).exists()
        calls, points = ollama.calls, copy.deepcopy(backend.collections)
        for stage in (
            "summarize-segments",
            "detect-topics",
            "extract-segments",
            "fragment-extracts",
        ):
            extra = ["--embed"] if stage == "fragment-extracts" else []
            dispatch(name, stage, "chatgpt", "--conversation-id", "shared", "--persist", *extra)
        assert ollama.calls == calls and backend.collections == points
    assert identities["personal"] == identities["mom"]
    assert set(backend.collections) == {"jarvis_fragments", "jarvis_mom_fragments"}
    query_args = ["--top-k", "1", "--min-results", "1"]
    for name, marker, foreign in (("personal", "cedar", "orchid"), ("mom", "orchid", "cedar")):
        output = dispatch(name, "retrieve", "--query", "atlas budget", *query_args)
        assert f"{marker} atlas" in output and foreign not in output
        memory = build_memory_layer(configs[name], read_only=True)
        rows, scores = cli.run_retrieval(memory, "atlas budget", 1, 1, 0.5)
        assert rows == memory.store.get_fragments_with_statements(["shared_s000_x_f000"])
        assert scores[rows[0]["fragment_id"]] > 0.5
        output = dispatch(name, "answer", "atlas budget", *query_args)
        assert f"{marker} atlas budget is 120 credits." in output and foreign not in output
        assert marker in ollama.answer_prompts[-1] and foreign not in ollama.answer_prompts[-1]

        # Model a pre-existing collection with an indexed legacy UUID and cached JSON.
        collection = backend.collections[configs[name]["qdrant_collection"]]
        fragment_id = "shared_s002_x_f000"
        deterministic = fragment_point_id(fragment_id)
        legacy = "00000000-0000-4000-8000-000000000123"
        point = collection.pop(deterministic)
        point.id = legacy
        collection[legacy] = point
        store = create_summary_store(configs[name])
        store.prepare_fragment_embedding(fragment_id, legacy)
        calls = ollama.calls
        dispatch(
            name,
            "fragment-extracts",
            "chatgpt",
            "--conversation-id",
            "shared",
            "--persist",
            "--embed",
        )
        assert ollama.calls == calls
        assert store.get_fragment(fragment_id)["qdrant_point_id"] == legacy
        assert deterministic not in collection and len(collection) == 3
        assert str(collection[legacy].id) == legacy

    config = configs[cleaned]
    other_name = "mom" if cleaned == "personal" else "personal"
    other = configs[other_name]
    other_before = (
        database_snapshot(other),
        artifact_snapshot(other),
        copy.deepcopy(backend.collections[other["qdrant_collection"]]),
    )
    boundary_before = (database_snapshot(config), artifact_snapshot(config))
    dispatch(
        cleaned,
        "extract-segments",
        "chatgpt",
        "--conversation-id",
        "shared",
        "--force",
        "--inbox-dir",
        other["chatgpt_inbox_dir"],
        expected=1,
    )
    assert (database_snapshot(config), artifact_snapshot(config)) == boundary_before
    backend.fail_delete = True
    force_args = [
        "extract-segments",
        "chatgpt",
        "--conversation-id",
        "shared",
        "--from-segment",
        "0",
        "--to-segment",
        "0",
    ]
    dispatch(cleaned, *force_args, "--force", expected=1)
    journal = next(CleanupService(config).directory.glob("cleanup_*.json"))
    assert journal.exists()
    assert "shared_s000_x_f000" not in {
        row["fragment_id"]
        for row in create_summary_store(config).get_fragments_for_conversation("shared")
    }
    stale_point = fragment_point_id("shared_s000_x_f000")
    assert stale_point in backend.collections[config["qdrant_collection"]]
    dispatch(cleaned, "retrieve", "--query", "atlas budget", expected=1)
    dispatch(cleaned, "answer", "atlas budget", expected=1)
    backend.fail_delete = False
    dispatch(cleaned, *force_args, "--persist")
    assert not journal.exists()
    assert stale_point not in backend.collections[config["qdrant_collection"]]
    assert len(backend.collections[config["qdrant_collection"]]) == 2
    assert (
        database_snapshot(other),
        artifact_snapshot(other),
        backend.collections[other["qdrant_collection"]],
    ) == other_before
    dispatch(other_name, "retrieve", "--query", "atlas budget", *query_args)

"""Profile-scoped, delete-first cleanup with a durable, replayable intent file."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from jarvis.artifacts import atomic_json, fragment_outputs, load_output
from jarvis.clients import create_summary_store, create_vector_store
from jarvis.paths import PathBoundaryError, ProfilePaths, confined, identifier, validate_record
from jarvis.store import SummaryStore

STAGES = {
    "summarize-segments": "summaries",
    "detect-topics": "topics",
    "extract-segments": "extracts",
    "fragment-extracts": "fragments",
}


def validate_range(start: int | None, end: int | None) -> None:
    for value in (start, end):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("Segment range must contain non-negative integers")
    if end is not None and end < (start or 0):
        raise ValueError("Segment range is reversed")


def recovery_directory(config: Mapping[str, Any]) -> Path:
    paths = ProfilePaths.from_config(config)
    candidate = paths.context.output_root / ".jarvis-recovery"
    resolved = confined(paths.context.output_root, candidate)
    if resolved != candidate.absolute():
        raise PathBoundaryError("Recovery directory must not use a symlink or junction alias")
    return resolved


def check_search_ready(config: Mapping[str, Any]) -> None:
    directory = recovery_directory(config)
    if directory.exists() and any(directory.glob("cleanup_*.json")):
        raise RuntimeError("Pending cleanup: rerun the interrupted processing command to recover")


class CleanupService:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.paths = ProfilePaths.from_config(config)
        self.context = self.paths.context
        self.inbox = self.context.chatgpt_inbox_dir
        self.directory = recovery_directory(config)

    def _safe_path(self, path: Path) -> Path:
        resolved = confined(self.context.output_root, path)
        if resolved != path.absolute():
            raise PathBoundaryError("Cleanup paths must not use symlink or junction aliases")
        return resolved

    def _files(self, stage: str, conversation: str, items: list[dict]) -> list[Path]:
        root = self._safe_path(self.context.output_root / conversation)
        candidates = []
        if stage in {"summaries", "topics"}:
            topics = self._safe_path(root / "topic_summaries")
            if topics.exists():
                candidates.extend(
                    path
                    for path in topics.iterdir()
                    if re.fullmatch(r"topic_\d+\.(json|md)", path.name)
                    or path.name == ".topic_inputs.json"
                )
        for item in items:
            index = item["segment_index"]
            if stage == "summaries":
                candidates.extend(
                    root / "segment_summaries" / f"{item['segment_id']}{ext}"
                    for ext in (".json", ".md")
                )
            if stage == "extracts":
                candidates.extend(
                    root / "extracts" / f"extract_{index:03d}{ext}" for ext in (".json", ".md")
                )
            if stage in {"extracts", "fragments"}:
                pending = [root / "fragments" / f"segment_{index:03d}"]
                while pending:
                    path = self._safe_path(pending.pop())
                    if path.is_dir():
                        pending.extend(path.iterdir())
                    elif path.exists() and re.fullmatch(
                        r"fragment_(\d+\.(json|md)|manifest\.json)|\.batch_\d+\.json", path.name
                    ):
                        candidates.append(path)
        files = sorted({self._safe_path(path) for path in candidates if path.exists()})
        if self._source_paths(conversation).intersection(files):
            raise PathBoundaryError("Cleanup overlaps an original source export")
        return files

    def _source_paths(self, conversation: str) -> set[Path]:
        sources = set()
        normalized = self.paths.conversation(conversation, self.inbox) / "normalized.json"
        if normalized.exists():
            data = json.loads(normalized.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Invalid normalized source provenance")
            validate_record(data, conversation)
            original = data.get("raw_export_path")
            if original is not None:
                if not isinstance(original, str) or not original:
                    raise ValueError("Invalid normalized source provenance")
                sources.add(Path(original).absolute())
            provenance = data.get("source_provenance", [])
            if not isinstance(provenance, list):
                raise ValueError("Invalid normalized source provenance")
            for entry in provenance:
                if not isinstance(entry, dict) or not isinstance(entry.get("original_path"), str):
                    raise ValueError("Invalid normalized source provenance")
                sources.add(Path(entry["original_path"]).absolute())
        if self.context.db_path.exists():
            store = create_summary_store(self.config, read_only=True)
            row = store.get_conversation(conversation)
            if row and row.get("raw_source_file_id"):
                source = store.get_source_file(row["raw_source_file_id"])
                if source:
                    sources.add(Path(source["storage_path"]).absolute())
        return sources

    def _binding(self) -> dict[str, str]:
        endpoint = {
            "url": self.config.get("qdrant_url") or "",
            "host": self.config.get("qdrant_host", "localhost"),
            "port": self.config.get("qdrant_port", 6333),
        }
        return {
            "profile": self.context.name,
            "database": str(self.context.db_path),
            "collection": self.context.qdrant_collection,
            "output_root": str(self.context.output_root),
            "endpoint_sha256": hashlib.sha256(
                json.dumps(endpoint, sort_keys=True).encode()
            ).hexdigest(),
        }

    def _journal(self, stage: str, conversation: str) -> Path:
        digest = hashlib.sha256(f"{stage}:{conversation}".encode()).hexdigest()[:24]
        return self._safe_path(self.directory / f"cleanup_{digest}.json")

    @staticmethod
    def _read_plan(journal: Path) -> dict[str, Any]:
        try:
            plan = json.loads(journal.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("Invalid cleanup journal; retain it for operator recovery") from None
        if (
            not isinstance(plan, dict)
            or not isinstance(plan.get("stage"), str)
            or not isinstance(plan.get("conversation"), str)
            or not isinstance(plan.get("database_exists"), bool)
            or not isinstance(plan.get("items"), list)
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("segment_id"), str)
                or type(item.get("segment_index")) is not int
                for item in plan["items"]
            )
            or any(
                not isinstance(plan.get(key), list)
                or any(not isinstance(value, str) for value in plan[key])
                for key in ("files", "point_ids")
            )
        ):
            raise ValueError("Invalid cleanup journal; retain it for operator recovery")
        return plan

    def resume(self, stage: str, conversation: str, start: int = 0, end: int | None = None) -> None:
        journal = self._journal(stage, conversation)
        if journal.exists():
            plan = self._read_plan(journal)
            if any(
                item["segment_index"] < start or (end is not None and item["segment_index"] > end)
                for item in plan["items"]
            ):
                raise ValueError(
                    "Pending cleanup has a different range; rerun its original command"
                )
            self._execute(journal)
        check_search_ready(self.config)

    def invalidate(
        self,
        stage: str,
        conversation: str,
        items: list[dict],
        *,
        preserve_outputs: bool = False,
        preserve_batches: bool = False,
    ) -> None:
        identifier(conversation, "conversation_id")
        if stage not in STAGES.values():
            raise ValueError("Unknown cleanup stage")
        files = self._files(stage, conversation, items)
        if preserve_batches:
            files = [path for path in files if not re.fullmatch(r"\.batch_\d+\.json", path.name)]
        if preserve_outputs:
            preserved = set()
            for item in items:
                index = item["segment_index"]
                if stage == "summaries":
                    preserved.update(
                        self.paths.output(
                            conversation, "segment_summaries", f"{item['segment_id']}{ext}"
                        )
                        for ext in (".json", ".md")
                    )
                elif stage == "extracts":
                    preserved.update(
                        self.paths.output(conversation, "extracts", f"extract_{index:03d}{ext}")
                        for ext in (".json", ".md")
                    )
                else:
                    extract = load_output(
                        self.paths.output(conversation, "extracts", f"extract_{index:03d}.json"),
                        "ai_chat_extract",
                    )
                    directory = self.paths.output(conversation, "fragments", f"segment_{index:03d}")
                    documents = fragment_outputs(directory, extract) if extract else None
                    if documents is None:
                        raise ValueError("Cannot preserve an incomplete fragment batch")
                    preserved.add(directory / "fragment_manifest.json")
                    preserved.update(
                        directory / f"fragment_{document['fragment_index']:03d}{ext}"
                        for document in documents
                        for ext in (".json", ".md")
                    )
            files = [path for path in files if path not in preserved]
        store = None
        fragments = []
        if self.context.db_path.exists():
            store = create_summary_store(self.config)
            if stage in {"extracts", "fragments"}:
                fragments = store.get_fragments_for_conversation(
                    conversation, [item["segment_id"] for item in items]
                )
        if not files and store is None:
            return
        plan = {
            "version": 1,
            **self._binding(),
            "stage": stage,
            "conversation": conversation,
            "items": items,
            "database_exists": store is not None,
            "files": [str(path.relative_to(self.context.output_root)) for path in files],
            "point_ids": [
                row["qdrant_point_id"] for row in fragments if row.get("qdrant_point_id")
            ],
        }
        journal = self._journal(stage, conversation)
        atomic_json(journal, plan)
        self._execute(journal)

    def _execute(self, journal: Path) -> None:
        plan = self._read_plan(journal)
        if plan.get("version") != 1 or any(plan.get(k) != v for k, v in self._binding().items()):
            raise ValueError("Cleanup journal does not match the selected profile")
        stage = plan["stage"]
        conversation = identifier(plan["conversation"], "conversation_id")
        if stage not in STAGES.values() or journal != self._journal(stage, conversation):
            raise ValueError("Invalid cleanup journal stage")
        items = plan["items"]
        for item in items:
            validate_record(item)
            identifier(item["segment_id"], "segment_id")
        # Validate the entire saved write set before any mutation, including absent files.
        root = self._safe_path(self.context.output_root / conversation)
        files = [self._safe_path(self.context.output_root / name) for name in plan["files"]]
        if self._source_paths(conversation).intersection(files):
            raise PathBoundaryError("Pending cleanup overlaps an original source export")
        for path in files:
            allowed = stage in {"summaries", "topics"} and path.parent == root / "topic_summaries"
            for item in items:
                index = item["segment_index"]
                allowed |= stage == "summaries" and path in {
                    root / "segment_summaries" / f"{item['segment_id']}{ext}"
                    for ext in (".json", ".md")
                }
                allowed |= stage == "extracts" and path in {
                    root / "extracts" / f"extract_{index:03d}{ext}" for ext in (".json", ".md")
                }
                allowed |= stage in {"extracts", "fragments"} and path.is_relative_to(
                    root / "fragments" / f"segment_{index:03d}"
                )
            if not allowed or path.is_dir():
                raise PathBoundaryError("Cleanup journal contains an out-of-scope path")
        if plan["database_exists"] and not self.context.db_path.exists():
            raise RuntimeError(
                "Cleanup database is missing; retain the journal and restore the database"
            )
        if self.context.db_path.exists():
            store = create_summary_store(self.config)
            store.invalidate_stage(stage, conversation, [item["segment_id"] for item in items])
        if plan["point_ids"]:
            create_vector_store(self.config).delete_points(plan["point_ids"])
        for path in files:
            self._safe_path(path).unlink(missing_ok=True)
        journal.unlink()

    def prepare(self, stage: str, args: Any) -> None:
        self.inbox = self.paths.inbox(getattr(args, "inbox_dir", None))
        conversation = args.conversation_id
        start, end = getattr(args, "from_segment", None) or 0, getattr(args, "to_segment", None)
        self.resume(stage, conversation, start, end)
        if stage == "topics":
            inputs = hashlib.sha256(str(args.threshold).encode())
            for path in sorted(self.paths.output(conversation, "segment_summaries").glob("*.json")):
                if (
                    load_output(path, "ai_chat_segment", {"parent_conversation_id": conversation})
                    is None
                ):
                    raise ValueError("Incomplete segment summary; rerun summarize-segments first")
                inputs.update(path.name.encode())
                inputs.update(path.read_bytes())
            marker = self.paths.output(conversation, "topic_summaries", ".topic_inputs.json")
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                previous = None
            current = {"sha256": inputs.hexdigest()}
            if args.force or previous != current:
                self.invalidate(stage, conversation, [])
                atomic_json(marker, current)
            return
        if stage == "fragments":
            sources = self.paths.output(conversation, "extracts").glob("extract_*.json")
        else:
            sources = (Path(args.inbox_dir) / conversation / "segments").glob("segment_*.json")
        selected = []
        seen_ids, seen_indices = set(), set()
        for source in sorted(sources):
            data = json.loads(source.read_text(encoding="utf-8"))
            validate_record(data, conversation)
            if data["segment_id"] in seen_ids or data["segment_index"] in seen_indices:
                raise ValueError("Duplicate segment identifiers or indices")
            seen_ids.add(data["segment_id"])
            seen_indices.add(data["segment_index"])
            if start <= data["segment_index"] and (end is None or data["segment_index"] <= end):
                selected.append(data)
        invalid = []
        repair = []
        store = (
            create_summary_store(self.config, read_only=True)
            if self.context.db_path.exists()
            else None
        )
        for data in selected:
            index, segment = data["segment_index"], data["segment_id"]
            expected = {
                "segment_id": segment,
                "segment_index": index,
                "parent_conversation_id": conversation,
            }
            if stage == "summaries":
                complete = load_output(
                    self.paths.output(conversation, "segment_summaries", f"{segment}.json"),
                    "ai_chat_segment",
                    expected,
                )
            elif stage == "extracts":
                complete = load_output(
                    self.paths.output(conversation, "extracts", f"extract_{index:03d}.json"),
                    "ai_chat_extract",
                    expected,
                )
            else:
                complete = fragment_outputs(
                    self.paths.output(conversation, "fragments", f"segment_{index:03d}"), data
                )
            if args.force or complete is None:
                invalid.append({"segment_id": segment, "segment_index": index})
            elif store is not None and self._incompatible(store, stage, segment, complete):
                repair.append({"segment_id": segment, "segment_index": index})
        if invalid:
            self.invalidate(
                stage,
                conversation,
                invalid,
                preserve_batches=stage == "fragments" and not args.force,
            )
        if repair:
            self.invalidate(stage, conversation, repair, preserve_outputs=True)

    @staticmethod
    def _incompatible(store: SummaryStore, stage: str, segment: str, output: Any) -> bool:
        if stage == "summaries":
            row = store.get_segment_summary(segment)
            return bool(
                row
                and (row["status"] not in {"ok", "degraded"} or row["summary"] != output["summary"])
            )
        if stage == "extracts":
            row = store.get_extract_by_segment(segment)
            if not row:
                return False
            expected = {
                item["statement_index"]: (item["speaker"], item["text"])
                for item in output["statements"]
            }
            return row["status"] not in {"ok", "degraded"} or any(
                expected.get(item["statement_index"]) != (item["speaker"], item["text"])
                for item in store.get_statements_for_extract(row["extract_id"])
            )
        extract = store.get_extract_by_segment(segment)
        if not extract:
            return False
        rows = {
            row["fragment_index"]: row
            for row in store.list_fragments_for_extract(extract["extract_id"])
        }
        if rows.keys() - {document["fragment_index"] for document in output}:
            return True
        for document in output:
            row = rows.get(document["fragment_index"])
            if row and (
                row["status"] not in {"ok", "degraded"} or row["retrieval_text"] != document["text"]
            ):
                return True
        return False

"""Read-only configuration loading: process environment > .env > YAML defaults."""

import os
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit, urlunsplit

import yaml
from dotenv.main import DotEnv
from dotenv.parser import parse_stream

from jarvis.profiles import ConfigurationError, resolve_path, resolve_profiles
from jarvis.endpoints import validate_qdrant_endpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


class _UniqueKeyLoader(yaml.SafeLoader):
    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._checked_mappings: set[yaml.MappingNode] = set()

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        # Flattening mutates shared anchor nodes; check their explicit keys only once.
        if node not in self._checked_mappings:
            keys: set[str] = set()
            merge_seen = False
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    if merge_seen:
                        raise ConfigurationError("YAML mappings require unique string keys")
                    merge_seen = True
                    continue
                key = (
                    key_node.value
                    if key_node.tag == "tag:yaml.org,2002:value"
                    else self.construct_object(key_node)
                )
                if not isinstance(key, str) or key in keys:
                    raise ConfigurationError("YAML mappings require unique string keys")
                keys.add(key)
            self._checked_mappings.add(node)
        super().flatten_mapping(node)


def _read_yaml(path: Path, required: bool = False) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=_UniqueKeyLoader)
    except FileNotFoundError as exc:
        if not required:
            return {}
        raise ConfigurationError(f"Required configuration file not found: {path}") from exc
    except ConfigurationError as exc:
        raise ConfigurationError(f"Invalid configuration in {path}: {exc}") from exc
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read valid YAML configuration from {path}") from exc
    if data is None and not required:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration must be a mapping: {path}")
    return data


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"config.yaml '{key}' must be a mapping")
    return value


def _positive_int(value: Any, label: str, maximum: int | None = None) -> int:
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError
        result = int(value)
        if result < 1 or (maximum is not None and result > maximum):
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be a positive integer in range") from exc


def _read_environment(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError("Cannot read .env configuration") from exc
    if any(binding.error for binding in parse_stream(StringIO(text))):
        raise ConfigurationError("Invalid .env configuration")
    # Preserve load_dotenv(override=False) interpolation without setting os.environ.
    values = DotEnv(None, stream=StringIO(text), override=False).dict()
    environment = {key: value for key, value in values.items() if value is not None}
    environment.update(os.environ)
    return environment


def load_config(profile: str | None = None) -> Dict[str, Any]:
    repo_root = REPO_ROOT.resolve()
    environment = _read_environment(repo_root / ".env")
    selected = profile if profile is not None else environment.get("JARVIS_PROFILE", "personal")
    if not isinstance(selected, str) or not selected:
        raise ConfigurationError("Profile selection must be a non-empty name")

    data = _read_yaml(repo_root / "config.yaml")
    defaults = _mapping(data, "defaults")
    models = _mapping(defaults, "model")
    outputs = _mapping(defaults, "outputs")
    paths = _mapping(defaults, "paths")
    logging = _mapping(defaults, "logging")
    ollama = _mapping(defaults, "ollama")
    memory = _mapping(defaults, "memory")
    profiles = resolve_profiles(
        _read_yaml(repo_root / "profiles.yaml", required=True),
        repo_root,
        {
            "db_path": memory.get("db_path", "data/jarvis.db"),
            "qdrant_collection": "jarvis_fragments",
            "inbox_root": "inbox",
            "output_root": outputs.get("root", "OUTPUTS"),
            "user_name": defaults.get("user_name", ""),
        },
        environment,
    )
    if selected not in profiles:
        raise ConfigurationError(
            f"Unknown profile '{selected}'. Available profiles: {', '.join(profiles)}"
        )
    context = profiles[selected]
    timestamped = environment.get("JARVIS_OUTPUT_TIMESTAMP", outputs.get("timestamped", True))
    config = {
        "provider": environment.get("JARVIS_PROVIDER", defaults.get("provider", "local")),
        "local_model_name": environment.get(
            "LOCAL_MODEL_NAME", models.get("local") or "gemma4:31b"
        ),
        "openai_model": environment.get("OPENAI_MODEL", models.get("openai", "gpt-4o-mini")),
        "output_root": str(context.output_root),
        "output_timestamp": str(timestamped).lower() in ("true", "1", "yes"),
        "output_ts_format": environment.get(
            "JARVIS_OUTPUT_TS_FORMAT", outputs.get("ts_format", "%Y%m%d_%H%M%S")
        ),
        "prompts_dir": str(
            resolve_path(
                environment.get("JARVIS_PROMPTS_DIR", paths.get("prompts_dir", "prompts")),
                repo_root,
                "prompts_dir",
            )
        ),
        "samples_dir": str(
            resolve_path(
                environment.get("JARVIS_SAMPLES_DIR", paths.get("samples_dir", "samples")),
                repo_root,
                "samples_dir",
            )
        ),
        "log_level": environment.get("JARVIS_LOG_LEVEL", logging.get("level", "INFO")),
        "schema": data.get("schema", "jarvis.summarization"),
        "schema_version": data.get("schema_version", "1.0.0"),
        "repo_root": repo_root,
        "ollama_base_url": environment.get(
            "OLLAMA_BASE_URL", ollama.get("base_url", "http://localhost:11434")
        ),
        "embedding_model": environment.get(
            "EMBEDDING_MODEL", memory.get("embedding_model", "qwen3-embedding")
        ),
        "db_path": str(context.db_path),
        "qdrant_host": environment.get("QDRANT_HOST", memory.get("qdrant_host", "localhost")),
        "qdrant_port": _positive_int(
            environment.get("QDRANT_PORT", memory.get("qdrant_port", 6333)), "QDRANT_PORT", 65535
        ),
        "ollama_timeout": _positive_int(
            environment.get("OLLAMA_TIMEOUT", ollama.get("timeout", 600)), "OLLAMA_TIMEOUT"
        ),
        "user_name": context.user_name,
        "profile_name": context.name,
        "profile_context": context,
        "profiles": MappingProxyType(profiles),
        "qdrant_collection": context.qdrant_collection,
        "inbox_root": str(context.inbox_root),
        "chatgpt_inbox_dir": str(context.chatgpt_inbox_dir),
        "inbox_raw_dir": str(context.inbox_raw_dir),
    }
    non_string_fields = {
        "repo_root",
        "profile_context",
        "profiles",
        "qdrant_port",
        "ollama_timeout",
        "output_timestamp",
    }
    for key, value in config.items():
        if key not in non_string_fields and not isinstance(value, str):
            raise ConfigurationError(f"Configuration '{key}' must be a string")
    embedding_endpoint = environment.get(
        "EMBEDDING_OLLAMA_BASE_URL", ollama.get("embedding_base_url", "")
    )
    if not isinstance(embedding_endpoint, str):
        raise ConfigurationError("Configuration 'embedding_ollama_base_url' must be a string")
    config["embedding_ollama_base_url"] = embedding_endpoint or config["ollama_base_url"]
    config["qdrant_url"] = environment.get("QDRANT_URL", memory.get("qdrant_url", ""))
    config["qdrant_api_key"] = environment.get("QDRANT_API_KEY", "")
    for key in ("embedding_ollama_base_url", "qdrant_url", "qdrant_api_key"):
        if not isinstance(config[key], str):
            raise ConfigurationError(f"Configuration '{key}' must be a string")
    if config["qdrant_url"]:
        conflict = any(key in environment for key in ("QDRANT_HOST", "QDRANT_PORT"))
        if "QDRANT_URL" not in environment:
            conflict |= any(key in memory for key in ("qdrant_host", "qdrant_port"))
        if conflict:
            raise ConfigurationError("QDRANT_URL cannot be combined with QDRANT_HOST/QDRANT_PORT")
        validate_qdrant_endpoint(config["qdrant_url"], None, None)
    else:
        validate_qdrant_endpoint(None, config["qdrant_host"], config["qdrant_port"])
    return config


def _redact_url(value: str) -> str:
    has_scheme = "://" in value
    try:
        parsed = urlsplit(value if has_scheme else f"//{value}")
        safe = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc.rsplit("@", 1)[-1],
                parsed.path,
                "<redacted>" if parsed.query else "",
                "<redacted>" if parsed.fragment else "",
            )
        )
        return safe if has_scheme else safe.removeprefix("//")
    except ValueError:
        return "<invalid URL>"


def format_config(config: Mapping[str, Any]) -> str:
    fields = (
        ("Profile", "profile_name"),
        ("Repository root", "repo_root"),
        ("SQLite database", "db_path"),
        ("Inbox root", "inbox_root"),
        ("ChatGPT ingestion directory", "chatgpt_inbox_dir"),
        ("Raw upload directory", "inbox_raw_dir"),
        ("Output root", "output_root"),
        ("Qdrant collection", "qdrant_collection"),
        ("Qdrant URL", "qdrant_url"),
        ("Qdrant host", "qdrant_host"),
        ("Qdrant port", "qdrant_port"),
        ("Provider", "provider"),
        ("Inference model", "local_model_name"),
        ("OpenAI model", "openai_model"),
        ("Embedding model", "embedding_model"),
        ("Ollama endpoint", "ollama_base_url"),
        ("Embedding Ollama endpoint", "embedding_ollama_base_url"),
        ("Ollama timeout (seconds)", "ollama_timeout"),
        ("Prompts directory", "prompts_dir"),
        ("Samples directory", "samples_dir"),
        ("Timestamped outputs", "output_timestamp"),
        ("Output timestamp format", "output_ts_format"),
        ("Log level", "log_level"),
        ("User identity", "user_name"),
    )
    lines = []
    for label, key in fields:
        value = str(config[key])
        if key in {"ollama_base_url", "embedding_ollama_base_url", "qdrant_host", "qdrant_url"}:
            value = _redact_url(value)
        lines.append(f"{label}: {value}")
    return "\n".join(lines)

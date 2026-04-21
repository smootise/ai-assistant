"""Configuration loader for JARVIS.

Precedence: ENV (.env) > config.yaml defaults.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from .env and config.yaml.

    Returns:
        Dictionary with configuration values.
    """
    # Load .env file if it exists
    repo_root = Path(__file__).parent.parent.parent
    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        logger.debug(f"Loaded environment from {dotenv_path}")

    # Load config.yaml
    config_path = repo_root / "config.yaml"
    config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        logger.debug(f"Loaded config from {config_path}")

    # Build effective config with precedence: ENV > YAML
    defaults = config_data.get("defaults", {})

    config = {
        # Provider (for future; hardcoded to 'local' in Sprint 0)
        "provider": os.getenv("JARVIS_PROVIDER", defaults.get("provider", "local")),
        # Local model name
        "local_model_name": os.getenv(
            "LOCAL_MODEL_NAME",
            defaults.get("model", {}).get("local") or "gemma4:31b",
        ),
        # OpenAI model (not used in Sprint 0)
        "openai_model": os.getenv(
            "OPENAI_MODEL", defaults.get("model", {}).get("openai", "gpt-4o-mini")
        ),
        # Output settings
        "output_root": os.getenv(
            "JARVIS_OUTPUT_ROOT", defaults.get("outputs", {}).get("root", "OUTPUTS")
        ),
        "output_timestamp": os.getenv(
            "JARVIS_OUTPUT_TIMESTAMP",
            str(defaults.get("outputs", {}).get("timestamped", True)),
        ).lower()
        in ("true", "1", "yes"),
        "output_ts_format": os.getenv(
            "JARVIS_OUTPUT_TS_FORMAT",
            defaults.get("outputs", {}).get("ts_format", "%Y%m%d_%H%M%S"),
        ),
        # Paths
        "prompts_dir": os.getenv(
            "JARVIS_PROMPTS_DIR",
            defaults.get("paths", {}).get("prompts_dir", "prompts"),
        ),
        "samples_dir": os.getenv(
            "JARVIS_SAMPLES_DIR", defaults.get("paths", {}).get("samples_dir", "samples")
        ),
        # Logging
        "log_level": os.getenv(
            "JARVIS_LOG_LEVEL", defaults.get("logging", {}).get("level", "INFO")
        ),
        # Schema info
        "schema": config_data.get("schema", "jarvis.summarization"),
        "schema_version": config_data.get("schema_version", "1.0.0"),
        # Repo root for path resolution
        "repo_root": repo_root,
        # Ollama base URL (shared by inference and embedding clients)
        "ollama_base_url": os.getenv(
            "OLLAMA_BASE_URL",
            defaults.get("ollama", {}).get("base_url", "http://localhost:11434"),
        ),
        # Memory / embedding
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL",
            defaults.get("memory", {}).get("embedding_model", "qwen3-embedding"),
        ),
        "db_path": os.getenv(
            "JARVIS_DB_PATH",
            defaults.get("memory", {}).get("db_path", "data/jarvis.db"),
        ),
        # Qdrant
        "qdrant_host": os.getenv(
            "QDRANT_HOST",
            defaults.get("memory", {}).get("qdrant_host", "localhost"),
        ),
        "qdrant_port": int(
            os.getenv(
                "QDRANT_PORT",
                str(defaults.get("memory", {}).get("qdrant_port", 6333)),
            )
        ),
        # Ollama inference timeout (seconds). 600s default for large models + long prompts.
        "ollama_timeout": int(
            os.getenv(
                "OLLAMA_TIMEOUT",
                str(defaults.get("ollama", {}).get("timeout", 600)),
            )
        ),
        # User identity — injected into the answer prompt so JARVIS knows who it belongs to.
        "user_name": os.getenv("JARVIS_USER_NAME", defaults.get("user_name", "")),
    }

    # Log effective configuration (excluding secrets)
    logger.info("Effective configuration:")
    logger.info(f"  Provider: {config['provider']}")
    logger.info(f"  Local Model: {config['local_model_name']}")
    logger.info(f"  Embedding Model: {config['embedding_model']}")
    logger.info(f"  Output Root: {config['output_root']}")
    logger.info(
        f"  Timestamped Outputs: {config['output_timestamp']} "
        f"(format: {config['output_ts_format']})"
    )
    logger.info(f"  Schema: {config['schema']} v{config['schema_version']}")
    logger.info(f"  DB Path: {config['db_path']}")
    logger.info(f"  Qdrant: {config['qdrant_host']}:{config['qdrant_port']}")

    return config

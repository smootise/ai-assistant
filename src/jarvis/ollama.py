"""Ollama API client for local model inference."""

import json
import logging
import re
from typing import Dict, Any, Tuple

import requests


logger = logging.getLogger(__name__)


class OllamaClient:
    """Simple HTTP client for Ollama API."""

    def __init__(
        self,
        model: str = "gemma4:31b",
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama server URL.
            model: Model name to use for inference.
            timeout: HTTP request timeout in seconds. Default 600s (10 min) to
                     accommodate large models and long prompts.
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.generate_url = f"{base_url}/api/generate"

    def generate(self, prompt: str, temperature: float = 0.3) -> Tuple[str, bool, str]:
        """Send prompt to Ollama and get response.

        Args:
            prompt: The full prompt to send.
            temperature: Sampling temperature (0.0-1.0).

        Returns:
            Tuple of (response_text, is_degraded, warning_message).
            is_degraded is True if recovery was needed.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError: If Ollama returns an error.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        logger.info(f"Sending prompt to Ollama (model={self.model})...")
        logger.debug(f"Prompt length: {len(prompt)} chars")

        try:
            response = requests.post(self.generate_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise ConnectionError(
                f"Ollama server unreachable at {self.base_url}. "
                "Is Ollama running? (Try: ollama serve)"
            ) from e
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            raise RuntimeError(f"Ollama inference timed out after {self.timeout}s") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Ollama request failed: {e}") from e

        result = response.json()
        if "error" in result:
            raise RuntimeError(f"Ollama error: {result['error']}")

        raw_response = result.get("response", "").strip()
        logger.debug(f"Raw response length: {len(raw_response)} chars")

        return raw_response, False, ""

    def parse_json_response(self, raw_response: str) -> Tuple[Dict[str, Any], bool, str]:
        """Parse JSON from model response with recovery logic.

        Strategy:
        1. Try strict JSON parse first.
        2. If fails, strip markdown code fences and try again (degraded).
        3. If both fail, raise ValueError.

        Args:
            raw_response: Raw text from model.

        Returns:
            Tuple of (parsed_dict, is_degraded, warning_message).

        Raises:
            ValueError: If JSON parsing fails even after recovery.
        """
        # Attempt 1: Strict parse
        try:
            parsed = json.loads(raw_response)
            logger.info("JSON parsed successfully (strict)")
            return parsed, False, ""
        except json.JSONDecodeError:
            logger.warning("Strict JSON parse failed, attempting recovery...")

        # Attempt 2: Strip markdown code fences
        cleaned = self._strip_code_fences(raw_response)
        if cleaned != raw_response:
            try:
                parsed = json.loads(cleaned)
                warning = "Model output required cleanup (code fences removed)"
                logger.warning(warning)
                return parsed, True, warning
            except json.JSONDecodeError:
                pass

        # Both failed
        logger.error("JSON parsing failed after recovery attempts")
        logger.debug(f"Raw response (first 500 chars): {raw_response[:500]}")
        raise ValueError(
            "Model did not return valid JSON. "
            "Check logs for raw response or adjust the prompt."
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences from text.

        Handles patterns like:
        ```json
        {...}
        ```
        or
        ```
        {...}
        ```

        Args:
            text: Input text.

        Returns:
            Text with code fences removed.
        """
        # Remove leading fence with optional language identifier
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
        # Remove trailing fence
        text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

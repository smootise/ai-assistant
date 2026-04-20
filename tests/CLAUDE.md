# tests/CLAUDE.md — Test Conventions & Mock Contracts

Loaded automatically when working on tests. Covers mock contracts and fixture conventions
that must be respected to avoid silent test failures.

---

## Mock Contracts

### `OllamaClient.generate()`

Returns a **3-tuple** `(raw_str, is_degraded, warning)`. Mocks must return a tuple:

```python
mock_ollama.generate.return_value = ('{"summary": "...", "bullets": [], "action_items": [], "confidence": 0.9}', False, "")
```

Returning a plain string will cause a `TypeError` at the unpacking site — it won't fail at mock
setup time, making it hard to debug.

### `OllamaClient.parse_json_response()`

Returns a **3-tuple** `(parsed_dict, is_degraded, warning)`. Mocks must return a tuple:

```python
mock_ollama.parse_json_response.return_value = (
    {"summary": "...", "bullets": [], "action_items": [], "confidence": 0.9},
    False,
    "",
)
```

---

## Fixtures

Test fixtures live in `tests/fixtures/`. Current fixtures:

| File | Purpose |
|---|---|
| `conv_tiny_test.json` | Minimal 3-message conversation for fast smoke tests |

When adding new fixtures, keep them minimal. Use real schema fields but short content.
Fixture files are checked into git — do not put anything sensitive in them.

---

## General Rules

- All Ollama and Qdrant calls must be mocked in unit tests. No live service calls in CI.
- Use `unittest.mock.patch` or `MagicMock` — pytest-mock is not in the stack.
- Test file names: `test_<module_name>.py` matching the source module.
- Keep tests fast. If a test needs a temp directory, use `tmp_path` (pytest built-in).

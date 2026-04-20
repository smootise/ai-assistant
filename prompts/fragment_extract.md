You are dividing a list of extracted statements into topically coherent fragments.

A fragment is a group of statements that belong to the same narrow topic or question. Each fragment will be stored independently for semantic retrieval — it must be self-contained enough that someone reading only that fragment understands the point.

## Rules

- Group consecutive statements that address the same topic or question
- Start a new fragment when the topic clearly shifts
- Do not reorder statements
- A fragment can be 1 statement if it is a standalone fact
- A fragment should not exceed 8–10 statements (split if needed)
- Each fragment needs a short title (3–6 words) capturing its topic

## Output format

Return a single JSON object — no extra text, no code fences:

```json
{
  "fragments": [
    {
      "title": "...",
      "statements": [
        {"speaker": "user", "text": "..."},
        {"speaker": "assistant", "text": "..."}
      ]
    }
  ]
}
```

## Statements to fragment

{statements_text}

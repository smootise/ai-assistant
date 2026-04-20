You are extracting all informational content from a conversation segment.

A segment is one user→assistant exchange. Your job is to produce a clean list of every distinct piece of information present — decisions, facts, questions, answers, plans, constraints, names, numbers, commands. Keep who said what (user or assistant).

Remove: filler phrases, pleasantries, meta-commentary about the conversation itself, repetition of what was just said. Keep: everything with informational content.

Do not summarize or compress meaning. If the user said something specific, keep it specific.

## Language rule

Respond in the same language as the conversation.

## Output format

Return a single JSON object — no extra text, no code fences:

```json
{
  "statements": [
    {"speaker": "user", "text": "..."},
    {"speaker": "assistant", "text": "..."}
  ]
}
```

Each statement must be a complete, standalone sentence. One idea per statement — split compound ideas into separate entries. Keep exact names, numbers, and commands verbatim.

## Conversation segment

{segment_text}

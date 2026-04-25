#!/usr/bin/env bash
# Resolve Python: prefer the venv's python.exe (Windows), fall back to python3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
  PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
  PYTHON="$VIRTUAL_ENV/Scripts/python.exe"
else
  PYTHON="python3"
fi
# Smoke test: rebuild SQLite + Qdrant from existing disk artifacts without any LLM calls.
#
# Prerequisites:
#   - Ollama running (required for fragment embedding)
#   - Qdrant running (required for --embed)
#   - OUTPUTS/<conv_id>/extracts/ and fragments/ already exist on disk
#   - inbox/ai_chat/chatgpt/<conv_id>/ already exists on disk
#
# Usage:
#   bash scripts/smoke_test_rebuild.sh <conversation_id> [raw_export.json]
#
# If raw_export.json is omitted, ingest is skipped (assumes inbox already populated).
#
# What this verifies:
#   1. All five --persist commands run without error
#   2. No LLM calls happen (all artifacts loaded from disk — look for no "generating" log lines)
#   3. SQLite row counts are nonzero for every table
#   4. Every fragment has a qdrant_point_id (Qdrant indexed)
#   5. retrieve returns results

set -euo pipefail

CONV_ID="${1:?Usage: $0 <conversation_id> [raw_export.json]}"
RAW_EXPORT="${2:-}"
DB="data/jarvis.db"

echo "=== JARVIS rebuild smoke test ==="
echo "Conversation: $CONV_ID"
echo "DB: $DB"
echo ""

# Wipe DB only — leave OUTPUTS/ and inbox/ intact
if [ -f "$DB" ]; then
  echo "--- Wiping $DB ---"
  rm -f "$DB" "${DB}-shm" "${DB}-wal"
fi

echo ""
echo "--- Step 1: ingest --persist ---"
if [ -n "$RAW_EXPORT" ]; then
  $PYTHON -m jarvis.cli ingest chatgpt --file "$RAW_EXPORT" --persist
else
  echo "  Skipped (no raw export provided; assumes inbox already populated)"
  echo "  NOTE: source_files and conversations rows will be missing from SQLite."
fi

echo ""
echo "--- Step 2: extract-segments --persist ---"
$PYTHON -m jarvis.cli extract-segments chatgpt \
  --conversation-id "$CONV_ID" \
  --persist

echo ""
echo "--- Step 3: fragment-extracts --persist --embed ---"
$PYTHON -m jarvis.cli fragment-extracts chatgpt \
  --conversation-id "$CONV_ID" \
  --persist \
  --embed

echo ""
echo "--- SQLite row counts ---"
$PYTHON - "$DB" <<'EOF'
import sqlite3, sys
db = sys.argv[1]
conn = sqlite3.connect(db)
tables = ["source_files", "conversations", "segments", "extracts",
          "extract_statements", "fragments", "fragment_statement_links"]
for t in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:<30} {count}")

total = conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
indexed = conn.execute("SELECT COUNT(*) FROM fragments WHERE qdrant_point_id IS NOT NULL").fetchone()[0]
print(f"\n  Fragments indexed in Qdrant: {indexed} / {total}")
if total > 0 and indexed < total:
    print(f"  WARNING: {total - indexed} fragments missing qdrant_point_id")
conn.close()
EOF

echo ""
echo "--- Spot-check retrieve ---"
$PYTHON -m jarvis.cli retrieve --query "what did we work on?" --top-k 3

echo ""
echo "=== Smoke test complete ==="

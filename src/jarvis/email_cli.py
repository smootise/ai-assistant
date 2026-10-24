"""Profile-routed email operators; content is printed only by explicit show."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
from typing import Any

from jarvis.clients import create_email_store, create_summary_store
from jarvis.email_store import EmailError, migrate_email_schema
from jarvis.ingest.email import import_pst
from jarvis.paths import ProfilePaths
from jarvis.profiles import ConfigurationError
from jarvis.store import DatabaseCompatibilityError


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def add_email_parsers(subparsers: Any, ingest_subparsers: Any) -> None:
    ingest = ingest_subparsers.add_parser("pst", help="Import original PST emails into SQLite")
    ingest.add_argument("--pst", required=True, type=Path)
    ingest.add_argument("--limit", type=positive_integer)
    ingest.add_argument("--dry-run", action="store_true")
    email = subparsers.add_parser("email", help="Email schema, counts and original records")
    commands = email.add_subparsers(dest="email_command", required=True)
    commands.add_parser("init", help="Explicitly initialize a missing profile database as v8")
    migration = commands.add_parser(
        "migrate", help="Back up, verify and install the email extension"
    )
    migration.add_argument("--backup", required=True, type=Path)
    commands.add_parser("status", help="Counts, archive integrity and recent import runs")
    listing = commands.add_parser("list", help="List local IDs without printing private text")
    listing.add_argument("--limit", type=positive_integer, default=20)
    listing.add_argument("--failures", action="store_true")
    show = commands.add_parser("show", help="Print an original email including private body text")
    show.add_argument("email_id")


def validate_backup_boundary(paths: ProfilePaths, backup: Path) -> Path:
    candidate = backup.expanduser().absolute()
    resolved = candidate.resolve()
    if candidate != resolved:
        raise EmailError("Backup path must not be redirected")
    for profile in paths.profiles.values():
        if any(
            resolved == Path(f"{profile.db_path}{suffix}")
            for suffix in ("", "-wal", "-shm", "-journal")
        ):
            raise EmailError("Backup overlaps profile database storage")
        if any(resolved.is_relative_to(root) for root in (profile.inbox_root, profile.output_root)):
            raise EmailError("Backup must be outside managed profile artifact directories")
    return resolved


def run_email_command(args: argparse.Namespace, config: dict[str, Any]) -> int:
    try:
        paths = ProfilePaths.from_config(config)
        profile = paths.context.name
        if args.command == "ingest":
            source = paths.source(args.pst)
            result = import_pst(
                source,
                paths.context.db_path,
                profile,
                limit=args.limit,
                dry_run=args.dry_run,
                progress=lambda message: print(message, file=sys.stderr),
            )
            print(json.dumps(result, indent=2))
            if result["status"] == "interrupted":
                return 130
            return 0 if result["status"] in ("complete", "limited") else 1
        if args.email_command == "init":
            if paths.context.db_path.exists():
                raise EmailError("Database already exists; init never replaces or changes it")
            create_summary_store(config)
            print("Initialized a new v8 database. Email migration is still required.")
            return 0
        if args.email_command == "migrate":
            backup = validate_backup_boundary(paths, args.backup)
            result = migrate_email_schema(paths.context.db_path, backup, profile)
            print(f"Email schema: {result}.")
            return 0
        with closing(create_email_store(config, read_only=True)) as store:
            if args.email_command == "status":
                result = store.status()
            elif args.email_command == "list":
                result = store.list_items(args.limit, args.failures)
            else:
                result = store.get_email(args.email_id)
                if result is None:
                    raise EmailError("Email record was not found in the selected profile")
            print(json.dumps(result, indent=2, ensure_ascii=True))
            return 0
    except EmailError as error:
        print(f"Email operation refused: {error}", file=sys.stderr)
        return 1
    except DatabaseCompatibilityError:
        print(
            "Email operation refused: a compatible initialized v8 database is required.",
            file=sys.stderr,
        )
        return 1
    except ConfigurationError:
        print(
            "Email operation refused: invalid profile or source/backup boundary.", file=sys.stderr
        )
        return 1
    except KeyboardInterrupt:
        print("Email operation interrupted before completion.", file=sys.stderr)
        return 130
    except Exception:
        print(
            "Email operation failed. Check installation, permissions and import status; "
            "private parser errors are suppressed.",
            file=sys.stderr,
        )
        return 1

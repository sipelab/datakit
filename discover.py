"""File-system discovery helpers that produce machine-readable manifests.

The functions and classes here encapsulate our expectations about experiment
layout (BIDS-style folder names, ``data`` vs ``processed`` roots, etc.) so that
analysts can focus on downstream transformation instead of directory walking.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from . import logger
from .datamodel import Manifest, ManifestEntry
from .sources.register import DataSource

#TODO this should go together in a global config file
_BIDS_COMPONENT_PATTERN = re.compile(r"(sub|ses|task)-([A-Za-z0-9]+)", re.IGNORECASE)


class DataManifest:
    """Eager manifest builder backed by :class:`datakit.sources.register.DataSource`.

    Point it at an experiment root and it finds every file matching the
    registered glob patterns, attaching derived subject/session/task tokens in
    the process.
    """

    def __init__(self, root: Path, entries: Iterable[ManifestEntry] | None = None) -> None:
        self.root = Path(root).resolve()
        if entries is None:
            self.entries = self._discover()
        else:
            self.entries = sorted(list(entries), key=lambda entry: (entry.tag, entry.path))
        if not self.entries:
            raise ValueError(f"No files discovered for {self.root}")

    def _discover(self) -> list[ManifestEntry]:
        logger.info(
            "Starting file discovery",
            extra={"phase": "discover", "experiment_dir": str(self.root)},
        )

        entries: list[ManifestEntry] = []
        sources = DataSource.get_registered_sources()

        for tag in sorted(sources.keys()):
            logger.debug("Discovering files", extra={"phase": "discover", "tag": tag})
            latest_version = DataSource.get_latest_version(tag)
            source_class = sources[tag][latest_version]

            for pattern in getattr(source_class, "patterns", ()):  # type: ignore[arg-type]
                for origin in ("processed", "data"):
                    search_root = self.root / origin
                    if not search_root.exists():
                        continue
                    for file_path in search_root.glob(pattern):
                        rel_path = file_path.relative_to(self.root)
                        subject, session, task = self._parse_components(rel_path, tag)
                        entry = ManifestEntry(
                            tag=tag,
                            path=rel_path.as_posix(),
                            origin=origin,
                            subject=subject,
                            session=session,
                            task=task,
                        )
                        entries.append(entry)
                        logger.debug(
                            "Found file",
                            extra={
                                "phase": "discover",
                                "tag": tag,
                                "path": entry.path,
                                "origin": origin,
                                "task": task,
                            },
                        )

        entries.sort(key=lambda entry: (entry.tag, entry.path))
        logger.info(
            "Discovery complete",
            extra={"phase": "discover", "total_files": len(entries)},
        )
        return entries

    @staticmethod
    def _parse_components(relative_path: Path, tag: str) -> tuple[str, str, str | None]:
        captures: dict[str, str] = {}
        normalized_path = relative_path.as_posix()
        for label, value in _BIDS_COMPONENT_PATTERN.findall(normalized_path):
            captures.setdefault(label.lower(), value)

        parts = list(relative_path.parts)
        if parts and parts[0] in {"data", "processed"}:
            parts = parts[1:]

        subject = captures.get("sub")
        if subject is None and parts:
            candidate = parts[0]
            if candidate.startswith("sub-"):
                _, _, remainder = candidate.partition("-")
                subject = remainder or None
            else:
                subject = candidate

        session_value = captures.get("ses")
        session_token: str | None = None
        if session_value is not None:
            session_token = f"ses-{session_value}"
        else:
            for component in parts:
                if component.startswith("ses-"):
                    session_token = component
                    break

        task_value = captures.get("task")
        task_token: str | None = None
        if task_value is not None:
            task_token = f"task-{task_value}"
        else:
            match = re.search(r"task-([A-Za-z0-9]+)", normalized_path)
            if match:
                task_token = f"task-{match.group(1)}"
            else:
                task_token = "task-session"

        missing: list[str] = []
        if not subject:
            missing.append("subject")
        if not session_token:
            missing.append("session")
        if missing:
            raise ValueError(
                f"Missing {', '.join(missing)} token(s) in path '{normalized_path}' for tag '{tag}'"
            )

        assert subject is not None
        assert session_token is not None
        return subject, session_token, task_token

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(sorted({entry.tag for entry in self.entries}))

    def require(self, *tags: str) -> None:
        missing = [tag for tag in tags if tag not in self.tags]
        if missing:
            raise KeyError(f"Required sources missing: {', '.join(missing)}")

    def for_tag(self, tag: str) -> list[ManifestEntry]:
        return [entry for entry in self.entries if entry.tag == tag]

    def filter(
        self,
        *,
        subject: str | None = None,
        session: str | None = None,
        task: str | None = None,
    ) -> list[ManifestEntry]:
        result: list[ManifestEntry] = []
        for entry in self.entries:
            if subject and entry.subject != subject:
                continue
            if session and entry.session != session:
                continue
            if task and entry.task != task:
                continue
            result.append(entry)
        return result

    def as_datamodel(self) -> Manifest:
        return Manifest(root=self.root, entries=list(self.entries))

    @property
    def summary(self) -> dict:
        unique_subjects = sorted({entry.subject for entry in self.entries})
        unique_sessions = sorted({entry.session for entry in self.entries})
        unique_tasks = sorted({entry.task for entry in self.entries if entry.task is not None})
        unique_tags = sorted({entry.tag for entry in self.entries})

        return {
            "root": str(self.root),
            "total_files": len(self.entries),
            "subjects": unique_subjects,
            "sessions": unique_sessions,
            "tasks": unique_tasks,
            "tags": unique_tags,
        }

    def save(self, path: Path | None = None) -> Path:
        manifest_path = Path(path) if path is not None else self.root / "manifest.json"

        payload = {
            "summary": self.summary,
            "entries": [
                {
                    "tag": entry.tag,
                    "path": entry.path,
                    "origin": entry.origin,
                    "subject": entry.subject,
                    "session": entry.session,
                    "task": entry.task,
                }
                for entry in self.entries
            ],
        }

        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Manifest saved", extra={"phase": "discover", "manifest_path": str(manifest_path)})
        return manifest_path

    @classmethod
    def load(cls, root: Path, path: Path | None = None) -> "DataManifest":
        manifest_path = Path(path) if path is not None else Path(root) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        entries_payload = payload.get("entries")
        if not isinstance(entries_payload, list):
            raise ValueError("Manifest must include an 'entries' list")

        entries = []
        for item in entries_payload:
            session = item.get("session")
            if not isinstance(session, str) or not session.startswith("ses-"):
                raise ValueError("Manifest session values must include 'ses-' prefix")

            task = item.get("task")
            if task is not None and (not isinstance(task, str) or not task.startswith("task-")):
                raise ValueError("Manifest task values must include 'task-' prefix")

            entries.append(
                ManifestEntry(
                    tag=item["tag"],
                    path=item["path"],
                    origin=item["origin"],
                    subject=item["subject"],
                    session=session,
                    task=task,
                )
            )

        return cls(root, entries=entries)


__all__ = ["DataManifest"]
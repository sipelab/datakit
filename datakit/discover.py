"""File-system discovery helpers that produce machine-readable manifests."""

from __future__ import annotations

from pathlib import Path

from mesokit_schema import parse_entities

from ._utils._logger import get_logger
from .datamodel import Manifest, ManifestEntry
from .sources import SOURCE_REGISTRY

#: Tasks are optional in BIDS but required by the inventory's index, so
#: task-less files are grouped under this sentinel.
_DEFAULT_TASK_TOKEN = "task-session"

logger = get_logger(__name__)


def discover_manifest(experiment_dir: Path | str) -> Manifest:
    """Return a :class:`Manifest` for ``experiment_dir`` by scanning registered sources."""

    root = Path(experiment_dir).resolve()
    entries = _discover_entries(root)
    if not entries:
        raise ValueError(f"No files discovered for {root}")
    entries.sort(key=lambda entry: (entry.tag, entry.path))
    return Manifest(root=root, entries=entries)


def _discover_entries(root: Path) -> list[ManifestEntry]:
    logger.info("Starting file discovery", extra={"phase": "discover", "experiment_dir": str(root)})

    entries: list[ManifestEntry] = []
    for tag in sorted(SOURCE_REGISTRY.keys()):
        logger.debug("Discovering files", extra={"phase": "discover", "tag": tag})
        source_class = SOURCE_REGISTRY[tag]

        for pattern in getattr(source_class, "patterns", ()):  # type: ignore[arg-type]
            for origin in ("processed", "data"):
                search_root = root / origin
                if not search_root.exists():
                    continue
                for file_path in search_root.glob(pattern):
                    rel_path = file_path.relative_to(root)
                    subject, session, task = _parse_components(rel_path, tag)
                    entries.append(
                        ManifestEntry(
                            tag=tag,
                            path=rel_path.as_posix(),
                            origin=origin,
                            subject=subject,
                            session=session,
                            task=task,
                        )
                    )
                    logger.debug(
                        "Found file",
                        extra={
                            "phase": "discover",
                            "tag": tag,
                            "path": rel_path.as_posix(),
                            "origin": origin,
                            "task": task,
                        },
                    )

    logger.info("Discovery complete", extra={"phase": "discover", "total_files": len(entries)})
    return entries


def _parse_components(relative_path: Path, tag: str) -> tuple[str, str, str | None]:
    """Entities in datakit's token form: bare subject, prefixed session and task."""
    try:
        entities = parse_entities(relative_path)
    except ValueError as exc:
        raise ValueError(f"{exc} (tag '{tag}')") from exc

    task_token = f"task-{entities.task}" if entities.task else _DEFAULT_TASK_TOKEN
    return entities.subject, f"ses-{entities.session}", task_token


__all__ = ["discover_manifest"]
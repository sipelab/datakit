from __future__ import annotations

import argparse
from pathlib import Path

from datakit.datamodel import LoadedStream
from datakit.experiment import ExperimentData
from datakit.sources.register import DataSource


def _ensure_datasource_registry() -> None:
    # Importing datakit.sources ensures all DataSource subclasses register with the registry.
    from datakit import sources as _  # noqa: F401


def _launch_ipython(namespace: dict[str, object], header: str) -> None:
    try:
        from IPython import embed
    except ImportError:  # pragma: no cover - optional dependency
        from code import interact

        interact(header, local=namespace)
        return

    embed(header=header, user_ns=namespace)


def _infer_experiment_root(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "data").exists() or (parent / "processed").exists():
            return parent
    return path.parent


def _load_source(path: Path, tag: str, version: str | None, *, defer_load: bool = False) -> None:
    _ensure_datasource_registry()
    experiment_root = _infer_experiment_root(path)
    experiment = ExperimentData(experiment_root, include_task_level=True)
    manifest = experiment.manifest
    inventory = experiment.data

    source = DataSource.create_loader(tag, version=version)

    def load() -> object:
        return source.load(path)

    loaded = None if defer_load else load()

    namespace: dict[str, object] = {
        "experiment": experiment,
        "manifest": manifest,
        "inventory": inventory,
        "source": source,
        "path": path,
        "load": load,
        "loaded": loaded,
    }

    if isinstance(loaded, LoadedStream):
        namespace.update(
            {
                "stream": loaded,
                "data": loaded.data,
                "timeline": loaded.t,
                "meta": loaded.meta,
            }
        )

    header = (
        "Embedded datakit source shell.\n"
        "Variables available: experiment, manifest, inventory, source, path, load, loaded"
        + ("; stream, data, timeline, meta" if isinstance(loaded, LoadedStream) else "")
    )
    _launch_ipython(namespace, header)


def main() -> int:
    """CLI entry point for datakit."""
    parser = argparse.ArgumentParser(prog="datakit", description="Datakit CLI")
    subparsers = parser.add_subparsers(dest="command")

    load_parser = subparsers.add_parser(
        "load_source",
        help="Load a single file with its DataSource into an IPython session",
    )
    load_parser.add_argument("--path", required=True, type=Path, help="Path to the source file")
    load_parser.add_argument("--tag", required=True, help="DataSource tag to use")
    load_parser.add_argument(
        "--version",
        default=None,
        help="Optional DataSource version (defaults to latest)",
    )
    load_parser.add_argument(
        "--defer-load",
        action="store_true",
        help="Open the shell without loading the file (use load() inside the session)",
    )

    args = parser.parse_args()

    if args.command == "load_source":
        path = args.path
        if not path.exists() or not path.is_file():
            parser.error(f"--path must be an existing file: {path}")
        _load_source(path, args.tag, args.version, defer_load=args.defer_load)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

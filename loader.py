"""Minimal-yet-complete data-ingestion workflow for HFSA experiments.

The module keeps the happy path front-and-centre:

1. Discover the experiment hierarchy (subjects → sessions → tasks) and express it
   as a MultiIndex inventory frame.
2. Register logical data sources as ``pd.Series`` of file paths coupled with loader
   callables.
3. Materialise each source into predictable column blocks (scalars, numpy arrays,
   or preserved raw tables for structured outputs).
4. Persist the resulting dataset out to disk with standard pandas HDF5 helpers.

Everything else—progress logging, provenance tracking, heavy orchestration—is left
to higher layers so new readers can understand exactly how data lands in the
analysis-ready table.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from datakit import logger
from datakit.config import settings
from datakit.datamodel import LoadedStream
from datakit.experiment import ExperimentData
from datakit.sources.register import DataSource


LoaderFn = Callable[[str], object]


def _ensure_datasource_registry() -> None:
    # Importing datakit.sources ensures all DataSource subclasses register with the registry.
    from datakit import sources as _  # noqa: F401


def _latest_source_class(tag: str) -> type[DataSource]:
    registry = DataSource.get_registered_sources()
    if tag not in registry:
        raise KeyError(f"No data source registered for tag '{tag}'")
    latest_version = DataSource.get_latest_version(tag)
    return registry[tag][latest_version]


def _make_loader(tag: str) -> LoaderFn:
    def load(path_str: str) -> object:
        source = DataSource.create_loader(tag)
        return source.load(Path(path_str))

    return load


def _structured_default(tag: str) -> bool:
    cls = _latest_source_class(tag)
    if hasattr(cls, "structured_output"):
        return bool(getattr(cls, "structured_output"))
    return not getattr(cls, "is_timeseries", True)


@dataclass
class SeriesSpec:
    """Declarative description of a file-backed source registered with the store."""

    name: str
    files: pd.Series  # MultiIndex aligned to (Subject, Session, Task)
    loader: LoaderFn
    structured: bool = False

    def aligned_to(self, index: pd.Index) -> SeriesSpec:
        target_index = pd.MultiIndex.from_tuples(index.tolist(), names=index.names)
        if not self.files.index.equals(target_index):
            reindexed = self.files.reindex(target_index)
            return SeriesSpec(self.name, reindexed, self.loader, self.structured)
        return self


class ExperimentStore:
    """Builder that turns an ``ExperimentData`` inventory into a rich DataFrame.

    Register one or more :class:`SeriesSpec` instances (each describing a source
    tag, file paths, and loader callable) then call :meth:`materialize` to fetch
    the data and flatten it into a single MultiIndex DataFrame plus normalized
    metadata.
    """

    def __init__(self, source: str | Path | pd.DataFrame):
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if src_path.is_dir():
                self.inventory = self._build_inventory(src_path)
            elif src_path.is_file() and src_path.suffix.lower() in {".h5", ".hdf5"}:
                self.inventory = pd.read_hdf(src_path)
            else:
                raise ValueError(f"Unsupported source: {source}")
        elif isinstance(source, pd.DataFrame):
            self.inventory = source.copy()
        else:
            raise TypeError("source must be a path or pre-built DataFrame")

        if not isinstance(self.inventory.index, pd.MultiIndex):
            raise ValueError("Inventory requires a MultiIndex on (Subject, Session, Task)")

        index_names = settings.dataset.index_names
        expected_levels = len(index_names)
        if self.inventory.index.nlevels != expected_levels:
            raise ValueError(
                f"Inventory must have {expected_levels} index levels {index_names};"
                f" got {self.inventory.index.nlevels}"
            )

        self.inventory.index.set_names(index_names, inplace=True)
        self._specs = []
        self._materialized = None
        self.meta_frame = pd.DataFrame(columns=self._meta_columns())
        self.session_attrs = {}
        self.experiment_attrs = {}
        self.time_basis = {}
        self._meta_rows = []

    @staticmethod
    def _build_inventory(root: Path) -> pd.DataFrame:
        experiment = ExperimentData(root, include_task_level=True)
        inventory = experiment.data
        if not isinstance(inventory.index, pd.MultiIndex):
            raise ValueError("Discovery inventory must provide a MultiIndex")
        index_names = settings.dataset.index_names
        expected_levels = len(index_names)
        if inventory.index.nlevels != expected_levels:
            raise ValueError(
                f"Discovery inventory is expected to have {expected_levels} levels, got {inventory.index.nlevels}"
            )
        inventory.index.set_names(index_names, inplace=True)
        return inventory

    def register_series(self, name: str, files: pd.Series, loader: LoaderFn, *, structured: bool = False) -> None:
        spec = SeriesSpec(name=name, files=files, loader=loader, structured=structured)
        self._specs.append(spec.aligned_to(self.inventory.index))

    # ------------------------------------------------------------------
    # Materialization helpers
    # ------------------------------------------------------------------
    def _flatten_frame(self, df: pd.DataFrame, prefix: str, structured: bool) -> dict:
        if structured:
            return {(prefix, "raw_output"): df}
        payload: dict[tuple[str, str], object] = {}
        for col in df.columns:
            col_series = df[col]
            payload[(prefix, str(col))] = self._series_to_cell(col_series)
        return payload

    @staticmethod
    def _series_to_cell(series: pd.Series) -> object:
        if series.empty:
            return np.nan
        if len(series) == 1:
            return series.iloc[0]
        if ExperimentStore._series_is_constant(series):
            return series.iloc[0]
        return series.to_numpy()

    @staticmethod
    def _series_is_constant(series: pd.Series) -> bool:
        if series.empty:
            return True
        try:
            unique = series.nunique(dropna=False)
        except TypeError:
            first = series.iloc[0]
            try:
                return bool(series.apply(lambda value: (value == first) or (pd.isna(value) and pd.isna(first))).all())
            except Exception:
                return False
        return unique <= 1

    def _load_file(self, path: str, spec: SeriesSpec, idx: Tuple[str, str, str]) -> dict:
        if pd.isna(path):
            return {}
        data = spec.loader(path)
        if isinstance(data, LoadedStream):
            self._record_meta(idx, spec.name, data.meta)
            payload = data.value
            if isinstance(payload, pd.DataFrame):
                return self._flatten_frame(payload, spec.name, spec.structured)
            if isinstance(payload, pd.Series):
                values = payload.to_numpy() if len(payload) > 1 else payload.iloc[0]
                return {(spec.name, "values"): values}
            if isinstance(payload, (np.ndarray, list)):
                return {(spec.name, "values"): np.asarray(payload)}
            return {(spec.name, "value"): payload}
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            return self._flatten_frame(data, spec.name, spec.structured)
        if isinstance(data, pd.Series):
            if spec.structured:
                return {(spec.name, "raw_output"): data.to_frame().T}
            values = self._series_to_cell(data)
            return {(spec.name, "values"): values}
        if isinstance(data, (np.ndarray, list)):
            return {(spec.name, "values"): np.asarray(data)}
        return {(spec.name, "value"): data}

    def materialize(self) -> pd.DataFrame:
        if not self._specs:
            raise RuntimeError("No sources registered. Call register_series first.")
        self._reset_meta_state()
        rows: List[dict] = []
        for idx in self.inventory.index:
            row: dict[tuple[str, str], object] = {}
            for spec in self._specs:
                path = spec.files.loc[idx]
                row.update(self._load_file(path, spec, idx))
            rows.append(row)
        df = pd.DataFrame(rows, index=self.inventory.index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns.set_names(["Source", "Feature"], inplace=True)
        else:
            column_values = list(df.columns)
            tuple_columns = [col for col in column_values if isinstance(col, tuple) and len(col) == 2]
            if len(tuple_columns) == len(column_values):
                df.columns = pd.MultiIndex.from_tuples(tuple_columns, names=["Source", "Feature"])
            else:
                df.columns = pd.MultiIndex.from_product([["default"], df.columns], names=["Source", "Feature"])
        self._materialized = df
        self._finalize_meta_frame()
        return df

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _meta_columns() -> List[str]:
        return list(settings.dataset.meta_columns)

    def _reset_meta_state(self) -> None:
        self._meta_rows.clear()
        self.session_attrs.clear()
        self.experiment_attrs.clear()
        self.time_basis.clear()
        self.meta_frame = pd.DataFrame(columns=self._meta_columns())

    def _record_meta(self, idx: Tuple[str, str, str], source: str, meta: Dict[str, Any]) -> None:
        if not meta:
            return
        subject, session, task = idx
        scope_key = settings.dataset.scope_key
        scope = meta.get(scope_key, "stream")
        values = {k: v for k, v in meta.items() if k != scope_key}
        if scope == settings.dataset.session_scope:
            target = self.session_attrs.setdefault((subject, session), {})
            target.update(values)
            return
        if scope == settings.dataset.experiment_scope:
            self.experiment_attrs.update(values)
            return
        meta_columns = list(settings.dataset.meta_columns)
        subject_col, session_col, task_col = settings.dataset.index_names
        source_col = meta_columns[3] if len(meta_columns) > 3 else "Source"
        key_col = meta_columns[4] if len(meta_columns) > 4 else "Key"
        value_col = meta_columns[5] if len(meta_columns) > 5 else "Value"
        dtype_col = meta_columns[6] if len(meta_columns) > 6 else "dtype"
        for key, value in values.items():
            self._meta_rows.append(
                {
                    subject_col: subject,
                    session_col: session,
                    task_col: task,
                    source_col: source,
                    key_col: key,
                    value_col: value,
                    dtype_col: type(value).__name__,
                }
            )
            if key == settings.dataset.time_basis_key and isinstance(value, str):
                self.time_basis[source] = value

    def _finalize_meta_frame(self) -> None:
        if self._meta_rows:
            self.meta_frame = pd.DataFrame(self._meta_rows, columns=self._meta_columns())
        else:
            self.meta_frame = pd.DataFrame(columns=self._meta_columns())

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_hdf(self, path: str | Path, key: str = "hfsa_mvp", *, mode: Literal["a", "w", "r+"] = "w") -> None:
        if self._materialized is None:
            raise RuntimeError("Call materialize() before saving")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._materialized.to_hdf(path, key=key, mode=mode, format="fixed")

    # Convenience ------------------------------------------------------
    def describe(self) -> pd.DataFrame:
        """Return a tidy summary of the registered sources and their coverage."""
        summary_rows = []
        for spec in self._specs:
            coverage = spec.files.notna().mean()
            summary_rows.append({
                "source": spec.name,
                "structured": spec.structured,
                "coverage_pct": round(float(coverage * 100), 2),
            })
        return pd.DataFrame(summary_rows)


__all__ = ["ExperimentStore", "SeriesSpec", "launch_dataset_shell"]


# ---------------------------------------------------------------------------
# Default pipeline wiring (parity with datakit.dataset)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DefaultSource:
    """Metadata describing how to pull a :class:`DataSource` into the store."""

    tag: str
    logical_name: str
    loader: LoaderFn
    structured: bool = False


def _build_default_sources() -> tuple[DefaultSource, ...]:
    _ensure_datasource_registry()
    desired_tags = settings.dataset.desired_tags
    sources: list[DefaultSource] = []
    logical_name_overrides = settings.dataset.logical_name_overrides

    for tag in desired_tags:
        try:
            loader = _make_loader(tag)
            structured = _structured_default(tag)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Skipping default source; no loader registered",
                extra={"phase": "mvp_default_sources", "tag": tag, "error": str(exc)},
            )
            continue

        logical_name = logical_name_overrides.get(tag, tag)
        sources.append(DefaultSource(tag=tag, logical_name=logical_name, loader=loader, structured=structured))
    return tuple(sources)


DEFAULT_SOURCES: tuple[DefaultSource, ...] = _build_default_sources()


def build_default_dataset(
    input_path: Path,
    *,
    output_path: Optional[Path] = None,
    sources: Iterable[DefaultSource] = DEFAULT_SOURCES,
) -> Path:
    """Replicate the classic experiment build using the minimalist loader."""

    experiment = ExperimentData(input_path, include_task_level=True)
    store = ExperimentStore(experiment.data)
    base_inventory = store.inventory
    missing_columns: list[str] = []

    for src in sources:
        if src.tag in base_inventory.columns:
            series = base_inventory[src.tag]
        else:
            missing_columns.append(src.tag)
            continue

        store.register_series(src.logical_name, series, loader=src.loader, structured=src.structured)

    if missing_columns:
        logger.warning(
            "Skipping sources missing from inventory",
            extra={
                "phase": "mvp_default_dataset",
                "missing_tags": sorted(missing_columns),
            },
        )

    store.materialize()
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    default_output = (input_path / "processed" / f"{timestamp}_dataset_mvp.h5").resolve()
    resolved_output = Path(output_path).resolve() if output_path is not None else default_output
    store.to_hdf(resolved_output)
    return resolved_output


def launch_dataset_shell(source: str | Path | pd.DataFrame, *, key: str = "hfsa_mvp") -> None:
    """Open an embedded IPython REPL with a dataset object preloaded.

    ``source`` can be an in-memory :class:`pandas.DataFrame` or a path to an
    HDF5 file produced by :func:`build_default_dataset`. The interactive session
    exposes two convenience names: ``dataset`` (the DataFrame) and
    ``dataset_path`` (when available).
    """

    namespace: dict[str, object] = {}

    if isinstance(source, pd.DataFrame):
        dataset = source
        dataset_path: Path | None = None
    else:
        dataset_path = Path(source).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if dataset_path.is_dir():
            raise ValueError("Provide a dataset HDF5 file, not a directory")
        dataset = pd.read_hdf(dataset_path, key=key)
        namespace["dataset_path"] = dataset_path

    namespace["dataset"] = dataset

    header = (
        "Embedded datakit MVP shell.\n"
        "Variables available: dataset (DataFrame)"
        + ("; dataset_path" if "dataset_path" in namespace else "")
    )

    try:
        from IPython import embed
    except ImportError:  # pragma: no cover - optional dependency
        from code import interact

        interact(header, local=namespace)
        return

    embed(header=header, user_ns=namespace)


def _cli():
    import typer

    def main(
        input_path: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="HFSA data root"),
        output_path: Optional[Path] = typer.Option(None, help="Optional output HDF5 path"),
    ) -> None:
        path = build_default_dataset(input_path, output_path=output_path)
        launch_dataset_shell(path)
        typer.echo(f"Saved MVP dataset to {path}")

    typer.run(main)


if __name__ == "__main__":
    _cli()

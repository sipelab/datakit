"""Agnostic dataset and experiment exploration.

Provides lightweight introspection for both pre-load experiment structures
(:class:`~datakit.experiment.ExperimentData`) and post-load materialized
datasets (``pandas.DataFrame``).  Output uses ``rich`` when available and
falls back to plain indented text otherwise.

Quick start::

    from datakit import explore

    # Pre-load: inspect an experiment directory
    explore("path/to/experiment")

    # Post-load: inspect a materialized DataFrame or pickle
    explore("path/to/dataset.pkl")

    # Programmatic access
    report = explore(experiment, print_output=False)
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TagSummary:
    """Per-tag statistics in an experiment."""
    tag: str
    file_count: int
    coverage_pct: float
    extensions: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentReport:
    """Pre-load overview of an experiment directory."""
    root: str
    n_subjects: int
    n_sessions: int
    n_tasks: int
    has_task_level: bool
    subjects: tuple[str, ...]
    sessions: tuple[str, ...]
    tasks: tuple[str, ...]
    tags: tuple[TagSummary, ...]
    n_total_files: int
    coverage_matrix: dict[str, dict[str, float]]  # tag -> {subject: pct}


@dataclass(frozen=True)
class ColumnInfo:
    """Type and structure information for a single dataset column."""
    source: str
    feature: str
    dtype: str
    detail: str  # e.g. "ndarray float64 (512, 1000)" or "DataFrame 20×3"


@dataclass(frozen=True)
class DatasetReport:
    """Post-load overview of a materialized dataset."""
    shape: tuple[int, int]
    index_names: tuple[str, ...]
    index_counts: dict[str, int]  # level name -> unique count
    n_sources: int
    n_features: int
    memory_mb: float
    sources: tuple[str, ...]
    source_features: dict[str, tuple[str, ...]]  # source -> features
    columns: tuple[ColumnInfo, ...]
    coverage: dict[str, float]  # source -> non-null pct
    hierarchy: dict[str, Any]  # nested subject -> session -> tasks


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _describe_value(val: object) -> str:
    """Return a concise type description for a single cell value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, np.ndarray):
        return f"ndarray {val.dtype} {val.shape}"
    if isinstance(val, pd.DataFrame):
        return f"DataFrame {val.shape[0]}×{val.shape[1]}"
    if isinstance(val, pd.Series):
        return f"Series len={len(val)}"
    if isinstance(val, dict):
        n = len(val)
        keys_preview = ", ".join(list(val.keys())[:3])
        if n > 3:
            keys_preview += ", …"
        return f"dict({n}) [{keys_preview}]"
    if isinstance(val, (list, tuple)):
        return f"{type(val).__name__} len={len(val)}"
    if isinstance(val, (int, float, bool, str)):
        return type(val).__name__
    return type(val).__name__


def _inspect_column(series: pd.Series) -> str:
    """Inspect actual values in an object-dtype column."""
    non_null = series.dropna()
    if non_null.empty:
        return "all null"
    sample = non_null.iloc[0]
    desc = _describe_value(sample)
    if len(non_null) > 1:
        other = non_null.iloc[-1]
        other_desc = _describe_value(other)
        if other_desc != desc:
            desc += f" (varies)"
    return desc


# ---------------------------------------------------------------------------
# Experiment exploration
# ---------------------------------------------------------------------------

def explore_experiment(experiment: Any) -> ExperimentReport:
    """Analyse an :class:`~datakit.experiment.ExperimentData` without loading files."""
    inventory = experiment.data  # DataFrame with tag columns, MultiIndex rows
    manifest = experiment.manifest

    subjects = tuple(experiment.subjects)
    sessions = tuple(experiment.sessions)
    has_task = experiment.has_task_level
    if has_task and inventory.index.nlevels >= 3:
        tasks = tuple(sorted(inventory.index.get_level_values(2).unique().tolist()))
    else:
        tasks = ()

    n_rows = len(inventory)
    tag_summaries: list[TagSummary] = []
    coverage_matrix: dict[str, dict[str, float]] = {}

    for tag in sorted(inventory.columns):
        col = inventory[tag]
        count = int(col.notna().sum())
        coverage = float(col.notna().mean()) * 100.0

        # Extension breakdown from manifest entries
        exts: list[str] = []
        for entry in manifest.entries:
            if entry.tag == tag:
                exts.append(Path(entry.path).suffix.lower())
        unique_exts = tuple(sorted(set(exts))) if exts else ()

        tag_summaries.append(TagSummary(
            tag=tag, file_count=count,
            coverage_pct=round(coverage, 1),
            extensions=unique_exts,
        ))

        # Per-subject coverage
        if subjects:
            subj_cov: dict[str, float] = {}
            for subj in subjects:
                try:
                    sub_slice = inventory.xs(subj, level=0)[tag]
                    subj_cov[subj] = round(float(sub_slice.notna().mean()) * 100.0, 1)
                except KeyError:
                    subj_cov[subj] = 0.0
            coverage_matrix[tag] = subj_cov

    return ExperimentReport(
        root=str(manifest.root),
        n_subjects=len(subjects),
        n_sessions=len(sessions),
        n_tasks=len(tasks),
        has_task_level=has_task,
        subjects=subjects,
        sessions=sessions,
        tasks=tasks,
        tags=tuple(tag_summaries),
        n_total_files=len(manifest.entries),
        coverage_matrix=coverage_matrix,
    )


# ---------------------------------------------------------------------------
# Dataset exploration
# ---------------------------------------------------------------------------

def explore_dataset(dataset: pd.DataFrame) -> DatasetReport:
    """Analyse a materialized dataset DataFrame."""
    shape = dataset.shape
    idx = dataset.index
    index_names = tuple(idx.names) if hasattr(idx, 'names') else ()
    index_counts: dict[str, int] = {}
    for i, name in enumerate(index_names):
        label = name or f"level_{i}"
        index_counts[label] = int(idx.get_level_values(i).nunique())

    # Source / feature decomposition
    cols = dataset.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2:
        sources = tuple(sorted(cols.get_level_values(0).unique().tolist()))
        src_features: dict[str, tuple[str, ...]] = {}
        for src in sources:
            feats = sorted(cols.get_level_values(1)[cols.get_level_values(0) == src].unique().tolist())
            src_features[src] = tuple(feats)
        n_features = cols.get_level_values(1).nunique()
    else:
        sources = ()
        src_features = {}
        n_features = len(cols)

    # Memory
    mem_bytes = int(dataset.memory_usage(deep=True).sum())
    memory_mb = round(mem_bytes / (1024 * 1024), 2)

    # Column info
    col_infos: list[ColumnInfo] = []
    for col_key in cols:
        if isinstance(col_key, tuple):
            src, feat = str(col_key[0]), str(col_key[1])
        else:
            src, feat = "", str(col_key)

        series = dataset[col_key]
        dt = str(series.dtype)
        detail = _inspect_column(series) if dt == "object" else dt

        col_infos.append(ColumnInfo(source=src, feature=feat, dtype=dt, detail=detail))

    # Per-source coverage
    coverage: dict[str, float] = {}
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2:
        for src in sources:
            src_cols = dataset.xs(src, axis=1, level=0, drop_level=True)
            non_null = src_cols.notna().any(axis=1).mean()
            coverage[src] = round(float(non_null) * 100.0, 1)
    else:
        for c in cols:
            coverage[str(c)] = round(float(dataset[c].notna().mean()) * 100.0, 1)

    # Hierarchy tree from index
    hierarchy: dict[str, Any] = {}
    for row_key in idx:
        if not isinstance(row_key, tuple):
            row_key = (row_key,)
        node = hierarchy
        for part in row_key:
            part_str = str(part)
            if part_str not in node:
                node[part_str] = {}
            node = node[part_str]

    return DatasetReport(
        shape=shape,
        index_names=index_names,
        index_counts=index_counts,
        n_sources=len(sources),
        n_features=n_features,
        memory_mb=memory_mb,
        sources=sources,
        source_features=src_features,
        columns=tuple(col_infos),
        coverage=coverage,
        hierarchy=hierarchy,
    )


# ---------------------------------------------------------------------------
# Rendering — rich (preferred) with plain-text fallback
# ---------------------------------------------------------------------------

_HAS_RICH: bool | None = None


def _rich_available() -> bool:
    global _HAS_RICH
    if _HAS_RICH is None:
        try:
            import rich  # noqa: F401
            _HAS_RICH = True
        except ImportError:
            _HAS_RICH = False
    return _HAS_RICH


# -- Experiment rendering --------------------------------------------------

def _render_experiment_rich(report: ExperimentReport) -> str:
    from io import StringIO
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    # Overview panel
    overview = (
        f"[bold]Root:[/bold] {report.root}\n"
        f"[bold]Subjects:[/bold] {report.n_subjects}  "
        f"[bold]Sessions:[/bold] {report.n_sessions}  "
        f"[bold]Tasks:[/bold] {report.n_tasks}  "
        f"[bold]Files:[/bold] {report.n_total_files}\n"
        f"[bold]Task-level index:[/bold] {report.has_task_level}"
    )
    console.print(Panel(overview, title="Experiment Overview", border_style="blue"))

    # Hierarchy tree
    tree = Tree("[bold]Subjects[/bold]")
    for subj in report.subjects:
        subj_node = tree.add(f"[cyan]{subj}[/cyan]")
        # Gather sessions for this subject
        for ses in report.sessions:
            ses_node = subj_node.add(f"[green]{ses}[/green]")
            if report.tasks:
                for task in report.tasks:
                    ses_node.add(f"[dim]{task}[/dim]")
    console.print(tree)

    # Tag table
    table = Table(title="Source Tags", show_lines=False)
    table.add_column("Tag", style="bold")
    table.add_column("Files", justify="right")
    table.add_column("Coverage %", justify="right")
    table.add_column("Extensions")

    for ts in report.tags:
        cov_style = "green" if ts.coverage_pct >= 80 else ("yellow" if ts.coverage_pct >= 50 else "red")
        table.add_row(
            ts.tag,
            str(ts.file_count),
            f"[{cov_style}]{ts.coverage_pct:.0f}%[/{cov_style}]",
            ", ".join(ts.extensions) if ts.extensions else "—",
        )
    console.print(table)

    # Coverage matrix (subjects × tags) — only when multiple subjects
    if len(report.subjects) > 1 and report.coverage_matrix:
        cov_table = Table(title="Coverage by Subject", show_lines=True)
        cov_table.add_column("Subject", style="bold")
        for ts in report.tags:
            cov_table.add_column(ts.tag, justify="center", max_width=8)
        for subj in report.subjects:
            cells: list[str] = []
            for ts in report.tags:
                pct = report.coverage_matrix.get(ts.tag, {}).get(subj, 0)
                style = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
                cells.append(f"[{style}]{pct:.0f}%[/{style}]")
            cov_table.add_row(subj, *cells)
        console.print(cov_table)

    return buf.getvalue()


def _render_experiment_plain(report: ExperimentReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("EXPERIMENT OVERVIEW")
    lines.append("=" * 60)
    lines.append(f"  Root:       {report.root}")
    lines.append(f"  Subjects:   {report.n_subjects}")
    lines.append(f"  Sessions:   {report.n_sessions}")
    lines.append(f"  Tasks:      {report.n_tasks}")
    lines.append(f"  Files:      {report.n_total_files}")
    lines.append(f"  Task-level: {report.has_task_level}")
    lines.append("")

    # Hierarchy
    lines.append("STRUCTURE")
    lines.append("-" * 40)
    for subj in report.subjects:
        lines.append(f"  {subj}")
        for ses in report.sessions:
            lines.append(f"    {ses}")
            for task in report.tasks:
                lines.append(f"      {task}")
    lines.append("")

    # Tags
    lines.append("SOURCE TAGS")
    lines.append("-" * 40)
    lines.append(f"  {'Tag':<25} {'Files':>5}  {'Coverage':>8}  Extensions")
    for ts in report.tags:
        ext_str = ", ".join(ts.extensions) if ts.extensions else "—"
        lines.append(f"  {ts.tag:<25} {ts.file_count:>5}  {ts.coverage_pct:>7.0f}%  {ext_str}")

    return "\n".join(lines)


# -- Dataset rendering -----------------------------------------------------

def _render_dataset_rich(report: DatasetReport) -> str:
    from io import StringIO
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    # Overview panel
    idx_parts = ", ".join(f"{k}={v}" for k, v in report.index_counts.items())
    overview = (
        f"[bold]Shape:[/bold] {report.shape[0]} rows × {report.shape[1]} columns\n"
        f"[bold]Index:[/bold] ({idx_parts})\n"
        f"[bold]Sources:[/bold] {report.n_sources}  "
        f"[bold]Features:[/bold] {report.n_features}  "
        f"[bold]Memory:[/bold] {report.memory_mb:.1f} MB"
    )
    console.print(Panel(overview, title="Dataset Overview", border_style="blue"))

    # Structure tree: sources → features
    tree = Tree("[bold]Sources[/bold]")
    for src in report.sources:
        src_node = tree.add(f"[cyan]{src}[/cyan]")
        for feat in report.source_features.get(src, ()):
            src_node.add(f"[dim]{feat}[/dim]")
    console.print(tree)

    # Data types table
    type_table = Table(title="Column Types", show_lines=False)
    type_table.add_column("Source", style="bold")
    type_table.add_column("Feature")
    type_table.add_column("dtype")
    type_table.add_column("Detail")

    for ci in report.columns:
        type_table.add_row(ci.source, ci.feature, ci.dtype, ci.detail)
    console.print(type_table)

    # Coverage table
    cov_table = Table(title="Source Coverage", show_lines=False)
    cov_table.add_column("Source", style="bold")
    cov_table.add_column("Available %", justify="right")

    for src in report.sources:
        pct = report.coverage.get(src, 0)
        style = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
        cov_table.add_row(src, f"[{style}]{pct:.0f}%[/{style}]")
    console.print(cov_table)

    # Index hierarchy tree
    if report.hierarchy:
        idx_tree = Tree("[bold]Index Hierarchy[/bold]")
        _build_hierarchy_tree(idx_tree, report.hierarchy, depth=0, level_names=report.index_names)
        console.print(idx_tree)

    return buf.getvalue()


def _build_hierarchy_tree(node: Any, tree: dict[str, Any], depth: int, level_names: tuple[str, ...]) -> None:
    """Recursively build a rich Tree from the hierarchy dict."""
    from rich.tree import Tree as RichTree

    styles = ["cyan", "green", "dim"]
    style = styles[depth] if depth < len(styles) else ""
    label = level_names[depth] if depth < len(level_names) else ""

    for key, children in sorted(tree.items()):
        child_node = node.add(f"[{style}]{key}[/{style}]" if style else key)
        if isinstance(children, dict) and children:
            _build_hierarchy_tree(child_node, children, depth + 1, level_names)


def _render_dataset_plain(report: DatasetReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("DATASET OVERVIEW")
    lines.append("=" * 60)
    idx_parts = ", ".join(f"{k}={v}" for k, v in report.index_counts.items())
    lines.append(f"  Shape:    {report.shape[0]} rows x {report.shape[1]} columns")
    lines.append(f"  Index:    ({idx_parts})")
    lines.append(f"  Sources:  {report.n_sources}")
    lines.append(f"  Features: {report.n_features}")
    lines.append(f"  Memory:   {report.memory_mb:.1f} MB")
    lines.append("")

    # Sources → features
    lines.append("STRUCTURE")
    lines.append("-" * 40)
    for src in report.sources:
        feats = report.source_features.get(src, ())
        lines.append(f"  {src}")
        for f in feats:
            lines.append(f"    {f}")
    lines.append("")

    # Column types
    lines.append("COLUMN TYPES")
    lines.append("-" * 40)
    lines.append(f"  {'Source':<20} {'Feature':<20} {'dtype':<12} Detail")
    for ci in report.columns:
        lines.append(f"  {ci.source:<20} {ci.feature:<20} {ci.dtype:<12} {ci.detail}")
    lines.append("")

    # Coverage
    lines.append("SOURCE COVERAGE")
    lines.append("-" * 40)
    for src in report.sources:
        pct = report.coverage.get(src, 0)
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        lines.append(f"  {src:<20} [{bar}] {pct:.0f}%")

    return "\n".join(lines)


# -- Dispatch ---------------------------------------------------------------

def _render(report: ExperimentReport | DatasetReport) -> str:
    if isinstance(report, ExperimentReport):
        return _render_experiment_rich(report) if _rich_available() else _render_experiment_plain(report)
    return _render_dataset_rich(report) if _rich_available() else _render_dataset_plain(report)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explore(
    target: Union["ExperimentData", pd.DataFrame, Path, str],
    *,
    print_output: bool = True,
    hdf_key: str = "hfsa_mvp",
) -> ExperimentReport | DatasetReport:
    """Explore the structure of an experiment or materialized dataset.

    Parameters
    ----------
    target
        One of:
        - :class:`~datakit.experiment.ExperimentData` instance
        - ``pandas.DataFrame`` (materialized dataset)
        - ``Path`` or ``str`` pointing to a directory (experiment root),
          ``.pkl`` file, or ``.h5``/``.hdf5`` file
    print_output
        If *True* (default), print a formatted summary to stdout.
    hdf_key
        HDF5 key used when *target* is an ``.h5`` file.

    Returns
    -------
    ExperimentReport or DatasetReport
        Structured report object for programmatic access.
    """
    # Avoid circular import at module level
    from datakit.experiment import ExperimentData

    if isinstance(target, ExperimentData):
        report = explore_experiment(target)
    elif isinstance(target, pd.DataFrame):
        report = explore_dataset(target)
    elif isinstance(target, (str, Path)):
        report = _explore_path(Path(target), hdf_key=hdf_key)
    else:
        raise TypeError(f"Unsupported target type: {type(target).__name__}")

    if print_output:
        text = _render(report)
        print(text)

    return report


def _explore_path(path: Path, *, hdf_key: str) -> ExperimentReport | DatasetReport:
    """Resolve a filesystem path to the appropriate exploration target."""
    from datakit.experiment import ExperimentData

    path = path.resolve()

    if path.is_dir():
        experiment = ExperimentData(path)
        return explore_experiment(experiment)

    if not path.is_file():
        raise FileNotFoundError(f"Path does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pkl":
        df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame in pickle, got {type(df).__name__}")
        return explore_dataset(df)

    if suffix in (".h5", ".hdf5"):
        df = pd.read_hdf(path, key=hdf_key)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame in HDF5, got {type(df).__name__}")
        return explore_dataset(df)

    raise ValueError(f"Unsupported file type: {suffix}  (expected directory, .pkl, .h5, or .hdf5)")


__all__ = [
    "explore",
    "explore_experiment",
    "explore_dataset",
    "ExperimentReport",
    "DatasetReport",
    "ColumnInfo",
    "TagSummary",
]

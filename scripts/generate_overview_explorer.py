from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore[import]
        from plotly.subplots import make_subplots  # type: ignore[import]
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
        raise ModuleNotFoundError(
            "plotly is required for the standalone explorer builder",
            name=exc.name,
        ) from exc
    return go, make_subplots


def _is_arraylike(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return False
    try:
        if np.isscalar(value):
            return False
    except Exception:
        return False
    return hasattr(value, "__len__")


def _mode_length(values: Sequence[int]) -> int:
    if not values:
        return 0
    counts: Dict[int, int] = {}
    for val in values:
        counts[int(val)] = counts.get(int(val), 0) + 1
    return max(counts, key=counts.get)


_MESO_SOURCE_SPECS: Dict[str, Dict[str, object]] = {
    "mesomap": {
        "source_tag": "mesomap",
        "drop": ("frame", "time_elapsed_s"),
        "keep": None,
        "label": "mesomap",
        # mesomap aligns its own time_elapsed_s to the dataqueue master clock
        # via the meso camera anchors (see MesoMapSource._aligned_timeline).
        "time_keys": (("mesomap", "time_elapsed_s"), ("time", "master_elapsed_s")),
        "secondary_features": (),
    },
    "meso": {
        "source_tag": "meso",
        "drop": ("time_elapsed_s",),
        "keep": ("Mean", "dF_F"),
        "label": "meso",
        # The meso CSV loader emits a frame-rate-only axis that is NOT
        # aligned to the master clock. Prefer the meso camera metadata
        # timestamps, which are derived from camera ElapsedTime and are
        # already on the master clock.
        "time_keys": (
            ("meso_metadata", "time_elapsed_s"),
            ("meso", "time_elapsed_s"),
            ("time", "master_elapsed_s"),
        ),
        # Features routed to a right-hand y-axis on the top panel.
        "secondary_features": ("dF_F",),
    },
}


def _extract_mesomap_traces(
    entry: pd.Series,
    *,
    source: str = "mesomap",
    roi_limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    spec = _MESO_SOURCE_SPECS.get(source)
    if spec is None:
        raise ValueError(f"Unknown mesomap source '{source}'")
    source_tag = str(spec.get("source_tag", source))
    if source_tag not in entry.index.get_level_values(0):
        raise ValueError(f"No '{source_tag}' columns found in dataset entry")
    meso_series = entry[source_tag]
    for drop_name in spec.get("drop", ()) or ():
        meso_series = meso_series.drop(drop_name, errors="ignore")
    keep = spec.get("keep")
    if keep:
        meso_series = meso_series[[name for name in meso_series.index if name in keep]]
    meso_series = meso_series[meso_series.apply(_is_arraylike)]
    if meso_series.empty:
        raise ValueError(f"No {spec['label']} ROI traces found in dataset entry")

    lengths = meso_series.apply(len)
    if keep:
        # Explicit feature list: keep every requested column and truncate to
        # the shortest array so heterogeneous lengths (e.g. Mean vs dF_F off
        # by a frame) don't cause one to be silently dropped.
        target_len = int(min(lengths.values))
    else:
        target_len = _mode_length([int(val) for val in lengths.values])
        meso_series = meso_series[lengths == target_len]
    if meso_series.empty:
        raise ValueError(f"No {spec['label']} ROI traces with consistent length found")

    roi_names = list(meso_series.index)
    if roi_limit is not None:
        roi_names = roi_names[: int(roi_limit)]
    meso_traces = np.stack(
        [np.asarray(meso_series.loc[name])[:target_len] for name in roi_names]
    )
    return meso_traces, [str(name) for name in roi_names]


def _extract_time_axis(
    entry: pd.Series,
    target_len: int,
    *,
    source: str = "mesomap",
) -> Tuple[np.ndarray, int]:
    """Return a time axis matched to the trace length.

    Returns (time_axis, matched_length). If the available axis is shorter
    than ``target_len``, traces should be truncated to the returned length.
    """
    spec = _MESO_SOURCE_SPECS.get(source, {})
    source_tag = str(spec.get("source_tag", source))
    candidates: List[Tuple[str, str]] = []
    for key in spec.get("time_keys", ()) or ():
        if isinstance(key, tuple) and key in entry.index:
            candidates.append(key)
    if (source_tag, "time_elapsed_s") in entry.index:
        candidates.append((source_tag, "time_elapsed_s"))
    if ("time", "master_elapsed_s") in entry.index:
        candidates.append(("time", "master_elapsed_s"))
    seen: set = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        t_raw = np.atleast_1d(np.asarray(entry[key]))
        if t_raw.ndim != 1 or t_raw.size == 0:
            continue
        try:
            if not np.isfinite(np.asarray(t_raw, dtype=np.float64)).any():
                continue
        except (TypeError, ValueError):
            continue
        matched = int(min(t_raw.size, target_len))
        return t_raw[:matched], matched
    # Final fallback: synthesize a frame-rate time axis. The mesoscope loader
    # already uses ``np.arange(n)/50`` when no master clock is available, so
    # mirroring that here keeps sessions without a dataqueue (e.g. older
    # STREHAB02/03/05) renderable instead of failing outright.
    if target_len > 0:
        return np.arange(target_len, dtype=np.float64) / 50.0, int(target_len)
    raise ValueError(
        f"No time axis available for {spec.get('label', source)} traces"
    )


def _native_trace(
    entry: pd.Series,
    *,
    source: str,
    feature: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (time, values) sampled on the source's own master-clock axis.

    No interpolation/resampling is performed: plotly shares the x-axis across
    panels, so each series stays on its native sampling grid and gaps remain
    visible instead of being papered over with held-forward values.
    """
    if (source, "time_elapsed_s") not in entry.index or (source, feature) not in entry.index:
        return None
    t_source = np.atleast_1d(np.asarray(entry[(source, "time_elapsed_s")], dtype=np.float64))
    values = np.atleast_1d(np.asarray(entry[(source, feature)], dtype=np.float64))
    if t_source.size == 0 or values.size == 0:
        return None
    if t_source.size != values.size:
        raise ValueError(f"{source} time/value arrays mismatch")
    finite = np.isfinite(t_source) & np.isfinite(values)
    t_source = t_source[finite]
    values = values[finite]
    if t_source.size == 0:
        return None
    order = np.argsort(t_source)
    t_source = t_source[order]
    values = values[order]
    _, unique_idx = np.unique(t_source, return_index=True)
    return t_source[unique_idx], values[unique_idx]


def _build_explorer_figure(
    meso_traces: np.ndarray,
    roi_names: Sequence[str],
    time_axis: np.ndarray,
    *,
    pupil: Optional[Tuple[np.ndarray, np.ndarray]],
    treadmill: Optional[Tuple[np.ndarray, np.ndarray]],
    title: str,
    secondary_features: Sequence[str] = (),
    primary_y_label: str = "ΔF/F",
    secondary_y_label: Optional[str] = None,
) -> object:
    go, make_subplots = _require_plotly()

    secondary_set = {str(name) for name in secondary_features}
    use_secondary = any(name in secondary_set for name in roi_names)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    for trace, name in zip(meso_traces, roi_names):
        on_secondary = name in secondary_set
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=trace,
                name=name,
                mode="lines",
                line=dict(width=1.4),
                hovertemplate="<b>%{meta}</b><br>%{x:.3f} : %{y:.4f}<extra></extra>",
                meta=name,
            ),
            row=1,
            col=1,
            secondary_y=on_secondary,
        )

    if pupil is not None:
        t_pupil, y_pupil = pupil
        fig.add_trace(
            go.Scatter(
                x=t_pupil,
                y=y_pupil,
                name="Pupil diameter (mm)",
                mode="lines",
                line=dict(color="#EF553B", width=1.6),
                hovertemplate="<b>Pupil</b><br>%{x:.3f} : %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if treadmill is not None:
        t_tread, y_tread = treadmill
        fig.add_trace(
            go.Scatter(
                x=t_tread,
                y=y_tread,
                name="Treadmill speed (mm)",
                mode="lines",
                line=dict(color="#00CC96", width=1.6),
                hovertemplate="<b>Treadmill</b><br>%{x:.3f} : %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=None,
        template="plotly_white",
        height=860,
        width=1280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, xanchor="left"),
        margin=dict(t=60, r=40, l=60, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text=primary_y_label, row=1, col=1, secondary_y=False)
    if use_secondary:
        fig.update_yaxes(
            title_text=secondary_y_label or "Secondary",
            row=1,
            col=1,
            secondary_y=True,
        )
    fig.update_yaxes(title_text="Pupil (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (mm)", row=3, col=1)
    return fig


def _iter_entries(dataset: pd.DataFrame) -> Iterable[Tuple[Tuple[str, str, str], pd.Series]]:
    if not isinstance(dataset.index, pd.MultiIndex):
        raise ValueError("Dataset index must be a MultiIndex of (subject, session, task)")
    for key, row in dataset.iterrows():
        if not isinstance(key, tuple) or len(key) < 3:
            continue
        subject, session, task = (str(key[0]), str(key[1]), str(key[2]))
        yield (subject, session, task), row


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_index(entries: List[Dict[str, str]], output_dir: Path) -> Path:
    manifest = json.dumps(entries, indent=2)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Mesomap Explorer Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
    label {{ font-weight: 600; }}
    select {{ min-width: 140px; padding: 4px 6px; }}
    iframe {{ width: 100%; height: 80vh; border: 1px solid #ccc; border-radius: 4px; }}
    .warning {{ color: #b33; font-weight: 600; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Mesomap Explorer Browser</h1>
  <div class=\"controls\">
    <label for=\"subject\">Subject</label>
    <select id=\"subject\"></select>
    <label for=\"session\">Session</label>
    <select id=\"session\"></select>
    <label for=\"task\">Task</label>
    <select id=\"task\"></select>
  </div>
  <div id=\"status\" class=\"warning\"></div>
  <iframe id=\"viewer\" title=\"Mesomap Explorer\" src=\"\"></iframe>

  <script>
    const manifest = {manifest};

    function uniq(values) {{ return [...new Set(values)]; }}

    const bySubject = new Map();
    manifest.forEach(entry => {{
      const subj = entry.subject || 'unknown';
      if (!bySubject.has(subj)) bySubject.set(subj, []);
      bySubject.get(subj).push(entry);
    }});

    const subjectSel = document.getElementById('subject');
    const sessionSel = document.getElementById('session');
    const taskSel = document.getElementById('task');
    const iframe = document.getElementById('viewer');
    const status = document.getElementById('status');

    function setStatus(msg) {{ status.textContent = msg || ''; }}

    function populateSubjects() {{
      subjectSel.innerHTML = '';
      uniq(Array.from(bySubject.keys())).forEach(subj => {{
        const opt = document.createElement('option');
        opt.value = subj; opt.textContent = subj; subjectSel.appendChild(opt);
      }});
    }}

    function populateSessions(subj) {{
      sessionSel.innerHTML = '';
      const entries = bySubject.get(subj) || [];
      const sessions = uniq(entries.map(e => e.session || 'unknown'));
      sessions.forEach(sess => {{
        const opt = document.createElement('option');
        opt.value = sess; opt.textContent = sess; sessionSel.appendChild(opt);
      }});
    }}

    function populateTasks(subj, sess) {{
      taskSel.innerHTML = '';
      const entries = (bySubject.get(subj) || []).filter(e => (e.session || 'unknown') === sess);
      entries.forEach(e => {{
        const opt = document.createElement('option');
        opt.value = e.html; opt.textContent = e.task;
        taskSel.appendChild(opt);
      }});
    }}

    function updateViewer() {{
      const html = taskSel.value;
      if (!html) {{
        iframe.src = '';
        setStatus('No run selected.');
        return;
      }}
      setStatus('');
      iframe.src = html;
    }}

    subjectSel.addEventListener('change', () => {{
      populateSessions(subjectSel.value);
      populateTasks(subjectSel.value, sessionSel.value);
      updateViewer();
    }});

    sessionSel.addEventListener('change', () => {{
      populateTasks(subjectSel.value, sessionSel.value);
      updateViewer();
    }});

    taskSel.addEventListener('change', updateViewer);

    populateSubjects();
    if (subjectSel.options.length) {{
      subjectSel.selectedIndex = 0;
      populateSessions(subjectSel.value);
      if (sessionSel.options.length) {{
        sessionSel.selectedIndex = 0;
        populateTasks(subjectSel.value, sessionSel.value);
      }}
    }}
    if (taskSel.options.length) {{
      taskSel.selectedIndex = 0;
      updateViewer();
    }} else {{
      setStatus('No explorer HTML files found.');
    }}
  </script>
</body>
</html>
"""
    index_path = output_dir / "mesomap_explorers.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def build_explorers_from_pickle(
    dataset_path: Path,
    *,
    output_dir: Optional[Path] = None,
    roi_limit: Optional[int] = None,
    source: str = "mesomap",
) -> List[Path]:
    dataset = pd.read_pickle(dataset_path)
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Pickle file does not contain a pandas DataFrame")

    if source not in _MESO_SOURCE_SPECS:
        raise ValueError(
            f"Unknown source '{source}'. Choose from: {sorted(_MESO_SOURCE_SPECS)}"
        )

    out_dir = output_dir or dataset_path.parent / "mesomap_explorers"
    _ensure_dir(out_dir)

    generated: List[Path] = []
    manifest: List[Dict[str, str]] = []

    for (subject, session, task), entry in _iter_entries(dataset):
        try:
            meso_traces, roi_names = _extract_mesomap_traces(
                entry, source=source, roi_limit=roi_limit
            )
            t_meso, matched_len = _extract_time_axis(
                entry, meso_traces.shape[1], source=source
            )
            if matched_len < meso_traces.shape[1]:
                meso_traces = meso_traces[:, :matched_len]
            pupil = _native_trace(
                entry, source="pupil_dlc", feature="pupil_diameter_mm"
            )
            if pupil is None:
                # Older datasets used the bare "pupil" tag.
                pupil = _native_trace(
                    entry, source="pupil", feature="pupil_diameter_mm"
                )
            treadmill = _native_trace(
                entry, source="treadmill", feature="speed_mm"
            )
            spec = _MESO_SOURCE_SPECS[source]
            secondary_features = tuple(spec.get("secondary_features", ()) or ())
            if source == "meso":
                primary_label = "Mean (a.u.)"
                secondary_label = "dF/F"
            else:
                primary_label = "ΔF/F"
                secondary_label = None
            title = f"Subject {subject} | Session {session} | Task {task}"
            fig = _build_explorer_figure(
                meso_traces,
                roi_names,
                t_meso,
                pupil=pupil,
                treadmill=treadmill,
                title=title,
                secondary_features=secondary_features,
                primary_y_label=primary_label,
                secondary_y_label=secondary_label,
            )
        except Exception as exc:
            print(f"[FAIL] {subject} {session} {task}: {exc}")
            continue

        file_label = f"{subject}_{session}_{task}_explorer.html".replace(" ", "_")
        html_path = out_dir / file_label
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        generated.append(html_path)
        manifest.append(
            {
                "subject": subject,
                "session": session,
                "task": task,
                "html": html_path.name,
            }
        )
        print(f"[OK] {subject} / {session} / {task} -> {html_path}")

    if manifest:
        _build_index(manifest, out_dir)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Plotly explorer HTML files from a mesomap dataset pickle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-pickle",
        type=Path,
        required=True,
        help="Dataset pickle containing mesomap, pupil, and treadmill arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for explorer HTML files (defaults next to the pickle)",
    )
    parser.add_argument(
        "--roi-limit",
        type=int,
        default=None,
        help="Limit the number of ROIs plotted per explorer",
    )
    parser.add_argument(
        "--source",
        choices=sorted(_MESO_SOURCE_SPECS.keys()),
        default="mesomap",
        help="Which mesoscope source to plot in the top panel",
    )
    args = parser.parse_args()

    try:
        generated = build_explorers_from_pickle(
            args.dataset_pickle,
            output_dir=args.output_dir,
            roi_limit=args.roi_limit,
            source=args.source,
        )
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    if not generated:
        print("No explorers generated. Verify that the dataset contains mesomap traces.")
        return 1

    print("\nGenerated explorers:")
    for path in generated:
        try:
            rel = path.relative_to(args.output_dir) if args.output_dir else path.name
        except Exception:
            rel = path
        print(f" - {rel}")
    if args.output_dir is not None:
        print(f"Index written to {args.output_dir / 'mesomap_explorers.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

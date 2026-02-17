from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TIME_OFFSET_THRESHOLD_S = 0.5


def _maybe_shift_timebase(
    t_source: np.ndarray,
    time_axis: np.ndarray,
    *,
    label: str,
    threshold_s: float = TIME_OFFSET_THRESHOLD_S,
) -> tuple[np.ndarray, float]:
    if t_source.size == 0 or time_axis.size == 0:
        return t_source, 0.0
    offset = float(t_source[0] - time_axis[0])
    if abs(offset) >= threshold_s:
        print(f"[WARN] {label}: shifting timebase by {offset:.3f}s to match master axis")
        return t_source - offset, offset
    return t_source, 0.0


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


def _extract_mesomap_traces(entry: pd.Series, *, roi_limit: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    meso_series = entry["mesomap"].drop("frame", errors="ignore")
    meso_series = meso_series[meso_series.apply(_is_arraylike)]
    if meso_series.empty:
        raise ValueError("No mesomap ROI traces found in dataset entry")

    lengths = meso_series.apply(len)
    target_len = _mode_length([int(val) for val in lengths.values])
    meso_series = meso_series[lengths == target_len]
    if meso_series.empty:
        raise ValueError("No mesomap ROI traces with consistent length found")

    roi_names = list(meso_series.index)
    if roi_limit is not None:
        roi_names = roi_names[: int(roi_limit)]
    meso_traces = np.stack([np.asarray(meso_series.loc[name]) for name in roi_names])
    return meso_traces, [str(name) for name in roi_names]


def _extract_time_axis(entry: pd.Series, target_len: int) -> np.ndarray:
    t_raw = np.asarray(entry[("time", "master_elapsed_s")])
    if t_raw.size < target_len + 1:
        raise ValueError("Time axis is shorter than mesomap traces")
    return t_raw[:target_len]


def _interp_trace(
    entry: pd.Series,
    time_axis: np.ndarray,
    *,
    source: str,
    feature: str,
    method: str = "linear",
    align_offset: bool = False,
) -> Optional[np.ndarray]:
    if (source, "time_elapsed_s") not in entry.index or (source, feature) not in entry.index:
        return None
    t_source = np.asarray(entry[(source, "time_elapsed_s")])
    values = np.asarray(entry[(source, feature)])
    if t_source.size == 0 or values.size == 0:
        return None
    if t_source.size != values.size:
        raise ValueError(f"{source} time/value arrays mismatch")
    order = np.argsort(t_source)
    t_source = t_source[order]
    values = values[order]
    if t_source.size:
        _, unique_idx = np.unique(t_source, return_index=True)
        t_source = t_source[unique_idx]
        values = values[unique_idx]
    if align_offset:
        t_source, _ = _maybe_shift_timebase(t_source, time_axis, label=source)
    if method == "previous":
        indices = np.searchsorted(t_source, time_axis, side="right") - 1
        indices = np.clip(indices, 0, len(values) - 1)
        return values[indices]
    return np.interp(time_axis, t_source, values)


def _build_explorer_figure(
    meso_traces: np.ndarray,
    roi_names: Sequence[str],
    time_axis: np.ndarray,
    *,
    pupil: Optional[np.ndarray],
    treadmill: Optional[np.ndarray],
    title: str,
) -> object:
    go, make_subplots = _require_plotly()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
    )

    for trace, name in zip(meso_traces, roi_names):
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
        )

    if pupil is not None:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=pupil,
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
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=treadmill,
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
    fig.update_yaxes(title_text="ΔF/F", row=1, col=1)
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
) -> List[Path]:
    dataset = pd.read_pickle(dataset_path)
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Pickle file does not contain a pandas DataFrame")

    out_dir = output_dir or dataset_path.parent / "mesomap_explorers"
    _ensure_dir(out_dir)

    generated: List[Path] = []
    manifest: List[Dict[str, str]] = []

    for (subject, session, task), entry in _iter_entries(dataset):
        try:
            meso_traces, roi_names = _extract_mesomap_traces(entry, roi_limit=roi_limit)
            t_meso = _extract_time_axis(entry, meso_traces.shape[1])
            pupil = _interp_trace(entry, t_meso, source="pupil", feature="pupil_diameter_mm")
            treadmill = _interp_trace(
                entry,
                t_meso,
                source="treadmill",
                feature="speed_mm",
                method="previous",
                align_offset=True,
            )
            title = f"Subject {subject} | Session {session} | Task {task}"
            fig = _build_explorer_figure(
                meso_traces,
                roi_names,
                t_meso,
                pupil=pupil,
                treadmill=treadmill,
                title=title,
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
    args = parser.parse_args()

    try:
        generated = build_explorers_from_pickle(
            args.dataset_pickle,
            output_dir=args.output_dir,
            roi_limit=args.roi_limit,
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

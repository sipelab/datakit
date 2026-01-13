from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from datakit.experiment import ExperimentData  # type: ignore


def _write_stub_dataqueue(root: Path, subject: str, session: str) -> None:
    data_dir = root / "data" / f"sub-{subject}" / session
    data_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"20240101_sub-{subject}_{session}_task-widefield_dataqueue.csv"
    (data_dir / file_name).write_text("queue_elapsed\n0\n1\n", encoding="utf-8")


def test_experiment_data_accepts_multiple_directories(tmp_path):
    roots: list[Path] = []
    expected_subjects: set[str] = set()

    for idx in range(3):
        root = tmp_path / f"experiment_{idx}"
        subject = f"STREHAB{idx:02d}"
        session = f"ses-{idx + 1:02d}"
        _write_stub_dataqueue(root, subject, session)
        roots.append(root)
        expected_subjects.add(subject)

    dataset = ExperimentData(roots)

    assert len(dataset.manifest.entries) == len(expected_subjects)
    assert set(dataset.subjects) == expected_subjects
    assert dataset.data.shape[0] == len(expected_subjects)
    for entry in dataset.manifest.entries:
        assert Path(entry.path).is_absolute()

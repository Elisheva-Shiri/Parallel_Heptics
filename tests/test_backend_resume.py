import csv
import time
from pathlib import Path
from threading import Lock

import pytest
import typer

import backend


PAUSE_ROW = (0, 0, 0, 0)
BREAK_ROW = (-2, -2, -2, -2)
END_ROW = (-1, -1, -1, -1)
COMPARISON_ROW = (85, 0, 85, 2)


def _build_config(rows: list[tuple[int, int, int, int]]) -> backend.Configuration:
    return backend.Configuration(
        pairs=[
            backend.StiffnessPair(
                first=backend.StiffnessValue(value=row[0], finger_id=row[1]),
                second=backend.StiffnessValue(value=row[2], finger_id=row[3]),
            )
            for row in rows
        ]
    )


def _make_experiment_stub(config: backend.Configuration | None = None) -> backend.Experiment:
    experiment = object.__new__(backend.Experiment)
    experiment._config = config if config is not None else backend.Configuration(pairs=[])
    experiment._state_lock = Lock()
    experiment._pair_counter = 0
    return experiment


def _seed_experiment(root: Path, name: str, pair_count: int = 2) -> Path:
    folder = root / name
    folder.mkdir()
    with open(folder / "configuration.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for _ in range(pair_count):
            writer.writerow(COMPARISON_ROW)
        writer.writerow(END_ROW)
    return folder


def _write_answers(path: Path, pair_numbers: list[int]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "pair_number", "object_1_finger", "object_1_stiffness",
            "object_2_finger", "object_2_stiffness", "time_to_answer", "answer",
        ])
        for pn in pair_numbers:
            writer.writerow(["2025-05-01T10:00", pn, "thumb", 85, "index", 85, 1.0, 0])


# --- marker classification -------------------------------------------------


@pytest.mark.parametrize(
    "row, classifier, expected",
    [
        (END_ROW, backend.Experiment._is_end_marker, True),
        (BREAK_ROW, backend.Experiment._is_break_marker, True),
        (PAUSE_ROW, backend.Experiment._is_pause_marker, True),
        (COMPARISON_ROW, backend.Experiment._is_comparison_pair, True),
        (PAUSE_ROW, backend.Experiment._is_comparison_pair, False),
        (BREAK_ROW, backend.Experiment._is_comparison_pair, False),
        (END_ROW, backend.Experiment._is_comparison_pair, False),
    ],
)
def test_marker_classifiers(row, classifier, expected):
    pair = _build_config([row]).pairs[0]
    assert classifier(pair) is expected


# --- _config_index_for_pair_number -----------------------------------------


@pytest.fixture
def mixed_experiment() -> backend.Experiment:
    # comparison, pause, comparison, break, comparison, end
    return _make_experiment_stub(
        _build_config([
            COMPARISON_ROW,
            PAUSE_ROW,
            COMPARISON_ROW,
            BREAK_ROW,
            COMPARISON_ROW,
            END_ROW,
        ])
    )


@pytest.mark.parametrize("pair_number, expected_index", [(1, 0), (2, 2), (3, 4)])
def test_maps_pair_number_to_config_index(mixed_experiment, pair_number, expected_index):
    assert mixed_experiment._config_index_for_pair_number(pair_number) == expected_index


@pytest.mark.parametrize("pair_number", [0, -1, 4])
def test_rejects_out_of_range_pair_number(mixed_experiment, pair_number):
    with pytest.raises(ValueError):
        mixed_experiment._config_index_for_pair_number(pair_number)


# --- _find_latest_experiment ------------------------------------------------


def test_find_latest_returns_newest_subfolder_by_mtime(tmp_path):
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    time.sleep(0.05)
    newer.mkdir()

    assert backend._find_latest_experiment(tmp_path) == newer


def test_find_latest_ignores_files(tmp_path):
    (tmp_path / "loose.txt").write_text("x")
    target = tmp_path / "exp"
    target.mkdir()

    assert backend._find_latest_experiment(tmp_path) == target


def test_find_latest_returns_none_when_empty(tmp_path):
    assert backend._find_latest_experiment(tmp_path) is None


def test_find_latest_returns_none_when_missing():
    assert backend._find_latest_experiment(Path("/nonexistent_path_for_test")) is None


# --- _find_last_pair_number -------------------------------------------------


def test_find_last_pair_number_returns_max(tmp_path):
    (tmp_path / "pair_001").mkdir()
    (tmp_path / "pair_002").mkdir()
    (tmp_path / "pair_010").mkdir()
    (tmp_path / "answers.csv").write_text("")
    (tmp_path / "not_a_pair").mkdir()

    assert backend._find_last_pair_number(tmp_path) == 10


def test_find_last_pair_number_returns_none_when_no_pair_folders(tmp_path):
    assert backend._find_last_pair_number(tmp_path) is None


# --- _answer_already_recorded ----------------------------------------------


def test_answer_already_recorded_false_when_file_missing(tmp_path):
    assert backend._answer_already_recorded(tmp_path / "answers.csv", 1) is False


@pytest.mark.parametrize(
    "recorded, query, expected",
    [
        ([1, 2, 3], 2, True),
        ([1, 2], 5, False),
    ],
)
def test_answer_already_recorded_lookup(tmp_path, recorded, query, expected):
    answers = tmp_path / "answers.csv"
    _write_answers(answers, recorded)
    assert backend._answer_already_recorded(answers, query) is expected


# --- _next_video_filenames -------------------------------------------------


@pytest.fixture
def first_pair_experiment(tmp_path):
    experiment = _make_experiment_stub()
    experiment._path = tmp_path
    experiment._pair_counter = 1
    (tmp_path / "pair_001").mkdir()
    return experiment


def test_uses_default_video_names_for_first_segment(first_pair_experiment):
    pair_path = first_pair_experiment._path / "pair_001"
    top, side = first_pair_experiment._next_video_filenames()

    assert top == pair_path / "top_camera.mp4"
    assert side == pair_path / "side_camera.mp4"


def test_picks_unique_resume_segment_when_files_exist(first_pair_experiment):
    pair_path = first_pair_experiment._path / "pair_001"
    (pair_path / "top_camera.mp4").write_bytes(b"")
    (pair_path / "side_camera.mp4").write_bytes(b"")

    top, side = first_pair_experiment._next_video_filenames()

    assert top == pair_path / "top_camera_resume_001.mp4"
    assert side == pair_path / "side_camera_resume_001.mp4"


# --- _resolve_resume_context -----------------------------------------------


def test_resolve_resume_context_defaults_to_last_pair_folder(tmp_path):
    folder = _seed_experiment(tmp_path, "exp", pair_count=3)
    (folder / "pair_001").mkdir()
    (folder / "pair_002").mkdir()

    path, config, pair_number = backend._resolve_resume_context(tmp_path, None)

    assert path == folder
    assert len(config.pairs) == 4
    assert pair_number == 2


def test_resolve_resume_context_defaults_to_pair_one_when_no_pair_folders(tmp_path):
    _seed_experiment(tmp_path, "exp")
    _, _, pair_number = backend._resolve_resume_context(tmp_path, None)
    assert pair_number == 1


def test_resolve_resume_context_honours_from_pair_override(tmp_path):
    folder = _seed_experiment(tmp_path, "exp", pair_count=3)
    (folder / "pair_001").mkdir()
    (folder / "pair_002").mkdir()
    (folder / "pair_003").mkdir()

    _, _, pair_number = backend._resolve_resume_context(tmp_path, 3)

    assert pair_number == 3


def test_resolve_resume_context_allows_redoing_last_started_pair(tmp_path):
    folder = _seed_experiment(tmp_path, "exp", pair_count=3)
    (folder / "pair_001").mkdir()
    (folder / "pair_002").mkdir()

    _, _, pair_number = backend._resolve_resume_context(tmp_path, 2)

    assert pair_number == 2


def test_resolve_resume_context_rejects_future_from_pair(tmp_path):
    folder = _seed_experiment(tmp_path, "exp", pair_count=3)
    (folder / "pair_001").mkdir()
    (folder / "pair_002").mkdir()

    with pytest.raises(typer.BadParameter, match="higher than the latest started pair"):
        backend._resolve_resume_context(tmp_path, 3)


def test_resolve_resume_context_rejects_future_from_pair_when_none_started(tmp_path):
    _seed_experiment(tmp_path, "exp", pair_count=3)

    with pytest.raises(typer.BadParameter, match="no pair_NNN folders exist"):
        backend._resolve_resume_context(tmp_path, 2)


def test_resolve_resume_context_allows_pair_one_when_none_started(tmp_path):
    _seed_experiment(tmp_path, "exp", pair_count=3)

    _, _, pair_number = backend._resolve_resume_context(tmp_path, 1)

    assert pair_number == 1


def test_resolve_resume_context_rejects_non_positive_from_pair(tmp_path):
    _seed_experiment(tmp_path, "exp")
    with pytest.raises(typer.BadParameter):
        backend._resolve_resume_context(tmp_path, 0)


def test_resolve_resume_context_raises_when_no_experiment_folder(tmp_path):
    with pytest.raises(typer.BadParameter):
        backend._resolve_resume_context(tmp_path, None)


def test_resolve_resume_context_raises_when_configuration_missing(tmp_path):
    (tmp_path / "exp").mkdir()
    with pytest.raises(typer.BadParameter):
        backend._resolve_resume_context(tmp_path, None)

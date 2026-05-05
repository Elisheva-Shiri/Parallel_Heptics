import csv
from queue import Queue
from threading import Lock
from types import SimpleNamespace

import pytest

import backend
import moderator_control
from structures import ControlAction, ExperimentControl, ExperimentState


@pytest.fixture
def experiment(tmp_path):
    stub = object.__new__(backend.Experiment)
    stub._path = tmp_path
    stub._state = ExperimentState.COMPARISON
    stub._state_lock = Lock()
    stub._control_queue = Queue()
    stub._moderator_pause_started_at = None
    stub._moderator_pause_state_when_started = None
    stub._is_interacting = True
    stub._virtual_object = SimpleNamespace(is_interacting=True)
    stub._manual_interaction_enabled = True
    return stub


# --- CLI parsing ------------------------------------------------------------


def test_toggle_interaction_flag_uses_moderator_action():
    args = moderator_control.build_parser().parse_args(["--toggle_interaction"])
    control = moderator_control.control_from_args(args)

    assert control.moderatorAction == ControlAction.TOGGLE_INTERACTION


def test_rejects_multiple_actions():
    parser = moderator_control.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--toggle-interaction", "--pause"])


# --- Backend behaviour ------------------------------------------------------


def test_pause_toggle_logs_start_and_finish(experiment):
    started = experiment._toggle_moderator_pause()
    assert started["ok"]
    assert experiment._is_interacting is False
    assert experiment._virtual_object.is_interacting is False

    finished = experiment._toggle_moderator_pause()
    assert finished["ok"]

    with open(experiment._path / "moderator_pauses.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["state_when_started"] == "COMPARISON"
    assert row["timestamp_start"] != ""
    assert row["timestamp_finish"] != ""
    assert float(row["pause_duration_seconds"]) >= 0.0


def test_finish_break_is_ignored_outside_break_state(experiment):
    response = experiment._handle_control_message(
        ExperimentControl(moderatorAction=ControlAction.FINISH_BREAK)
    )

    assert response["ok"] is False
    assert experiment._control_queue.empty()


def test_toggle_interaction_is_available_with_side_camera_enabled(experiment):
    experiment._manual_interaction_enabled = False

    response = experiment._handle_control_message(
        ExperimentControl(moderatorAction=ControlAction.TOGGLE_INTERACTION)
    )

    assert response["ok"] is True
    assert experiment._is_interacting is False

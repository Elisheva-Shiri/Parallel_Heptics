from threading import Lock

import pytest

import backend


class DummyCapture:
    def __init__(self):
        self.release_count = 0

    def release(self):
        self.release_count += 1


class DummySocket:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class DummyVision:
    def __init__(self):
        self.cleanup_count = 0

    def cleanup(self):
        self.cleanup_count += 1


@pytest.fixture
def experiment() -> backend.Experiment:
    stub = object.__new__(backend.Experiment)
    stub._running = True
    stub._camera_lock = Lock()
    stub._top_camera_capture = None
    stub._side_camera_capture = None
    return stub


@pytest.fixture
def cleanup_experiment(experiment) -> backend.Experiment:
    experiment._cleanup_lock = Lock()
    experiment._cleaned_up = False
    experiment._state_lock = Lock()
    experiment._moderator_pause_started_at = None
    experiment._moderator_pause_state_when_started = None
    experiment._state = backend.ExperimentState.COMPARISON
    experiment._top_writer = None
    experiment._side_writer = None
    experiment._frontend_socket = DummySocket()
    experiment._listening_socket = DummySocket()
    experiment._hardware_socket = DummySocket()
    experiment._vision = DummyVision()
    experiment._top_camera_capture = DummyCapture()
    return experiment


def test_request_shutdown_stops_loops_and_releases_camera_captures(experiment):
    top = DummyCapture()
    side = DummyCapture()
    experiment._top_camera_capture = top
    experiment._side_camera_capture = side

    experiment.request_shutdown()

    assert experiment.is_running() is False
    assert top.release_count == 1
    assert side.release_count == 1
    assert experiment._top_camera_capture is None
    assert experiment._side_camera_capture is None


def test_cleanup_is_idempotent_and_closes_resources_once(cleanup_experiment, monkeypatch):
    destroy_all_calls = []
    monkeypatch.setattr(backend.cv2, "destroyAllWindows", lambda: destroy_all_calls.append(True))
    capture = cleanup_experiment._top_camera_capture

    cleanup_experiment.cleanup()
    cleanup_experiment.cleanup()

    assert cleanup_experiment.is_running() is False
    assert capture.release_count == 1
    assert cleanup_experiment._frontend_socket.close_count == 1
    assert cleanup_experiment._listening_socket.close_count == 1
    assert cleanup_experiment._hardware_socket.close_count == 1
    assert cleanup_experiment._vision.cleanup_count == 1
    assert len(destroy_all_calls) == 1

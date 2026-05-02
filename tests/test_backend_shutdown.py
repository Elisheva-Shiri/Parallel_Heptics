from threading import Lock

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


def _make_experiment_stub() -> backend.Experiment:
    experiment = object.__new__(backend.Experiment)
    experiment._running = True
    experiment._camera_lock = Lock()
    experiment._top_camera_capture = None
    experiment._side_camera_capture = None
    return experiment


def test_request_shutdown_stops_loops_and_releases_camera_captures():
    experiment = _make_experiment_stub()
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


def test_cleanup_is_idempotent_and_closes_resources_once(monkeypatch):
    experiment = _make_experiment_stub()
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
    capture = DummyCapture()
    experiment._top_camera_capture = capture
    destroy_all_calls = []
    monkeypatch.setattr(backend.cv2, "destroyAllWindows", lambda: destroy_all_calls.append(True))

    experiment.cleanup()
    experiment.cleanup()

    assert experiment.is_running() is False
    assert capture.release_count == 1
    assert experiment._frontend_socket.close_count == 1
    assert experiment._listening_socket.close_count == 1
    assert experiment._hardware_socket.close_count == 1
    assert experiment._vision.cleanup_count == 1
    assert len(destroy_all_calls) == 1

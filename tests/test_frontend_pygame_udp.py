import pytest

from frontend_pygame import _recv_latest_datagram, _should_show_cycle_counter


class FakeUdpSocket:
    def __init__(self, datagrams: list[bytes], timeout: float | None = None):
        self._datagrams = list(datagrams)
        self._timeout = timeout
        self.blocking_values: list[bool] = []
        self.timeout_values: list[float | None] = []

    def recvfrom(self, _max_bytes: int):
        if not self._datagrams:
            raise BlockingIOError
        return self._datagrams.pop(0), ("127.0.0.1", 12346)

    def gettimeout(self):
        return self._timeout

    def setblocking(self, enabled: bool):
        self.blocking_values.append(enabled)
        self._timeout = None if enabled else 0.0

    def settimeout(self, timeout: float | None):
        self.timeout_values.append(timeout)
        self._timeout = timeout


def test_recv_latest_datagram_discards_queued_stale_packets():
    fake_socket = FakeUdpSocket([b"old", b"newer", b"newest"])

    assert _recv_latest_datagram(fake_socket, 4096) == b"newest"
    assert fake_socket.blocking_values == [False]
    assert fake_socket.timeout_values == [None]


def test_recv_latest_datagram_preserves_existing_timeout():
    fake_socket = FakeUdpSocket([b"only"], timeout=0.25)

    assert _recv_latest_datagram(fake_socket, 4096) == b"only"
    assert fake_socket.timeout_values == [0.25]


@pytest.mark.parametrize(
    "iterations, expected",
    [(1, False), (2, True)],
    ids=["single_hidden", "multiple_shown"],
)
def test_cycle_counter_visibility(iterations, expected):
    assert _should_show_cycle_counter(iterations) is expected

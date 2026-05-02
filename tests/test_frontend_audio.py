from array import array

import pytest

from frontend_pygame import _make_white_noise_buffer
from structures import ExperimentPacket


def _samples_from(buffer: bytes) -> array:
    samples = array("h")
    samples.frombytes(buffer)
    return samples


_BASE_PACKET = (
    '"stateData":{"state":1,"pauseTime":0},'
    '"landmarks":[{"x":0.25,"z":0.75}],'
    '"trackingObject":{"x":0.5,"z":0.4,"size":40.0,"isInteracting":true,'
    '"progress":0.25,"returnProgress":0.5,"cycleCount":1,'
    '"targetCycleCount":2,"pairIndex":0}'
)


def test_white_noise_buffer_is_deterministic_16_bit_mono_pcm():
    sample_rate = 100
    seconds = 0.5

    kwargs = dict(sample_rate=sample_rate, seconds=seconds, amplitude=0.5, seed=123)
    first = _make_white_noise_buffer(**kwargs)
    second = _make_white_noise_buffer(**kwargs)

    assert first == second
    assert len(first) == int(sample_rate * seconds) * 2


def test_white_noise_buffer_respects_amplitude():
    amplitude = 0.25
    samples = _samples_from(
        _make_white_noise_buffer(sample_rate=100, seconds=1.0, amplitude=amplitude, seed=456)
    )

    peak = int(32767 * amplitude)
    assert max(samples) <= peak
    assert min(samples) >= -peak
    assert len(set(samples)) > 1


def test_white_noise_buffer_is_softened_for_ambient_masking():
    samples = _samples_from(
        _make_white_noise_buffer(sample_rate=1000, seconds=1.0, amplitude=1.0, seed=789)
    )

    mean_adjacent_delta = sum(
        abs(b - a) for a, b in zip(samples, samples[1:])
    ) / (len(samples) - 1)

    assert mean_adjacent_delta < 32767 * 0.45


@pytest.mark.parametrize(
    "extra, expected_white_noise, expected_debug",
    [
        ("", False, True),
        (',"playWhiteNoise":true,"isDebug":false', True, False),
    ],
    ids=["defaults", "explicit_flags"],
)
def test_experiment_packet_flags(extra, expected_white_noise, expected_debug):
    packet = ExperimentPacket.model_validate_json("{" + _BASE_PACKET + extra + "}")

    assert packet.playWhiteNoise is expected_white_noise
    assert packet.isDebug is expected_debug

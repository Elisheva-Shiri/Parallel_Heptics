from array import array
import unittest

from frontend_pygame import _make_white_noise_buffer
from structures import ExperimentPacket


class FrontendAudioTests(unittest.TestCase):
    def test_white_noise_buffer_is_deterministic_16_bit_mono_pcm(self):
        sample_rate = 100
        seconds = 0.5

        first = _make_white_noise_buffer(sample_rate=sample_rate, seconds=seconds, amplitude=0.5, seed=123)
        second = _make_white_noise_buffer(sample_rate=sample_rate, seconds=seconds, amplitude=0.5, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), int(sample_rate * seconds) * 2)

    def test_white_noise_buffer_respects_amplitude(self):
        buffer = _make_white_noise_buffer(sample_rate=100, seconds=1.0, amplitude=0.25, seed=456)
        samples = array("h")
        samples.frombytes(buffer)

        self.assertLessEqual(max(samples), int(32767 * 0.25))
        self.assertGreaterEqual(min(samples), -int(32767 * 0.25))
        self.assertGreater(len(set(samples)), 1)

    def test_experiment_packet_white_noise_flag_defaults_to_false(self):
        packet = ExperimentPacket.model_validate_json(
            '{"stateData":{"state":1,"pauseTime":0},'
            '"landmarks":[{"x":0.25,"z":0.75}],'
            '"trackingObject":{"x":0.5,"z":0.4,"size":40.0,"isPinched":true,'
            '"progress":0.25,"returnProgress":0.5,"cycleCount":1,'
            '"targetCycleCount":2,"pairIndex":0}}'
        )

        self.assertFalse(packet.playWhiteNoise)
        self.assertTrue(packet.isDebug)

    def test_experiment_packet_accepts_backend_flags(self):
        packet = ExperimentPacket.model_validate_json(
            '{"stateData":{"state":1,"pauseTime":0},'
            '"landmarks":[{"x":0.25,"z":0.75}],'
            '"trackingObject":{"x":0.5,"z":0.4,"size":40.0,"isPinched":true,'
            '"progress":0.25,"returnProgress":0.5,"cycleCount":1,'
            '"targetCycleCount":2,"pairIndex":0},'
            '"playWhiteNoise":true,'
            '"isDebug":false}'
        )

        self.assertTrue(packet.playWhiteNoise)
        self.assertFalse(packet.isDebug)


if __name__ == "__main__":
    unittest.main()

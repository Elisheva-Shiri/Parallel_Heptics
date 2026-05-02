import builtins
import io
import unittest

import typer

import backend


class BackendCliConfigTests(unittest.TestCase):
    def tearDown(self):
        backend.set_is_debug(True)

    def test_debug_print_gate_controls_global_print(self):
        output = io.StringIO()

        backend.set_is_debug(False)
        builtins.print("hidden", file=output)
        self.assertEqual("", output.getvalue())

        backend.set_is_debug(True)
        builtins.print("visible", file=output)
        self.assertEqual("visible\n", output.getvalue())

    def test_parse_repeated_finger_color_flags(self):
        colors = backend._parse_finger_color_flags(["thumb=red", "index=blue"])

        self.assertEqual({"thumb": "red", "index": "blue"}, colors)

    def test_rejects_duplicate_finger_color_values(self):
        with self.assertRaises(typer.BadParameter):
            backend._parse_finger_color_flags(["thumb=red", "index=red"])

    def test_comparison_pair_finger_flag_rejects_invalid_choice(self):
        with self.assertRaises(typer.BadParameter):
            backend._resolve_pair_finger("comparison", backend.FingerChoice.INDEX)

    def test_single_finger_pair_finger_flag_accepts_any_finger(self):
        self.assertEqual(
            "thumb",
            backend._resolve_pair_finger("single_finger", backend.FingerChoice.THUMB),
        )


if __name__ == "__main__":
    unittest.main()

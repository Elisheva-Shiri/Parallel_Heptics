from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import video_crop_gui as gui


class VideoCropGuiHelperTests(unittest.TestCase):
    def test_rotation_options_are_requested_45_degree_set(self):
        self.assertEqual(gui.ROTATION_OPTIONS, (0, 45, 90, 135, 180, -135, -90, -45))
        for angle in gui.ROTATION_OPTIONS:
            self.assertEqual(gui.validate_rotation(angle), angle)
        with self.assertRaises(ValueError):
            gui.validate_rotation(30)

    def test_default_output_dir_is_pair_crop_videos(self):
        pair_dir = Path("Data") / "experiment" / "pair_001"
        self.assertEqual(gui.default_output_dir(pair_dir), pair_dir / "crop_videos")

    def test_guess_default_data_root_finds_local_data_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "Data"
            data.mkdir()
            self.assertEqual(gui.guess_default_data_root(root), data)

    def test_build_video_map_keeps_all_videos_with_unique_keys(self):
        videos = [
            Path("side_camera.mp4"),
            Path("top_camera.mp4"),
            Path("top_camera.avi"),
        ]
        video_map = gui.build_video_map(videos)
        self.assertEqual(list(video_map), ["side_camera", "top_camera", "top_camera_avi"])
        self.assertEqual(video_map["top_camera_avi"], Path("top_camera.avi"))

    def test_video_files_in_pair_accepts_direct_video_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "top_camera.mp4").write_bytes(b"not a real video for discovery only")
            (folder / "side_camera.mp4").write_bytes(b"not a real video for discovery only")
            (folder / "tracking.csv").write_text("x,y\n", encoding="utf-8")
            self.assertEqual([p.name for p in gui.video_files_in_pair(folder)], ["side_camera.mp4", "top_camera.mp4"])

    def test_safe_output_filename_contains_provenance(self):
        name = gui.make_output_filename(
            "2026 experiment",
            "pair_001",
            "side_camera",
            "Hand / Thimble",
            -45,
        )
        self.assertEqual(name, "2026_experiment_pair_001_side_camera_Hand_Thimble_rotm45.mp4")

    def test_crop_frame_clamps_rectangle(self):
        frame = np.arange(10 * 20 * 3, dtype=np.uint8).reshape((10, 20, 3))
        cropped = gui.crop_frame(frame, (5, 2, 6, 4))
        self.assertEqual(cropped.shape, (4, 6, 3))

    def test_rotate_frame_bound_preserves_nonzero_image(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[:, 5:15] = 255
        rotated = gui.rotate_frame_bound(frame, 45)
        self.assertGreater(rotated.shape[0], 10)
        self.assertGreater(rotated.shape[1], 20)
        self.assertGreater(int(rotated.sum()), 0)

    def test_export_crop_video_writes_derived_file_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "source.mp4"
            out = tmp_path / "crop_videos" / "derived.mp4"
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 30))
            self.assertTrue(writer.isOpened())
            for i in range(5):
                frame = np.zeros((30, 40, 3), dtype=np.uint8)
                frame[5:20, 10:30] = (0, 255, 0)
                frame[:, :, 0] = i * 20
                writer.write(frame)
            writer.release()

            gui.export_crop_video(src, (10, 5, 20, 15), 90, out)

            self.assertTrue(src.exists(), "source video must remain in place")
            self.assertTrue(out.exists(), "derived crop video should be created")
            cap = cv2.VideoCapture(str(out))
            try:
                self.assertTrue(cap.isOpened())
                self.assertGreater(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            finally:
                cap.release()

    def test_script_has_no_destructive_original_file_operations(self):
        source = Path("video_crop_gui.py").read_text(encoding="utf-8")
        dangerous_patterns = [
            "os.remove(",
            "Path.unlink(",
            ".unlink(",
            "shutil.move(",
            "shutil.rmtree(",
            "os.rename(",
            "os.replace(",
        ]
        for pattern in dangerous_patterns:
            self.assertNotIn(pattern, source)


if __name__ == "__main__":
    unittest.main()

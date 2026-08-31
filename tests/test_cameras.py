from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter, sleep
from zlib import crc32

import cv2

# This file is a manual hardware utility, not a pytest module.
__test__ = False


BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


@dataclass
class FpsProbeResult:
    camera_index: int
    requested_fps: int
    opened: bool
    captured_frames: int = 0
    measured_fps: float = 0.0
    changed_frames: int = 0
    changed_fps: float = 0.0
    reported_fps: float = 0.0
    actual_width: int = 0
    actual_height: int = 0


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers and ranges, e.g. '0,2-4'."""
    numbers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            numbers.extend(range(start, end + step, step))
        else:
            numbers.append(int(part))
    return list(dict.fromkeys(numbers))


def open_camera(camera_index: int, backend: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(camera_index, backend)


def configure_camera(
    cap: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    fps: int,
    use_mjpg: bool,
) -> None:
    # Many USB cameras only expose their high-FPS modes when MJPG is selected.
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def measure_capture_fps(
    cap: cv2.VideoCapture,
    *,
    duration_seconds: float,
    warmup_frames: int,
) -> tuple[int, float, int, float]:
    for _ in range(warmup_frames):
        cap.read()

    frame_count = 0
    changed_frame_count = 0
    previous_signature: int | None = None
    started_at = perf_counter()
    deadline = started_at + duration_seconds

    while perf_counter() < deadline:
        ok, frame = cap.read()
        if ok:
            frame_count += 1
            # Some drivers return the most recent buffered frame immediately
            # instead of blocking for a new sensor frame. Track exact sampled
            # frame changes so repeated reads are visible in the report.
            signature = crc32(frame[::16, ::16].tobytes())
            if signature != previous_signature:
                changed_frame_count += 1
                previous_signature = signature

    elapsed = max(perf_counter() - started_at, 0.001)
    return frame_count, frame_count / elapsed, changed_frame_count, changed_frame_count / elapsed


def probe_camera_fps(
    camera_index: int,
    requested_fps: int,
    *,
    backend: int,
    width: int,
    height: int,
    duration_seconds: float,
    warmup_frames: int,
    use_mjpg: bool,
) -> FpsProbeResult:
    cap = open_camera(camera_index, backend)
    if not cap.isOpened():
        cap.release()
        return FpsProbeResult(camera_index=camera_index, requested_fps=requested_fps, opened=False)

    try:
        configure_camera(cap, width=width, height=height, fps=requested_fps, use_mjpg=use_mjpg)
        sleep(0.2)
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        captured_frames, measured_fps, changed_frames, changed_fps = measure_capture_fps(
            cap,
            duration_seconds=duration_seconds,
            warmup_frames=warmup_frames,
        )
        return FpsProbeResult(
            camera_index=camera_index,
            requested_fps=requested_fps,
            opened=True,
            captured_frames=captured_frames,
            measured_fps=measured_fps,
            changed_frames=changed_frames,
            changed_fps=changed_fps,
            reported_fps=reported_fps,
            actual_width=actual_width,
            actual_height=actual_height,
        )
    finally:
        cap.release()


def measure_open_camera_fps(
    camera_index: int,
    requested_fps: int,
    cap: cv2.VideoCapture,
    *,
    duration_seconds: float,
    warmup_frames: int,
) -> FpsProbeResult:
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    captured_frames, measured_fps, changed_frames, changed_fps = measure_capture_fps(
        cap,
        duration_seconds=duration_seconds,
        warmup_frames=warmup_frames,
    )
    return FpsProbeResult(
        camera_index=camera_index,
        requested_fps=requested_fps,
        opened=True,
        captured_frames=captured_frames,
        measured_fps=measured_fps,
        changed_frames=changed_frames,
        changed_fps=changed_fps,
        reported_fps=reported_fps,
        actual_width=actual_width,
        actual_height=actual_height,
    )


def probe_cameras_fps_simultaneously(
    camera_indices: list[int],
    requested_fps: int,
    *,
    backend: int,
    width: int,
    height: int,
    duration_seconds: float,
    warmup_frames: int,
    use_mjpg: bool,
) -> list[FpsProbeResult]:
    caps: dict[int, cv2.VideoCapture] = {}
    results: list[FpsProbeResult] = []

    try:
        for camera_index in camera_indices:
            cap = open_camera(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                results.append(
                    FpsProbeResult(
                        camera_index=camera_index,
                        requested_fps=requested_fps,
                        opened=False,
                    )
                )
                continue

            configure_camera(
                cap,
                width=width,
                height=height,
                fps=requested_fps,
                use_mjpg=use_mjpg,
            )
            caps[camera_index] = cap

        if not caps:
            return sorted(results, key=lambda result: result.camera_index)

        sleep(0.2)
        with ThreadPoolExecutor(max_workers=len(caps)) as executor:
            futures = [
                executor.submit(
                    measure_open_camera_fps,
                    camera_index,
                    requested_fps,
                    cap,
                    duration_seconds=duration_seconds,
                    warmup_frames=warmup_frames,
                )
                for camera_index, cap in caps.items()
            ]
            results.extend(future.result() for future in futures)

        return sorted(results, key=lambda result: result.camera_index)
    finally:
        for cap in caps.values():
            cap.release()


def print_probe_results(results: list[FpsProbeResult]) -> None:
    print("\nCamera FPS probe results")
    print("camera  requested  reported  read-fps  changed-fps  frames  changed  actual-size")
    print("------  ---------  --------  --------  -----------  ------  -------  -----------")
    for result in results:
        if not result.opened:
            print(f"{result.camera_index:>6}  {result.requested_fps:>9}  {'FAILED TO OPEN':>35}")
            continue
        print(
            f"{result.camera_index:>6}  "
            f"{result.requested_fps:>9}  "
            f"{result.reported_fps:>8.2f}  "
            f"{result.measured_fps:>8.2f}  "
            f"{result.changed_fps:>11.2f}  "
            f"{result.captured_frames:>6}  "
            f"{result.changed_frames:>7}  "
            f"{result.actual_width}x{result.actual_height}"
        )

    print("\nBest observed FPS per camera")
    for camera_index in sorted({result.camera_index for result in results}):
        opened_results = [
            result
            for result in results
            if result.camera_index == camera_index and result.opened and result.captured_frames > 0
        ]
        if not opened_results:
            print(f"Camera {camera_index}: not available")
            continue
        best = max(opened_results, key=lambda result: result.changed_fps or result.measured_fps)
        print(
            f"Camera {camera_index}: {best.changed_fps:.2f} changed-frame FPS "
            f"({best.measured_fps:.2f} read FPS) "
            f"(requested {best.requested_fps}, driver reported {best.reported_fps:.2f}, "
            f"size {best.actual_width}x{best.actual_height})"
        )


def preview_camera(
    camera_index: int,
    *,
    backend: int,
    width: int,
    height: int,
    fps: int,
    use_mjpg: bool,
) -> bool:
    print(f"Testing camera at index {camera_index}...")
    cap = open_camera(camera_index, backend)

    if not cap.isOpened():
        cap.release()
        print(f"Failed to open camera at index {camera_index}")
        return False

    configure_camera(cap, width=width, height=height, fps=fps, use_mjpg=use_mjpg)
    print(
        f"Camera {camera_index} opened: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"driver reports {cap.get(cv2.CAP_PROP_FPS):.2f} FPS. Press 'q' to close the feed."
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_index}. Retrying...")
            continue
        cv2.imshow(f"Camera {camera_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


def run_max_fps_probe(args: argparse.Namespace) -> None:
    results: list[FpsProbeResult] = []
    for camera_index in parse_int_list(args.indices):
        for fps in parse_int_list(args.fps_candidates):
            print(f"Probing camera {camera_index} at requested {fps} FPS...")
            results.append(
                probe_camera_fps(
                    camera_index,
                    fps,
                    backend=BACKENDS[args.backend],
                    width=args.width,
                    height=args.height,
                    duration_seconds=args.duration,
                    warmup_frames=args.warmup_frames,
                    use_mjpg=not args.no_mjpg,
                )
            )
    print_probe_results(results)


def run_simultaneous_max_fps_probe(args: argparse.Namespace) -> None:
    results: list[FpsProbeResult] = []
    camera_indices = parse_int_list(args.indices)
    for fps in parse_int_list(args.fps_candidates):
        print(f"Probing cameras {camera_indices} simultaneously at requested {fps} FPS...")
        results.extend(
            probe_cameras_fps_simultaneously(
                camera_indices,
                fps,
                backend=BACKENDS[args.backend],
                width=args.width,
                height=args.height,
                duration_seconds=args.duration,
                warmup_frames=args.warmup_frames,
                use_mjpg=not args.no_mjpg,
            )
        )
    print_probe_results(results)


def run_preview(args: argparse.Namespace) -> None:
    for camera_index in parse_int_list(args.indices):
        if preview_camera(
            camera_index,
            backend=BACKENDS[args.backend],
            width=args.width,
            height=args.height,
            fps=args.preview_fps,
            use_mjpg=not args.no_mjpg,
        ):
            print(f"Camera {camera_index} seems functional. Use this index in your application.")
        else:
            print(f"Camera {camera_index} is not functional.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual camera preview and FPS probe utility.",
    )
    parser.add_argument(
        "--indices",
        default="0-2",
        help="Camera indices to test. Supports comma-separated values and ranges, e.g. '0,2' or '0-3'.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="dshow",
        help="OpenCV capture backend to use.",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument(
        "--no-mjpg",
        action="store_true",
        help="Do not request MJPG format before setting FPS.",
    )
    parser.add_argument(
        "--max-fps",
        action="store_true",
        help="Probe candidate FPS settings and report the highest measured rate for each camera.",
    )
    parser.add_argument(
        "--simultaneous",
        action="store_true",
        help="With --max-fps, open all selected cameras together and measure them at the same time.",
    )
    parser.add_argument(
        "--fps-candidates",
        default="24,25,30,50,60,120,144,240,480",
        help="FPS values to request when --max-fps is used.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Seconds to measure each FPS candidate.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Frames to discard before each FPS measurement.",
    )
    parser.add_argument(
        "--preview-fps",
        type=int,
        default=30,
        help="FPS to request in preview mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_fps and args.simultaneous:
        run_simultaneous_max_fps_probe(args)
    elif args.max_fps:
        run_max_fps_probe(args)
    else:
        run_preview(args)


if __name__ == "__main__":
    main()

"""One-shot moderator command utility for a running experiment backend.

Examples:
    uv run moderator_control.py --toggle-interaction
    uv run moderator_control.py --toggle_interaction
    uv run moderator_control.py -i
    uv run moderator_control.py --toggle-break
    uv run moderator_control.py -b
    uv run moderator_control.py --pause
    uv run moderator_control.py -p
"""

from __future__ import annotations

import argparse
import json
import socket
from typing import Sequence

from consts import BACKEND_PORT
from structures import ControlAction, ExperimentControl


DEFAULT_HOST = "localhost"
DEFAULT_TIMEOUT_SECONDS = 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send exactly one moderator command to the running Parallel Heptics backend. "
            "The backend owns whether the command is valid for the current experiment state."
        )
    )
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "-i",
        "--toggle-interaction",
        "--toggle_interaction",
        action="store_true",
        help="Toggle manual interaction/is_interacting during comparison when manual interaction mode is active.",
    )
    action_group.add_argument(
        "-b",
        "--toggle-break",
        "--toggle_break",
        action="store_true",
        help="Finish the current configured break and continue the experiment.",
    )
    action_group.add_argument(
        "-p",
        "--pause",
        action="store_true",
        help="Toggle moderator pause. Run again to resume; start and finish are logged.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Backend host. Default: {DEFAULT_HOST}.")
    parser.add_argument("--port", type=int, default=BACKEND_PORT, help=f"Backend TCP port. Default: {BACKEND_PORT}.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Connection/read timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    return parser


def control_from_args(args: argparse.Namespace) -> ExperimentControl:
    if args.toggle_interaction:
        return ExperimentControl(moderatorAction=ControlAction.TOGGLE_INTERACTION)
    if args.toggle_break:
        return ExperimentControl(moderatorAction=ControlAction.FINISH_BREAK)
    if args.pause:
        return ExperimentControl(moderatorAction=ControlAction.TOGGLE_PAUSE)

    raise ValueError("No moderator command was selected")


def send_control(
    control: ExperimentControl,
    host: str = DEFAULT_HOST,
    port: int = BACKEND_PORT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    payload = (control.model_dump_json() + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(payload)
        response = sock.recv(4096)

    if not response:
        return {"ok": False, "message": "Backend closed connection without an acknowledgement"}

    try:
        return json.loads(response.decode("utf-8").strip())
    except json.JSONDecodeError:
        return {
            "ok": False,
            "message": f"Backend returned a non-JSON acknowledgement: {response!r}",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    control = control_from_args(args)

    try:
        response = send_control(control, host=args.host, port=args.port, timeout=args.timeout)
    except OSError as ex:
        parser.exit(2, f"Failed to send moderator command: {ex}\n")

    print(response.get("message", response))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

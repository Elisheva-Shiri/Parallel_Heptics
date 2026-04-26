# Frontend Unity

Efficient Unity PCVR frontend for the Parallel Heptics backend.

## Switching between pygame and Unity

The Python backend is unchanged. Both pygame and Unity use the same protocol:

- UDP `ExperimentPacket` from backend to frontend on port `12346`
- TCP `ExperimentControl` from frontend to backend on port `12344`

Run exactly one frontend at a time:

1. Pygame: run `run_pygame.bat` / `frontend_pygame.py`.
2. Unity: open this project, open `Assets/Scenes/FrontendUnity.unity`, press Play, and use Quest Link for PCVR.

Unity listens on the pygame port (`12346`) by default so it can replace pygame without a backend architecture change.

## Unity frontend design

- Receives UDP on a background thread.
- Stores only the latest packet; old packets are dropped to avoid visual latency/backlog.
- Parses one latest JSON packet per Unity frame.
- Reuses scene objects/materials; no per-frame instantiate/destroy.
- Uses backend-provided landmarks for left/right hold-to-answer behavior.

## Quest Link / OpenXR notes

For headset use, set Meta Quest Link as the active OpenXR runtime in the Meta Quest PC app, then run this Unity project in Play Mode or as a Windows build.

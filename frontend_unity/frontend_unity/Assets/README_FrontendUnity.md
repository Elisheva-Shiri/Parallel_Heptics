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
- Renders the experiment as a world-locked horizontal tabletop surface rather than a head-locked display.
- Defaults to a large 2 m × 2 m tabletop visualization area so the experiment is readable in the headset. The backend coordinate system is still normalized and unchanged.

## Tabletop calibration

The first calibration implementation is intentionally simple and local to Unity. It does not change the Python backend or pygame.

In Unity Play Mode, align the virtual 20 cm surface to the real table area:

- Arrow keys: slide the surface across the table plane.
- Shift + arrow keys: slide faster.
- PageUp / PageDown: raise/lower the surface height.
- Q / E: yaw-rotate the surface on the table.
- Hold `[` / `]`: shrink/grow the surface uniformly.
- Click `[` / `]`: shrink/grow in small steps.
- Enter: save calibration to local Unity `PlayerPrefs` and stop calibration movement.
- R: restart calibration movement after it has been saved/locked.

During calibration, the black panel becomes nearly transparent and AR Foundation passthrough is enabled so the real table is visible behind the virtual working area. Pressing Enter saves the calibration, disables calibration movement, restores the opaque black experiment surface, and turns passthrough off.

After saving, the tabletop transform reloads on the next Play Mode run and starts locked. The surface is a root-level world object, not a child of the XR Origin/camera, so moving the headset should change the viewpoint but should not drag the experiment area with the head.

Limit: Quest/OpenXR tracking can still shift if the runtime loses tracking or recenters. This first pass does not use spatial anchors or camera/fiducial tracking; those can be added later if the physical alignment needs persistence across runtime recentering or sessions.

## Quest Link / OpenXR notes

For headset use, set Meta Quest Link as the active OpenXR runtime in the Meta Quest PC app, then run this Unity project in Play Mode or as a Windows build.

The project includes Unity's Meta OpenXR package and the menu item `Parallel Heptics > Configure OpenXR for Quest Link` enables the Standalone OpenXR loader plus Meta Quest passthrough features. Quest Link passthrough also requires Meta's PC runtime to allow passthrough over Link; in the Meta Quest Link app, enable the beta/developer runtime passthrough option if it is not already enabled.

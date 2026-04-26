using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace ParallelHeptics.FrontendUnity
{
    /// <summary>
    /// Pygame-like flat panel frontend for Quest Link/PCVR. It preserves the existing
    /// backend protocol while using Unity objects that can later be remapped into a 3D scene.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class FrontendUnityController : MonoBehaviour
    {
        private static readonly Color FirstColor = new Color32(255, 165, 0, 255);
        private static readonly Color SecondColor = new Color32(0, 0, 255, 255);
        private static readonly Color GrayColor = new Color32(128, 128, 128, 255);
        private static readonly Color FingerColor = new Color32(211, 211, 211, 255);
        private static readonly Color PanelColor = Color.black;
        private static readonly Color CalibrationPanelColor = new Color(0f, 0f, 0f, 0.08f);
        private static readonly Color DarkBarColor = new Color32(64, 64, 64, 255);
        private static readonly Color LightGreenColor = new Color32(144, 238, 144, 255);
        private static readonly Color GreenColor = new Color32(0, 200, 0, 255);
        private static readonly Vector3 DefaultTabletopPosition = new Vector3(0f, 0.75f, 0.55f);
        private static readonly Vector3 DefaultTabletopEulerAngles = new Vector3(-90f, 0f, 0f);

        private const float DefaultTabletopScale = 1f;
        private const float SurfaceOffset = 0.002f;
        private const float ForegroundOffset = 0.004f;
        private const float TextOffset = 0.006f;

        [Header("Backend protocol")]
        [SerializeField] private string backendHost = "localhost";
        [SerializeField] private int udpListenPort = 12346;
        [SerializeField] private int tcpBackendPort = 12344;
        [SerializeField] private bool bindUdpLoopbackOnly;

        [Header("Panel")]
        [SerializeField] private FlatPanelMapper mapper = new FlatPanelMapper();
        [SerializeField] private Vector3 panelWorldPosition = DefaultTabletopPosition;
        [SerializeField] private Vector3 panelEulerAngles = DefaultTabletopEulerAngles;
        [SerializeField] private float panelUniformScale = DefaultTabletopScale;
        [SerializeField] private int maxFingerObjects = 10;
        [SerializeField] private float fingerDiameter = 0.1f;
        [SerializeField] private float holdSeconds = 1.0f;
        [SerializeField] private bool showDebugStatus;

        [Header("Tabletop calibration")]
        [SerializeField] private bool enableKeyboardCalibration = true;
        [SerializeField] private bool calibrationActive = true;
        [SerializeField] private bool loadSavedCalibration = true;
        [SerializeField] private string calibrationPrefsKey = "ParallelHeptics.FrontendUnity.TabletopCalibration";
        [SerializeField] private float calibrationMoveMetersPerSecond = 0.05f;
        [SerializeField] private float calibrationYawDegreesPerSecond = 30f;
        [SerializeField] private float calibrationScalePerSecond = 0.5f;
        [SerializeField] private float calibrationScaleStep = 0.025f;

        private readonly List<GameObject> _fingerObjects = new List<GameObject>();
        private readonly Dictionary<Color, Material> _materials = new Dictionary<Color, Material>();
        private ExperimentUdpReceiver _receiver;
        private ExperimentControlClient _controlClient;
        private HoldButtonSelector _holdSelector;
        private Transform _panelRoot;
        private Transform _dynamicRoot;
        private GameObject _panelBackground;
        private GameObject _trackingObject;
        private GameObject _progressBackground;
        private GameObject _outboundProgress;
        private GameObject _returnProgress;
        private GameObject _leftButton;
        private GameObject _rightButton;
        private GameObject _leftHoldBackground;
        private GameObject _rightHoldBackground;
        private GameObject _leftHoldFill;
        private GameObject _rightHoldFill;
        private TextMesh _titleText;
        private TextMesh _subtitleText;
        private TextMesh _counterText;
        private TextMesh _statusText;
        private Camera _mainCamera;
        private ARSession _arSession;
        private ARCameraManager _arCameraManager;
        private GameObject _arSessionObject;
        private Color _cameraBackgroundColor;
        private CameraClearFlags _cameraClearFlags;
        private ExperimentState _lastState = (ExperimentState)999;
        private float _lastPacketTime = -1000f;
        private long _lastMalformedLogFrame = -9999;

        private enum CalibrationKey
        {
            Left,
            Right,
            Down,
            Up,
            PageDown,
            PageUp,
            Q,
            E,
            LeftBracket,
            RightBracket,
            R,
            Enter,
            Shift
        }

        [System.Serializable]
        private sealed class TabletopCalibrationData
        {
            public Vector3 position;
            public Vector3 eulerAngles;
            public float scale = DefaultTabletopScale;
        }

        private void Awake()
        {
            Application.runInBackground = true;
            Application.targetFrameRate = -1;
            QualitySettings.vSyncCount = 0;

            _receiver = GetComponent<ExperimentUdpReceiver>() ?? gameObject.AddComponent<ExperimentUdpReceiver>();
            _receiver.Configure(udpListenPort, bindUdpLoopbackOnly);

            _controlClient = GetComponent<ExperimentControlClient>() ?? gameObject.AddComponent<ExperimentControlClient>();
            _controlClient.Configure(backendHost, tcpBackendPort);

            _holdSelector = new HoldButtonSelector(holdSeconds);
            LoadCalibrationIfAvailable();
            BuildSceneGraph();
            ConfigurePassthroughSupport();
            ApplyCalibrationVisibility();
            SetStateVisibility(ExperimentState.Start);
        }

        private void OnDestroy()
        {
            if (_panelRoot != null)
            {
                DestroyRuntimeObject(_panelRoot.gameObject);
            }
            if (_arSessionObject != null)
            {
                DestroyRuntimeObject(_arSessionObject);
            }
        }

        private void Update()
        {
            HandleCalibrationInput();

            if (_receiver.TryGetLatestJson(out string json))
            {
                try
                {
                    ExperimentPacket packet = JsonUtility.FromJson<ExperimentPacket>(json);
                    if (packet != null && packet.stateData != null)
                    {
                        _lastPacketTime = Time.unscaledTime;
                        RenderPacket(packet);
                    }
                }
                catch (System.Exception ex)
                {
                    if (Time.frameCount - _lastMalformedLogFrame > 120)
                    {
                        _lastMalformedLogFrame = Time.frameCount;
                        Debug.LogWarning($"Malformed ExperimentPacket JSON ignored: {ex.Message}");
                    }
                }
            }

            if (showDebugStatus && _statusText != null)
            {
                float age = Time.unscaledTime - _lastPacketTime;
                string packetStatus = _lastPacketTime < 0f ? "waiting for UDP" : $"last UDP {age:0.00}s ago";
                _statusText.text = $"Unity Frontend | UDP:{udpListenPort} TCP:{tcpBackendPort} | {packetStatus}";
            }
        }

        private void RenderPacket(ExperimentPacket packet)
        {
            ExperimentState state = (ExperimentState)packet.stateData.state;
            SetStateVisibility(state);
            RenderFingers(packet.landmarks);

            switch (state)
            {
                case ExperimentState.Comparison:
                    RenderComparison(packet.trackingObject);
                    _holdSelector.Reset();
                    break;
                case ExperimentState.Question:
                    RenderQuestion(packet.landmarks);
                    break;
                case ExperimentState.Pause:
                    _holdSelector.Reset();
                    _titleText.text = "Take a break!";
                    _subtitleText.text = $"{packet.stateData.pauseTime} seconds left before moving to the next test...";
                    break;
                case ExperimentState.End:
                    _holdSelector.Reset();
                    _titleText.text = "Thank you for participating!";
                    _subtitleText.text = string.Empty;
                    break;
                default:
                    _holdSelector.Reset();
                    _titleText.text = string.Empty;
                    _subtitleText.text = string.Empty;
                    break;
            }
        }

        private void RenderComparison(TrackingObject trackingObject)
        {
            if (trackingObject == null)
            {
                return;
            }

            _trackingObject.transform.localPosition = mapper.NormalizedToLocal(trackingObject.x, trackingObject.z, ForegroundOffset);
            Vector2 objectSize = mapper.PixelsToPanelSize(trackingObject.size, trackingObject.size);
            _trackingObject.transform.localScale = new Vector3(objectSize.x, objectSize.y, 0.006f);
            Color color = trackingObject.isPinched ? (trackingObject.pairIndex == 0 ? FirstColor : SecondColor) : GrayColor;
            SetMaterial(_trackingObject, color);

            float barWidth = mapper.PanelWidth * 0.72f;
            float barHeight = mapper.PanelHeight * 0.04f;
            float halfWidth = barWidth * 0.5f;
            _progressBackground.transform.localScale = new Vector3(barWidth, barHeight, 1f);
            SetProgressBarHalf(_outboundProgress, Mathf.Clamp01(trackingObject.progress), leftHalf: true, halfWidth, barHeight);
            SetProgressBarHalf(_returnProgress, Mathf.Clamp01(trackingObject.returnProgress), leftHalf: false, halfWidth, barHeight);
            _counterText.text = $"{trackingObject.cycleCount}/{trackingObject.targetCycleCount}";
        }

        private void RenderQuestion(List<FingerPosition> landmarks)
        {
            _titleText.text = "Which object is stiffer?";
            _subtitleText.text = string.Empty;

            bool leftTouched = false;
            bool rightTouched = false;
            Rect leftRect = RectForObject(_leftButton);
            Rect rightRect = RectForObject(_rightButton);

            if (landmarks != null)
            {
                for (int i = 0; i < landmarks.Count; i++)
                {
                    Vector2 point = mapper.NormalizedToPanel2D(landmarks[i].x, landmarks[i].z);
                    leftTouched |= leftRect.Contains(point);
                    rightTouched |= rightRect.Contains(point);
                }
            }

            QuestionInput? answer = _holdSelector.Update(leftTouched, rightTouched, Time.deltaTime);
            if (answer.HasValue)
            {
                _controlClient.SendQuestionInput(answer.Value);
            }

            SetChildFillBar(_leftHoldFill, _holdSelector.LeftProgress);
            SetChildFillBar(_rightHoldFill, _holdSelector.RightProgress);
        }

        private void RenderFingers(List<FingerPosition> landmarks)
        {
            int count = landmarks == null ? 0 : Mathf.Min(landmarks.Count, _fingerObjects.Count);
            for (int i = 0; i < _fingerObjects.Count; i++)
            {
                bool active = i < count;
                _fingerObjects[i].SetActive(active);
                if (active)
                {
                    FingerPosition landmark = landmarks[i];
                    _fingerObjects[i].transform.localPosition = mapper.NormalizedToLocal(landmark.x, landmark.z, TextOffset);
                }
            }
        }

        private void SetStateVisibility(ExperimentState state)
        {
            if (state == _lastState)
            {
                return;
            }

            _lastState = state;
            bool comparison = state == ExperimentState.Comparison;
            bool question = state == ExperimentState.Question;
            bool message = state == ExperimentState.Pause || state == ExperimentState.End || question;

            _trackingObject.SetActive(comparison);
            _progressBackground.SetActive(comparison);
            _outboundProgress.SetActive(comparison);
            _returnProgress.SetActive(comparison);
            _counterText.gameObject.SetActive(comparison);

            _leftButton.SetActive(question);
            _rightButton.SetActive(question);
            _leftHoldBackground.SetActive(question);
            _rightHoldBackground.SetActive(question);
            _leftHoldFill.SetActive(question);
            _rightHoldFill.SetActive(question);
            _titleText.gameObject.SetActive(message);
            _subtitleText.gameObject.SetActive(message);
        }

        private void BuildSceneGraph()
        {
            _panelRoot = new GameObject("Flat VR Frontend Panel").transform;
            _panelRoot.SetParent(null, false);
            ApplyPanelTransform();

            _dynamicRoot = new GameObject("Dynamic Experiment Elements").transform;
            _dynamicRoot.SetParent(_panelRoot, false);

            _panelBackground = CreateQuad("Black Panel", _panelRoot, Vector3.zero, new Vector3(mapper.PanelWidth, mapper.PanelHeight, 1f), PanelColor);
            _trackingObject = CreateCube("Tracking Object", _dynamicRoot, Vector3.zero, Vector3.one, GrayColor);

            float progressY = -mapper.PanelHeight * 0.41f;
            _progressBackground = CreateQuad("Progress Background", _dynamicRoot, new Vector3(0f, progressY, SurfaceOffset), Vector3.one, DarkBarColor);
            _outboundProgress = CreateQuad("Outbound Progress", _dynamicRoot, new Vector3(-mapper.PanelWidth * 0.18f, progressY, ForegroundOffset), Vector3.one, LightGreenColor);
            _returnProgress = CreateQuad("Return Progress", _dynamicRoot, new Vector3(mapper.PanelWidth * 0.18f, progressY, ForegroundOffset), Vector3.one, GreenColor);
            _counterText = CreateText("Cycle Counter", _dynamicRoot, new Vector3(mapper.PanelWidth * 0.34f, mapper.PanelHeight * 0.42f, TextOffset), mapper.PanelHeight * 0.0125f, TextAnchor.MiddleCenter);

            float buttonDiameter = Mathf.Min(mapper.PanelWidth, mapper.PanelHeight) * 0.22f;
            Vector2 buttonSize = new Vector2(buttonDiameter, buttonDiameter);
            float buttonX = mapper.PanelWidth * 0.28f;
            float holdBarHeight = mapper.PanelHeight * 0.025f;
            _leftButton = CreateQuad("Question Left Button", _dynamicRoot, new Vector3(-buttonX, 0f, SurfaceOffset), new Vector3(buttonSize.x, buttonSize.y, 1f), FirstColor);
            _rightButton = CreateQuad("Question Right Button", _dynamicRoot, new Vector3(buttonX, 0f, SurfaceOffset), new Vector3(buttonSize.x, buttonSize.y, 1f), SecondColor);
            _leftHoldBackground = CreateQuad("Left Hold Background", _dynamicRoot, _leftButton.transform.localPosition + new Vector3(0f, buttonSize.y * 0.65f, SurfaceOffset), new Vector3(buttonSize.x, holdBarHeight, 1f), DarkBarColor);
            _rightHoldBackground = CreateQuad("Right Hold Background", _dynamicRoot, _rightButton.transform.localPosition + new Vector3(0f, buttonSize.y * 0.65f, SurfaceOffset), new Vector3(buttonSize.x, holdBarHeight, 1f), DarkBarColor);
            _leftHoldFill = CreateQuad("Left Hold Fill", _leftHoldBackground.transform, Vector3.zero, new Vector3(0f, 1f, 1f), GreenColor);
            _rightHoldFill = CreateQuad("Right Hold Fill", _rightHoldBackground.transform, Vector3.zero, new Vector3(0f, 1f, 1f), GreenColor);

            _titleText = CreateText("Title Text", _dynamicRoot, new Vector3(0f, -mapper.PanelHeight * 0.34f, TextOffset), mapper.PanelHeight * 0.03f, TextAnchor.MiddleCenter);
            _subtitleText = CreateText("Subtitle Text", _dynamicRoot, new Vector3(0f, mapper.PanelHeight * 0.1f, TextOffset), mapper.PanelHeight * 0.0225f, TextAnchor.MiddleCenter);
            _statusText = CreateText("Status Text", _panelRoot, new Vector3(0f, mapper.PanelHeight * 0.47f, TextOffset), mapper.PanelHeight * 0.0175f, TextAnchor.MiddleCenter);
            _statusText.gameObject.SetActive(showDebugStatus);

            for (int i = 0; i < maxFingerObjects; i++)
            {
                GameObject finger = CreateSphere($"Finger Landmark {i + 1}", _dynamicRoot, Vector3.zero, new Vector3(fingerDiameter, fingerDiameter, fingerDiameter), FingerColor);
                finger.SetActive(false);
                _fingerObjects.Add(finger);
            }
        }

        private GameObject CreateQuad(string objectName, Transform parent, Vector3 localPosition, Vector3 localScale, Color color)
        {
            return CreatePrimitive(PrimitiveType.Quad, objectName, parent, localPosition, localScale, color);
        }

        private GameObject CreateCube(string objectName, Transform parent, Vector3 localPosition, Vector3 localScale, Color color)
        {
            return CreatePrimitive(PrimitiveType.Cube, objectName, parent, localPosition, localScale, color);
        }

        private GameObject CreateSphere(string objectName, Transform parent, Vector3 localPosition, Vector3 localScale, Color color)
        {
            return CreatePrimitive(PrimitiveType.Sphere, objectName, parent, localPosition, localScale, color);
        }

        private GameObject CreatePrimitive(PrimitiveType primitiveType, string objectName, Transform parent, Vector3 localPosition, Vector3 localScale, Color color)
        {
            GameObject go = GameObject.CreatePrimitive(primitiveType);
            go.name = objectName;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPosition;
            go.transform.localScale = localScale;
            DestroyRuntimeObject(go.GetComponent<Collider>());
            SetMaterial(go, color);
            return go;
        }

        private static void DestroyRuntimeObject(Object objectToDestroy)
        {
            if (objectToDestroy == null)
            {
                return;
            }

            if (Application.isPlaying)
            {
                Destroy(objectToDestroy);
            }
            else
            {
                DestroyImmediate(objectToDestroy);
            }
        }

        private TextMesh CreateText(string objectName, Transform parent, Vector3 localPosition, float characterSize, TextAnchor anchor)
        {
            GameObject go = new GameObject(objectName);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPosition;
            go.transform.localRotation = Quaternion.identity;
            TextMesh text = go.AddComponent<TextMesh>();
            text.anchor = anchor;
            text.alignment = TextAlignment.Center;
            text.characterSize = characterSize;
            text.fontSize = 80;
            text.color = Color.white;
            return text;
        }

        private Rect RectForObject(GameObject go)
        {
            Vector3 center = go.transform.localPosition;
            Vector3 scale = go.transform.localScale;
            return new Rect(center.x - scale.x * 0.5f, center.y - scale.y * 0.5f, scale.x, scale.y);
        }

        private void ApplyPanelTransform()
        {
            if (_panelRoot == null)
            {
                return;
            }

            _panelRoot.position = panelWorldPosition;
            _panelRoot.eulerAngles = panelEulerAngles;
            _panelRoot.localScale = Vector3.one * Mathf.Max(0.1f, panelUniformScale);
        }

        private void HandleCalibrationInput()
        {
            if (!enableKeyboardCalibration || _panelRoot == null)
            {
                return;
            }

            if (!calibrationActive)
            {
                if (WasKeyPressedThisFrame(CalibrationKey.R))
                {
                    calibrationActive = true;
                    ApplyCalibrationVisibility();
                    Debug.Log("Tabletop calibration restarted. Press Enter to save and lock it again.");
                }

                return;
            }

            bool changed = false;
            float speedMultiplier = IsKeyPressed(CalibrationKey.Shift) ? 5f : 1f;
            float moveStep = calibrationMoveMetersPerSecond * speedMultiplier * Time.unscaledDeltaTime;
            float yawStep = calibrationYawDegreesPerSecond * speedMultiplier * Time.unscaledDeltaTime;
            float scaleStep = calibrationScalePerSecond * speedMultiplier * Time.unscaledDeltaTime;

            if (IsKeyPressed(CalibrationKey.Left))
            {
                panelWorldPosition -= _panelRoot.right * moveStep;
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.Right))
            {
                panelWorldPosition += _panelRoot.right * moveStep;
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.Down))
            {
                panelWorldPosition += _panelRoot.up * moveStep;
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.Up))
            {
                panelWorldPosition -= _panelRoot.up * moveStep;
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.PageDown))
            {
                panelWorldPosition -= _panelRoot.forward * moveStep;
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.PageUp))
            {
                panelWorldPosition += _panelRoot.forward * moveStep;
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.Q))
            {
                RotateTabletopYaw(-yawStep);
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.E))
            {
                RotateTabletopYaw(yawStep);
                changed = true;
            }
            if (WasKeyPressedThisFrame(CalibrationKey.LeftBracket))
            {
                panelUniformScale = Mathf.Max(0.01f, panelUniformScale - calibrationScaleStep);
                changed = true;
            }
            if (WasKeyPressedThisFrame(CalibrationKey.RightBracket))
            {
                panelUniformScale += calibrationScaleStep;
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.LeftBracket))
            {
                panelUniformScale = Mathf.Max(0.01f, panelUniformScale - scaleStep);
                changed = true;
            }
            if (IsKeyPressed(CalibrationKey.RightBracket))
            {
                panelUniformScale += scaleStep;
                changed = true;
            }
            if (WasKeyPressedThisFrame(CalibrationKey.R))
            {
                calibrationActive = true;
                changed = true;
            }
            if (WasKeyPressedThisFrame(CalibrationKey.Enter))
            {
                SaveCalibration();
            }

            if (changed)
            {
                ApplyPanelTransform();
            }
        }

        private static bool IsKeyPressed(CalibrationKey key)
        {
#if ENABLE_INPUT_SYSTEM
            Keyboard keyboard = Keyboard.current;
            if (keyboard != null)
            {
                return key switch
                {
                    CalibrationKey.Left => keyboard.leftArrowKey.isPressed,
                    CalibrationKey.Right => keyboard.rightArrowKey.isPressed,
                    CalibrationKey.Down => keyboard.downArrowKey.isPressed,
                    CalibrationKey.Up => keyboard.upArrowKey.isPressed,
                    CalibrationKey.PageDown => keyboard.pageDownKey.isPressed,
                    CalibrationKey.PageUp => keyboard.pageUpKey.isPressed,
                    CalibrationKey.Q => keyboard.qKey.isPressed,
                    CalibrationKey.E => keyboard.eKey.isPressed,
                    CalibrationKey.LeftBracket => keyboard.leftBracketKey.isPressed,
                    CalibrationKey.RightBracket => keyboard.rightBracketKey.isPressed,
                    CalibrationKey.R => keyboard.rKey.isPressed,
                    CalibrationKey.Enter => keyboard.enterKey.isPressed || keyboard.numpadEnterKey.isPressed,
                    CalibrationKey.Shift => keyboard.leftShiftKey.isPressed || keyboard.rightShiftKey.isPressed,
                    _ => false
                };
            }
#endif

#if ENABLE_LEGACY_INPUT_MANAGER
            return key switch
            {
                CalibrationKey.Left => Input.GetKey(KeyCode.LeftArrow),
                CalibrationKey.Right => Input.GetKey(KeyCode.RightArrow),
                CalibrationKey.Down => Input.GetKey(KeyCode.DownArrow),
                CalibrationKey.Up => Input.GetKey(KeyCode.UpArrow),
                CalibrationKey.PageDown => Input.GetKey(KeyCode.PageDown),
                CalibrationKey.PageUp => Input.GetKey(KeyCode.PageUp),
                CalibrationKey.Q => Input.GetKey(KeyCode.Q),
                CalibrationKey.E => Input.GetKey(KeyCode.E),
                CalibrationKey.LeftBracket => Input.GetKey(KeyCode.LeftBracket),
                CalibrationKey.RightBracket => Input.GetKey(KeyCode.RightBracket),
                CalibrationKey.R => Input.GetKey(KeyCode.R),
                CalibrationKey.Enter => Input.GetKey(KeyCode.Return) || Input.GetKey(KeyCode.KeypadEnter),
                CalibrationKey.Shift => Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift),
                _ => false
            };
#else
            return false;
#endif
        }

        private static bool WasKeyPressedThisFrame(CalibrationKey key)
        {
#if ENABLE_INPUT_SYSTEM
            Keyboard keyboard = Keyboard.current;
            if (keyboard != null)
            {
                return key switch
                {
                    CalibrationKey.LeftBracket => keyboard.leftBracketKey.wasPressedThisFrame,
                    CalibrationKey.RightBracket => keyboard.rightBracketKey.wasPressedThisFrame,
                    CalibrationKey.R => keyboard.rKey.wasPressedThisFrame,
                    CalibrationKey.Enter => keyboard.enterKey.wasPressedThisFrame || keyboard.numpadEnterKey.wasPressedThisFrame,
                    _ => false
                };
            }
#endif

#if ENABLE_LEGACY_INPUT_MANAGER
            return key switch
            {
                CalibrationKey.LeftBracket => Input.GetKeyDown(KeyCode.LeftBracket),
                CalibrationKey.RightBracket => Input.GetKeyDown(KeyCode.RightBracket),
                CalibrationKey.R => Input.GetKeyDown(KeyCode.R),
                CalibrationKey.Enter => Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter),
                _ => false
            };
#else
            return false;
#endif
        }

        private void RotateTabletopYaw(float degrees)
        {
            Quaternion current = Quaternion.Euler(panelEulerAngles);
            panelEulerAngles = (Quaternion.AngleAxis(degrees, Vector3.up) * current).eulerAngles;
        }

        private void LoadCalibrationIfAvailable()
        {
            if (!loadSavedCalibration || string.IsNullOrWhiteSpace(calibrationPrefsKey) || !PlayerPrefs.HasKey(calibrationPrefsKey))
            {
                return;
            }

            try
            {
                TabletopCalibrationData data = JsonUtility.FromJson<TabletopCalibrationData>(PlayerPrefs.GetString(calibrationPrefsKey));
                if (data != null)
                {
                    panelWorldPosition = data.position;
                    panelEulerAngles = data.eulerAngles;
                    panelUniformScale = Mathf.Max(0.1f, data.scale);
                    calibrationActive = false;
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"Failed to load tabletop calibration; using defaults. {ex.Message}");
            }
        }

        private void SaveCalibration()
        {
            if (string.IsNullOrWhiteSpace(calibrationPrefsKey))
            {
                return;
            }

            var data = new TabletopCalibrationData
            {
                position = panelWorldPosition,
                eulerAngles = panelEulerAngles,
                scale = panelUniformScale
            };
            PlayerPrefs.SetString(calibrationPrefsKey, JsonUtility.ToJson(data));
            PlayerPrefs.Save();
            calibrationActive = false;
            ApplyCalibrationVisibility();
            Debug.Log($"Saved tabletop calibration: position={panelWorldPosition}, rotation={panelEulerAngles}, scale={panelUniformScale:0.###}");
        }

        private void ConfigurePassthroughSupport()
        {
            _mainCamera = Camera.main;
            if (_mainCamera == null)
            {
                return;
            }

            _cameraClearFlags = _mainCamera.clearFlags;
            _cameraBackgroundColor = _mainCamera.backgroundColor;

            _arSessionObject = new GameObject("AR Session (Passthrough Calibration)");
            _arSession = _arSessionObject.AddComponent<ARSession>();
            _arSession.enabled = false;

            _arCameraManager = _mainCamera.GetComponent<ARCameraManager>() ?? _mainCamera.gameObject.AddComponent<ARCameraManager>();
            _arCameraManager.enabled = false;
        }

        private void ApplyCalibrationVisibility()
        {
            if (_panelBackground != null)
            {
                SetMaterial(_panelBackground, calibrationActive ? CalibrationPanelColor : PanelColor);
            }

            if (_mainCamera == null)
            {
                return;
            }

            if (calibrationActive)
            {
                _mainCamera.clearFlags = CameraClearFlags.SolidColor;
                _mainCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
            }
            else
            {
                _mainCamera.clearFlags = _cameraClearFlags;
                _mainCamera.backgroundColor = _cameraBackgroundColor;
            }

            if (_arSession != null)
            {
                _arSession.enabled = calibrationActive;
            }
            if (_arCameraManager != null)
            {
                _arCameraManager.enabled = calibrationActive;
            }
        }

        private void SetBar(GameObject bar, float width, float height, float centerX)
        {
            width = Mathf.Max(0f, width);
            bar.transform.localScale = new Vector3(width, height, 1f);
            bar.transform.localPosition = new Vector3(centerX, bar.transform.localPosition.y, bar.transform.localPosition.z);
            bar.SetActive(width > 0.0001f);
        }

        private void SetProgressBarHalf(GameObject bar, float progress, bool leftHalf, float halfWidth, float height)
        {
            float width = Mathf.Clamp01(progress) * halfWidth;
            float halfLeftEdge = leftHalf ? -halfWidth : 0f;
            float x = halfLeftEdge + width * 0.5f;
            SetBar(bar, width, height, x);
        }

        private void SetChildFillBar(GameObject bar, float progress)
        {
            float width = Mathf.Clamp01(progress);
            bar.transform.localScale = new Vector3(width, 1f, 1f);
            bar.transform.localPosition = new Vector3(-0.5f + width * 0.5f, 0f, SurfaceOffset);
            bar.SetActive(width > 0.0001f);
        }

        private void SetMaterial(GameObject go, Color color)
        {
            Renderer renderer = go.GetComponent<Renderer>();
            if (renderer == null)
            {
                return;
            }

            if (!_materials.TryGetValue(color, out Material material))
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Color") ?? Shader.Find("Standard");
                material = new Material(shader);
                if (material.HasProperty("_BaseColor"))
                {
                    material.SetColor("_BaseColor", color);
                }
                if (material.HasProperty("_Color"))
                {
                    material.SetColor("_Color", color);
                }
                if (material.HasProperty("_Cull"))
                {
                    material.SetFloat("_Cull", 0f);
                }
                ConfigureMaterialSurface(material, color.a);

                _materials.Add(color, material);
            }

            renderer.sharedMaterial = material;
        }

        private static void ConfigureMaterialSurface(Material material, float alpha)
        {
            if (alpha >= 0.999f)
            {
                material.renderQueue = -1;
                if (material.HasProperty("_Surface"))
                {
                    material.SetFloat("_Surface", 0f);
                }
                if (material.HasProperty("_ZWrite"))
                {
                    material.SetFloat("_ZWrite", 1f);
                }

                return;
            }

            material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            if (material.HasProperty("_Surface"))
            {
                material.SetFloat("_Surface", 1f);
            }
            if (material.HasProperty("_SrcBlend"))
            {
                material.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            }
            if (material.HasProperty("_DstBlend"))
            {
                material.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            }
            if (material.HasProperty("_ZWrite"))
            {
                material.SetFloat("_ZWrite", 0f);
            }
            material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        }
    }
}

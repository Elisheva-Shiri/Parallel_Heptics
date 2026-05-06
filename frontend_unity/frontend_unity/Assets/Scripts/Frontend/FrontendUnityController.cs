using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
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
        private static readonly Color CenterCueColor = GreenColor;
        private static readonly Vector3 DefaultTabletopPosition = new Vector3(0f, 0.75f, 0.55f);
        private static readonly Vector3 DefaultTabletopEulerAngles = new Vector3(-90f, 0f, 0f);

        private const float DefaultTabletopScale = 1f;
        private const float SurfaceOffset = 0.002f;
        private const float ForegroundOffset = 0.004f;
        private const float TextOffset = 0.006f;
        private const float TextScaleFactor = 0.5f;
        private const int CueCircleSegments = 96;
        private const float AmbientNoiseLowPassAlpha = 0.075f;
        private const float AmbientNoiseRawMix = 0.16f;
        private const float ControllerAnalogPressThreshold = 0.55f;
        private const string CueModeCircleBorder = "circle_border";
        private const string CueModeCircleFilled = "circle_filled";
        private const string CueModeRubberBand = "rubber_band";
        private static readonly string[] InputSystemControllerControlNames =
        {
            "triggerPressed",
            "gripPressed",
            "trigger",
            "grip",
            "indexTrigger",
            "handTrigger",
            "primaryButton",
            "secondaryButton"
        };
        // Intentionally excludes CommonUsages.menuButton because Quest maps
        // menu/system-style buttons closest to the Meta/calibration role.
        private static readonly UnityEngine.XR.InputFeatureUsage<bool>[] ControllerInteractionButtons =
        {
            UnityEngine.XR.CommonUsages.primaryButton,
            UnityEngine.XR.CommonUsages.secondaryButton,
            UnityEngine.XR.CommonUsages.gripButton,
            UnityEngine.XR.CommonUsages.triggerButton,
            UnityEngine.XR.CommonUsages.primary2DAxisClick,
            UnityEngine.XR.CommonUsages.secondary2DAxisClick
        };
        private static readonly UnityEngine.XR.InputFeatureUsage<float>[] ControllerInteractionAxes =
        {
            UnityEngine.XR.CommonUsages.trigger,
            UnityEngine.XR.CommonUsages.grip
        };

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
        [SerializeField] private bool enableControllerInteractionToggle = true;
        [SerializeField] private bool showDebugStatus;

        [Header("Audio masking")]
        [SerializeField, Range(0f, 1f)] private float whiteNoiseVolume = 0.15f;
        [SerializeField] private int whiteNoiseSampleRate = 44100;

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
        private readonly Dictionary<string, bool> _controllerButtonPressed = new Dictionary<string, bool>();
        private readonly List<UnityEngine.XR.InputDevice> _controllerDevices = new List<UnityEngine.XR.InputDevice>();
        private ExperimentUdpReceiver _receiver;
        private ExperimentControlClient _controlClient;
        private HoldButtonSelector _holdSelector;
        private Transform _panelRoot;
        private Transform _dynamicRoot;
        private GameObject _panelBackground;
        private GameObject _trackingObject;
        private GameObject _outboundCueCircle;
        private LineRenderer _outboundCueRenderer;
        private GameObject _filledCueCircle;
        private MeshFilter _filledCueMeshFilter;
        private GameObject _rubberBandCue;
        private LineRenderer _rubberBandRenderer;
        private GameObject _returnCuePoint;
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
        private GameObject _xrOriginObject;
        private GameObject _cameraOffsetObject;
        private XROrigin _xrOrigin;
        private AudioSource _whiteNoiseSource;
        private AudioClip _whiteNoiseClip;
        private bool _whiteNoiseEnabled;
        private bool _whiteNoiseUnavailable;
#if ENABLE_INPUT_SYSTEM
        private TrackedPoseDriver _trackedPoseDriver;
        private InputAction _hmdPositionAction;
        private InputAction _hmdRotationAction;
        private InputAction _hmdTrackingStateAction;
#endif
        private Color _cameraBackgroundColor;
        private CameraClearFlags _cameraClearFlags;
        private bool _ownsArSession;
        private bool _ownsXrOrigin;
        private ExperimentState _lastState = (ExperimentState)999;
        private float _lastPacketTime = -1000f;
        private long _lastMalformedLogFrame = -9999;
        private bool _isDebug = true;
        private float _lastBorderCueRadius = -1f;
        private float _lastFilledCueRadius = -1f;
        private Vector3 _lastRubberBandEnd = new Vector3(float.NaN, float.NaN, float.NaN);
        private float _nextControllerDiagnosticTime;

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

            _receiver = GetOrAddComponent<ExperimentUdpReceiver>(gameObject);
            _receiver.Configure(udpListenPort, bindUdpLoopbackOnly);

            _controlClient = GetOrAddComponent<ExperimentControlClient>(gameObject);
            _controlClient.Configure(backendHost, tcpBackendPort);

            _holdSelector = new HoldButtonSelector(holdSeconds);
            LoadCalibrationIfAvailable();
            EnsureXrCameraRig();
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
            if (_ownsArSession && _arSessionObject != null)
            {
                DestroyRuntimeObject(_arSessionObject);
            }
            if (_ownsXrOrigin && _xrOriginObject != null)
            {
                DestroyRuntimeObject(_xrOriginObject);
            }
            if (_whiteNoiseSource != null)
            {
                _whiteNoiseSource.Stop();
            }
            if (_whiteNoiseClip != null)
            {
                DestroyRuntimeObject(_whiteNoiseClip);
                _whiteNoiseClip = null;
            }
#if ENABLE_INPUT_SYSTEM
            DisposeAction(ref _hmdPositionAction);
            DisposeAction(ref _hmdRotationAction);
            DisposeAction(ref _hmdTrackingStateAction);
#endif
        }

#if ENABLE_INPUT_SYSTEM
        private static void DisposeAction(ref InputAction action)
        {
            if (action == null)
            {
                return;
            }
            action.Disable();
            action.Dispose();
            action = null;
        }
#endif

        private void SetWhiteNoiseEnabled(bool enabled)
        {
            if (_whiteNoiseUnavailable)
            {
                return;
            }

            if (enabled == _whiteNoiseEnabled)
            {
                return;
            }

            try
            {
                _whiteNoiseEnabled = enabled;
                if (enabled)
                {
                    EnsureWhiteNoiseAudio();
                    _whiteNoiseSource.Play();
                }
                else if (_whiteNoiseSource != null)
                {
                    _whiteNoiseSource.Stop();
                }
            }
            catch (System.Exception ex)
            {
                _whiteNoiseEnabled = false;
                _whiteNoiseUnavailable = true;
                if (_whiteNoiseSource != null)
                {
                    _whiteNoiseSource.Stop();
                }
                LogWarningIfDebug($"White noise disabled; Unity audio initialization failed. {ex.Message}");
            }
        }

        private void EnsureWhiteNoiseAudio()
        {
            if (_whiteNoiseClip == null)
            {
                int configuredSampleRate = Mathf.Clamp(whiteNoiseSampleRate, 8000, 96000);
                int sampleRate = AudioSettings.outputSampleRate > 0
                    ? AudioSettings.outputSampleRate
                    : configuredSampleRate;
                int sampleCount = sampleRate * 2;
                float[] samples = new float[sampleCount];
                FillWhiteNoiseSamples(samples, whiteNoiseVolume, 0x1234ABCDu);

                _whiteNoiseClip = AudioClip.Create(
                    "Procedural White Noise",
                    sampleCount,
                    1,
                    sampleRate,
                    false);
                _whiteNoiseClip.SetData(samples, 0);
            }

            _whiteNoiseSource = GetOrAddComponent<AudioSource>(gameObject);
            _whiteNoiseSource.playOnAwake = false;
            _whiteNoiseSource.loop = true;
            _whiteNoiseSource.spatialBlend = 0f;
            _whiteNoiseSource.volume = 1f;
            _whiteNoiseSource.clip = _whiteNoiseClip;
        }

        private static T GetOrAddComponent<T>(GameObject owner) where T : Component
        {
            T component = owner.GetComponent<T>();
            return component != null ? component : owner.AddComponent<T>();
        }

        private static void FillWhiteNoiseSamples(float[] data, float requestedVolume, uint seed)
        {
            float volume = Mathf.Clamp01(requestedVolume);
            uint state = seed;
            float filtered = 0f;

            for (int i = 0; i < data.Length; i++)
            {
                unchecked
                {
                    state = (state * 1664525u) + 1013904223u;
                }

                float raw = ((state >> 8) / 8388607.5f) - 1f;
                filtered += (raw - filtered) * AmbientNoiseLowPassAlpha;
                float softened = Mathf.Lerp(filtered, raw, AmbientNoiseRawMix);
                data[i] = softened * volume;
            }
        }

        private void Update()
        {
            HandleCalibrationInput();
            HandleControllerInteractionToggle();

            if (_receiver.TryGetLatestJson(out string json))
            {
                ExperimentPacket packet = null;
                try
                {
                    packet = JsonUtility.FromJson<ExperimentPacket>(json);
                }
                catch (System.Exception ex)
                {
                    if (Time.frameCount - _lastMalformedLogFrame > 120)
                    {
                        _lastMalformedLogFrame = Time.frameCount;
                        LogWarningIfDebug($"Malformed ExperimentPacket JSON ignored: {ex.Message}");
                    }
                }

                if (packet != null && packet.stateData != null)
                {
                    _lastPacketTime = Time.unscaledTime;
                    RenderPacket(packet);
                }
            }

            if (_isDebug && showDebugStatus && _statusText != null)
            {
                float age = Time.unscaledTime - _lastPacketTime;
                string packetStatus = _lastPacketTime < 0f ? "waiting for UDP" : $"last UDP {age:0.00}s ago";
                _statusText.text = $"Unity Frontend | UDP:{udpListenPort} TCP:{tcpBackendPort} | {packetStatus}";
            }
        }

        private void HandleControllerInteractionToggle()
        {
            if (!enableControllerInteractionToggle || _controlClient == null)
            {
                return;
            }

#if ENABLE_INPUT_SYSTEM
            if (HasKeyboardSpacePressedThisFrame())
            {
                SendInteractionToggle("keyboard Space");
                return;
            }

            if (TrySendControllerToggleFromInputSystemDevices())
            {
                return;
            }
#endif

            bool sentThisFrame = TrySendControllerToggleFromXrNode(UnityEngine.XR.XRNode.LeftHand)
                || TrySendControllerToggleFromXrNode(UnityEngine.XR.XRNode.RightHand);
            if (!sentThisFrame)
            {
                _controllerDevices.Clear();
                UnityEngine.XR.InputDevices.GetDevicesWithCharacteristics(
                    UnityEngine.XR.InputDeviceCharacteristics.Controller,
                    _controllerDevices);

                for (int deviceIndex = 0; deviceIndex < _controllerDevices.Count; deviceIndex++)
                {
                    if (TrySendControllerToggleFromXrDevice(
                            _controllerDevices[deviceIndex],
                            $"characteristics:{deviceIndex}",
                            ref sentThisFrame))
                    {
                        sentThisFrame = true;
                    }
                }
            }

            LogControllerDiagnosticsIfNeeded();
        }

        private bool TrySendControllerToggleFromXrNode(UnityEngine.XR.XRNode node)
        {
            UnityEngine.XR.InputDevice device = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(node);
            bool sentThisFrame = false;
            return TrySendControllerToggleFromXrDevice(device, $"node:{node}", ref sentThisFrame);
        }

        private bool TrySendControllerToggleFromXrDevice(UnityEngine.XR.InputDevice device, string deviceKeyPrefix, ref bool sentThisFrame)
        {
            if (!device.isValid)
            {
                return false;
            }

            bool sawInput = false;
            for (int usageIndex = 0; usageIndex < ControllerInteractionButtons.Length; usageIndex++)
            {
                UnityEngine.XR.InputFeatureUsage<bool> usage = ControllerInteractionButtons[usageIndex];
                bool isPressed = false;
                if (!device.TryGetFeatureValue(usage, out isPressed))
                {
                    continue;
                }

                sawInput = true;
                string key = $"xr:{deviceKeyPrefix}:{device.name}:{device.characteristics}:{usage.name}";
                _controllerButtonPressed.TryGetValue(key, out bool wasPressed);
                if (isPressed && !wasPressed && !sentThisFrame)
                {
                    SendInteractionToggle($"{device.name} {usage.name}");
                    sentThisFrame = true;
                }

                _controllerButtonPressed[key] = isPressed;
            }

            for (int usageIndex = 0; usageIndex < ControllerInteractionAxes.Length; usageIndex++)
            {
                UnityEngine.XR.InputFeatureUsage<float> usage = ControllerInteractionAxes[usageIndex];
                float value = 0f;
                if (!device.TryGetFeatureValue(usage, out value))
                {
                    continue;
                }

                sawInput = true;
                bool isPressed = value >= ControllerAnalogPressThreshold;
                string key = $"xr:{deviceKeyPrefix}:{device.name}:{device.characteristics}:axis:{usage.name}";
                _controllerButtonPressed.TryGetValue(key, out bool wasPressed);
                if (isPressed && !wasPressed && !sentThisFrame)
                {
                    SendInteractionToggle($"{device.name} {usage.name} axis ({value:0.00})");
                    sentThisFrame = true;
                }

                _controllerButtonPressed[key] = isPressed;
            }

            return sawInput;
        }

        private void SendInteractionToggle(string source)
        {
            _controlClient.SendInteractionToggle();
            LogIfDebug($"Interaction toggle from {source}");
        }

        private void LogControllerDiagnosticsIfNeeded()
        {
            if (!_isDebug || Time.unscaledTime < _nextControllerDiagnosticTime)
            {
                return;
            }

            _nextControllerDiagnosticTime = Time.unscaledTime + 5f;
            var left = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.LeftHand);
            var right = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.RightHand);
            if (!left.isValid && !right.isValid)
            {
                LogIfDebug("No valid XR left/right controller devices detected yet.");
            }
        }

#if ENABLE_INPUT_SYSTEM
        private static bool HasKeyboardSpacePressedThisFrame()
        {
            Keyboard keyboard = Keyboard.current;
            return keyboard != null && keyboard.spaceKey.wasPressedThisFrame;
        }

        private bool TrySendControllerToggleFromInputSystemDevices()
        {
            bool sentThisFrame = false;
            foreach (UnityEngine.InputSystem.InputDevice device in UnityEngine.InputSystem.InputSystem.devices)
            {
                if (!IsControllerOrHandInputSystemDevice(device))
                {
                    continue;
                }

                for (int i = 0; i < InputSystemControllerControlNames.Length; i++)
                {
                    string controlName = InputSystemControllerControlNames[i];
                    if (TryReadInputSystemControlPressed(device, controlName, out bool isPressed, out float value))
                    {
                        string key = $"inputsystem:{device.deviceId}:{controlName}";
                        _controllerButtonPressed.TryGetValue(key, out bool wasPressed);
                        if (isPressed && !wasPressed && !sentThisFrame)
                        {
                            SendInteractionToggle($"{device.displayName}/{controlName} ({value:0.00})");
                            sentThisFrame = true;
                        }

                        _controllerButtonPressed[key] = isPressed;
                    }
                }
            }

            return sentThisFrame;
        }

        private static bool TryReadInputSystemControlPressed(
            UnityEngine.InputSystem.InputDevice device,
            string controlName,
            out bool isPressed,
            out float value)
        {
            isPressed = false;
            value = 0f;

            UnityEngine.InputSystem.InputControl control = device.TryGetChildControl<UnityEngine.InputSystem.InputControl>(controlName);
            if (control == null)
            {
                return false;
            }

            if (control is UnityEngine.InputSystem.Controls.ButtonControl button)
            {
                value = button.ReadValue();
                isPressed = button.isPressed || value >= ControllerAnalogPressThreshold;
                return true;
            }

            if (control is UnityEngine.InputSystem.Controls.AxisControl axis)
            {
                value = axis.ReadValue();
                isPressed = value >= ControllerAnalogPressThreshold;
                return true;
            }

            return false;
        }

        private static bool IsControllerOrHandInputSystemDevice(UnityEngine.InputSystem.InputDevice device)
        {
            string layout = (device.layout ?? string.Empty).ToLowerInvariant();
            string name = (device.name ?? string.Empty).ToLowerInvariant();
            string displayName = (device.displayName ?? string.Empty).ToLowerInvariant();
            if (layout.Contains("controller") || layout.Contains("touch") ||
                name.Contains("controller") || name.Contains("touch") ||
                displayName.Contains("controller") || displayName.Contains("touch"))
            {
                return true;
            }

            foreach (UnityEngine.InputSystem.Utilities.InternedString usage in device.usages)
            {
                string usageName = usage.ToString();
                if (usageName == "LeftHand" || usageName == "RightHand")
                {
                    return true;
                }
            }

            return false;
        }

#endif

        private void RenderPacket(ExperimentPacket packet)
        {
            EnsureHoldSelector();
            _isDebug = packet.isDebug;
            if (_receiver != null)
            {
                _receiver.IsDebug = _isDebug;
            }
            if (_controlClient != null)
            {
                _controlClient.IsDebug = _isDebug;
            }
            if (_statusText != null)
            {
                _statusText.gameObject.SetActive(_isDebug && showDebugStatus);
            }
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
                case ExperimentState.Break:
                    _holdSelector.Reset();
                    _titleText.text = "Break";
                    _subtitleText.text = $"Break so far: {packet.stateData.pauseTime} s\nSwitch fingers on the screen";
                    break;
                case ExperimentState.ModeratorPause:
                    _holdSelector.Reset();
                    _titleText.text = "Pause";
                    _subtitleText.text = $"Paused for: {packet.stateData.pauseTime}";
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

            // Audio masking is intentionally best-effort. Rendering the experiment
            // must continue even on machines/headsets where Unity cannot initialize
            // an audio output device.
            SetWhiteNoiseEnabled(packet.playWhiteNoise);
        }

        private void EnsureHoldSelector()
        {
            if (_holdSelector == null)
            {
                _holdSelector = new HoldButtonSelector(holdSeconds);
            }
        }

        private void LogIfDebug(string message)
        {
            if (_isDebug)
            {
                Debug.Log(message);
            }
        }

        private void LogWarningIfDebug(string message)
        {
            if (_isDebug)
            {
                Debug.LogWarning(message);
            }
        }

        private void RenderComparison(TrackingObject trackingObject)
        {
            if (trackingObject == null)
            {
                return;
            }

            _trackingObject.transform.localPosition = mapper.NormalizedToLocalUnclamped(trackingObject.x, trackingObject.z, ForegroundOffset);
            Vector2 objectSize = mapper.PixelsToPanelSize(trackingObject.size, trackingObject.size);
            _trackingObject.transform.localScale = new Vector3(objectSize.x, objectSize.y, 0.006f);
            Color color = trackingObject.isInteracting ? (trackingObject.pairIndex == 0 ? FirstColor : SecondColor) : GrayColor;
            SetMaterial(_trackingObject, color);

            SetVisualCue(trackingObject.isInteracting, trackingObject, color);
            _returnCuePoint.SetActive(false);
            _titleText.text = string.Empty;

            float barWidth = mapper.PanelWidth * 0.72f;
            float barHeight = mapper.PanelHeight * 0.04f;
            float halfWidth = barWidth * 0.5f;
            _progressBackground.transform.localScale = new Vector3(barWidth, barHeight, 1f);
            SetProgressBarHalf(_outboundProgress, Mathf.Clamp01(trackingObject.progress), leftHalf: true, halfWidth, barHeight);
            SetProgressBarHalf(_returnProgress, Mathf.Clamp01(trackingObject.returnProgress), leftHalf: false, halfWidth, barHeight);
            bool showCounter = ShouldShowCycleCounter(trackingObject.targetCycleCount);
            _counterText.gameObject.SetActive(showCounter);
            _counterText.text = showCounter ? $"{trackingObject.cycleCount}/{trackingObject.targetCycleCount}" : string.Empty;
        }

        private static bool ShouldShowCycleCounter(int targetCycleCount)
        {
            return targetCycleCount > 1;
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
            bool message = state == ExperimentState.Pause || state == ExperimentState.Break || state == ExperimentState.ModeratorPause || state == ExperimentState.End || question;

            _trackingObject.SetActive(comparison);
            SetAllVisualCuesActive(false);
            _returnCuePoint.SetActive(false);
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
            _titleText.gameObject.SetActive(message || comparison);
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

            _outboundCueCircle = CreateCircleCue("Outbound Edge Cue", _dynamicRoot, FirstColor);
            _filledCueCircle = CreateFilledCircleCue("Filled Circle Cue", _dynamicRoot, FirstColor);
            _rubberBandCue = CreateRubberBandCue("Rubber Band Cue", _dynamicRoot, FirstColor);
            float pointDiameter = Mathf.Min(mapper.PanelWidth, mapper.PanelHeight) * 0.07f;
            _returnCuePoint = CreateSphere("Center Return Point", _dynamicRoot, new Vector3(0f, 0f, ForegroundOffset), new Vector3(pointDiameter, pointDiameter, pointDiameter * 0.18f), CenterCueColor);
            SetAllVisualCuesActive(false);
            _returnCuePoint.SetActive(false);

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

        private GameObject CreateCircleCue(string objectName, Transform parent, Color color)
        {
            GameObject go = new GameObject(objectName);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = new Vector3(0f, 0f, ForegroundOffset);

            _outboundCueRenderer = go.AddComponent<LineRenderer>();
            _outboundCueRenderer.useWorldSpace = false;
            _outboundCueRenderer.loop = true;
            _outboundCueRenderer.positionCount = CueCircleSegments;
            _outboundCueRenderer.widthMultiplier = Mathf.Min(mapper.PanelWidth, mapper.PanelHeight) * 0.008f;
            _outboundCueRenderer.numCapVertices = 5;
            _outboundCueRenderer.numCornerVertices = 5;
            _outboundCueRenderer.material = GetMaterial(color);
            _outboundCueRenderer.startColor = color;
            _outboundCueRenderer.endColor = color;
            SetCircleRadius(0f);

            return go;
        }

        private GameObject CreateFilledCircleCue(string objectName, Transform parent, Color color)
        {
            GameObject go = new GameObject(objectName);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = new Vector3(0f, 0f, SurfaceOffset);

            _filledCueMeshFilter = go.AddComponent<MeshFilter>();
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = GetMaterial(color);
            SetFilledCircleRadius(0f);

            return go;
        }

        private GameObject CreateRubberBandCue(string objectName, Transform parent, Color color)
        {
            GameObject go = new GameObject(objectName);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = new Vector3(0f, 0f, SurfaceOffset);

            _rubberBandRenderer = go.AddComponent<LineRenderer>();
            _rubberBandRenderer.useWorldSpace = false;
            _rubberBandRenderer.loop = false;
            _rubberBandRenderer.positionCount = 2;
            _rubberBandRenderer.widthMultiplier = Mathf.Min(mapper.PanelWidth, mapper.PanelHeight) * 0.01f;
            _rubberBandRenderer.numCapVertices = 5;
            _rubberBandRenderer.material = GetMaterial(color);
            _rubberBandRenderer.startColor = color;
            _rubberBandRenderer.endColor = color;
            _rubberBandRenderer.SetPosition(0, Vector3.zero);
            _rubberBandRenderer.SetPosition(1, Vector3.zero);

            return go;
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
            // The panel is laid down as a horizontal tabletop (-90 deg X), which leaves
            // the TextMesh's "up" axis pointing toward the viewer instead of away.
            // No pure rotation can fix this without also reversing the reading direction
            // (the chirality flips), so flip the local Y scale. Unity's default font
            // material is Cull Off, so the negated winding still renders both sides.
            go.transform.localScale = new Vector3(1f, -1f, 1f);
            TextMesh text = go.AddComponent<TextMesh>();
            text.anchor = anchor;
            text.alignment = TextAlignment.Center;
            text.characterSize = characterSize * TextScaleFactor;
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
                    LogIfDebug("Tabletop calibration restarted. Press Enter to save and lock it again.");
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
                LogWarningIfDebug($"Failed to load tabletop calibration; using defaults. {ex.Message}");
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
            LogIfDebug($"Saved tabletop calibration: position={panelWorldPosition}, rotation={panelEulerAngles}, scale={panelUniformScale:0.###}");
        }

        private void EnsureXrCameraRig()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                return;
            }

            _xrOrigin = FindFirstObjectInSceneOfType<XROrigin>();
            if (_xrOrigin == null)
            {
                _xrOriginObject = new GameObject("XR Origin (Bootstrap)");
                _xrOriginObject.SetActive(false);
                _xrOriginObject.transform.SetParent(null, false);
                _xrOriginObject.transform.position = Vector3.zero;
                _xrOriginObject.transform.rotation = Quaternion.identity;

                _cameraOffsetObject = new GameObject("Camera Offset");
                _cameraOffsetObject.transform.SetParent(_xrOriginObject.transform, false);

                // Preserve the camera's authored standing height so the editor preview is
                // still readable even when no headset pose is overriding the transform.
                mainCamera.transform.SetParent(_cameraOffsetObject.transform, true);

                EnsureTrackedPoseDriver(mainCamera);
                _xrOrigin = _xrOriginObject.AddComponent<XROrigin>();
                _xrOrigin.Camera = mainCamera;
                _xrOrigin.CameraFloorOffsetObject = _cameraOffsetObject;
                _xrOrigin.RequestedTrackingOriginMode = XROrigin.TrackingOriginMode.Floor;
                _xrOriginObject.SetActive(true);
                _ownsXrOrigin = true;
            }
            else
            {
                _xrOriginObject = _xrOrigin.gameObject;
                _cameraOffsetObject = _xrOrigin.CameraFloorOffsetObject != null
                    ? _xrOrigin.CameraFloorOffsetObject
                    : _xrOriginObject;
                if (_xrOrigin.Camera == null)
                {
                    _xrOrigin.Camera = mainCamera;
                }

                if (mainCamera.transform.parent != _cameraOffsetObject.transform &&
                    !mainCamera.transform.IsChildOf(_xrOriginObject.transform))
                {
                    mainCamera.transform.SetParent(_cameraOffsetObject.transform, true);
                }
            }

#if ENABLE_INPUT_SYSTEM
            if (_trackedPoseDriver == null)
#endif
            {
                EnsureTrackedPoseDriver(mainCamera);
            }
        }

        private void EnsureTrackedPoseDriver(Camera mainCamera)
        {
#if ENABLE_INPUT_SYSTEM
            _trackedPoseDriver = mainCamera.GetComponent<TrackedPoseDriver>();
            if (_trackedPoseDriver == null)
            {
                _trackedPoseDriver = mainCamera.gameObject.AddComponent<TrackedPoseDriver>();
                _trackedPoseDriver.trackingType = TrackedPoseDriver.TrackingType.RotationAndPosition;
                _trackedPoseDriver.updateType = TrackedPoseDriver.UpdateType.UpdateAndBeforeRender;
                _trackedPoseDriver.ignoreTrackingState = false;
            }

            if (NeedsBindings(_trackedPoseDriver.positionInput))
            {
                _hmdPositionAction = new InputAction("HMD Center Eye Position", InputActionType.Value, expectedControlType: "Vector3");
                _hmdPositionAction.AddBinding("<XRHMD>/centerEyePosition");
                _hmdPositionAction.AddBinding("<XRHMD>/devicePosition");
                _hmdPositionAction.Enable();
                _trackedPoseDriver.positionInput = new InputActionProperty(_hmdPositionAction);
            }

            if (NeedsBindings(_trackedPoseDriver.rotationInput))
            {
                _hmdRotationAction = new InputAction("HMD Center Eye Rotation", InputActionType.Value, expectedControlType: "Quaternion");
                _hmdRotationAction.AddBinding("<XRHMD>/centerEyeRotation");
                _hmdRotationAction.AddBinding("<XRHMD>/deviceRotation");
                _hmdRotationAction.Enable();
                _trackedPoseDriver.rotationInput = new InputActionProperty(_hmdRotationAction);
            }

            if (NeedsBindings(_trackedPoseDriver.trackingStateInput))
            {
                _hmdTrackingStateAction = new InputAction("HMD Tracking State", InputActionType.Value, expectedControlType: "Integer");
                _hmdTrackingStateAction.AddBinding("<XRHMD>/trackingState");
                _hmdTrackingStateAction.Enable();
                _trackedPoseDriver.trackingStateInput = new InputActionProperty(_hmdTrackingStateAction);
            }
#endif
        }

#if ENABLE_INPUT_SYSTEM
        private static bool NeedsBindings(InputActionProperty property)
        {
            InputAction action = property.action;
            return action == null || action.bindings.Count == 0;
        }
#endif

        private static T FindFirstObjectInSceneOfType<T>() where T : Object
        {
#if UNITY_2023_1_OR_NEWER
            return Object.FindAnyObjectByType<T>(FindObjectsInactive.Include);
#else
            return Object.FindObjectOfType<T>(true);
#endif
        }

        private void ConfigurePassthroughSupport()
        {
            _mainCamera = _xrOrigin != null && _xrOrigin.Camera != null ? _xrOrigin.Camera : Camera.main;
            if (_mainCamera == null)
            {
                return;
            }

            _cameraClearFlags = _mainCamera.clearFlags;
            _cameraBackgroundColor = _mainCamera.backgroundColor;

            _arSession = FindFirstObjectInSceneOfType<ARSession>();
            if (_arSession != null)
            {
                _arSessionObject = _arSession.gameObject;
                _ownsArSession = false;
                _arSession.enabled = false;
                _arSession.attemptUpdate = true;
            }

            _arCameraManager = _mainCamera.GetComponent<ARCameraManager>();
            if (_arCameraManager != null)
            {
                _arCameraManager.enabled = false;
            }
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

        private void SetVisualCue(bool visible, TrackingObject trackingObject, Color color)
        {
            if (!visible || trackingObject == null)
            {
                SetAllVisualCuesActive(false);
                return;
            }

            string mode = NormalizeVisualCueMode(trackingObject.visualCueMode);
            float radius = mapper.PixelsToPanelSize(0f, GetVisualCueRadiusPixels(trackingObject)).y;
            bool filledMode = mode == CueModeCircleFilled;
            bool rubberBandMode = mode == CueModeRubberBand;
            SetActiveIfNeeded(_outboundCueCircle, !filledMode && !rubberBandMode);
            SetActiveIfNeeded(_filledCueCircle, filledMode);
            SetActiveIfNeeded(_rubberBandCue, rubberBandMode);
            switch (mode)
            {
                case CueModeCircleFilled:
                    SetMaterial(_filledCueCircle, color);
                    SetFilledCircleRadius(radius);
                    break;
                case CueModeRubberBand:
                    SetRubberBandCue(trackingObject, color);
                    break;
                case CueModeCircleBorder:
                default:
                    _outboundCueRenderer.material = GetMaterial(color);
                    _outboundCueRenderer.startColor = color;
                    _outboundCueRenderer.endColor = color;
                    SetCircleRadius(radius);
                    break;
            }
        }

        private void SetAllVisualCuesActive(bool active)
        {
            SetActiveIfNeeded(_outboundCueCircle, active);
            SetActiveIfNeeded(_filledCueCircle, active);
            SetActiveIfNeeded(_rubberBandCue, active);
        }

        private static void SetActiveIfNeeded(GameObject go, bool active)
        {
            if (go != null && go.activeSelf != active)
            {
                go.SetActive(active);
            }
        }

        private static string NormalizeVisualCueMode(string visualCueMode)
        {
            if (string.IsNullOrEmpty(visualCueMode))
            {
                return CueModeCircleBorder;
            }

            string normalized = visualCueMode.Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "border":
                case "outline":
                    return CueModeCircleBorder;
                case "filled":
                case "fill":
                    return CueModeCircleFilled;
                case "line":
                case "rubber":
                case "rubber-band":
                    return CueModeRubberBand;
                default:
                    return normalized;
            }
        }

        private static float GetVisualCueRadiusPixels(TrackingObject trackingObject)
        {
            if (trackingObject.visualCueRadiusPixels > 0f)
            {
                return trackingObject.visualCueRadiusPixels;
            }

            float dx = (trackingObject.x - 0.5f) * 640f;
            float dz = (trackingObject.z - 0.5f) * 480f;
            return Mathf.Sqrt(dx * dx + dz * dz) + trackingObject.size * 0.5f;
        }

        private void SetCircleRadius(float radius)
        {
            if (_outboundCueRenderer == null)
            {
                return;
            }

            radius = Mathf.Max(0f, radius);
            if (Mathf.Abs(radius - _lastBorderCueRadius) < 0.0001f)
            {
                return;
            }

            _lastBorderCueRadius = radius;
            for (int i = 0; i < CueCircleSegments; i++)
            {
                float angle = i * Mathf.PI * 2f / CueCircleSegments;
                _outboundCueRenderer.SetPosition(i, new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0f));
            }
        }

        private void SetFilledCircleRadius(float radius)
        {
            if (_filledCueMeshFilter == null)
            {
                return;
            }

            radius = Mathf.Max(0f, radius);
            if (Mathf.Abs(radius - _lastFilledCueRadius) < 0.0001f)
            {
                return;
            }

            _lastFilledCueRadius = radius;
            var vertices = new Vector3[CueCircleSegments + 1];
            var triangles = new int[CueCircleSegments * 3];
            vertices[0] = Vector3.zero;
            for (int i = 0; i < CueCircleSegments; i++)
            {
                float angle = i * Mathf.PI * 2f / CueCircleSegments;
                vertices[i + 1] = new Vector3(Mathf.Cos(angle) * radius, Mathf.Sin(angle) * radius, 0f);
            }

            for (int i = 0; i < CueCircleSegments; i++)
            {
                int triangleIndex = i * 3;
                triangles[triangleIndex] = 0;
                triangles[triangleIndex + 1] = i + 1;
                triangles[triangleIndex + 2] = i == CueCircleSegments - 1 ? 1 : i + 2;
            }

            Mesh mesh = _filledCueMeshFilter.sharedMesh;
            if (mesh == null)
            {
                mesh = new Mesh { name = "Filled Circle Cue Mesh" };
                _filledCueMeshFilter.sharedMesh = mesh;
            }
            mesh.Clear();
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateBounds();
        }

        private void SetRubberBandCue(TrackingObject trackingObject, Color color)
        {
            if (_rubberBandRenderer == null)
            {
                return;
            }

            _rubberBandRenderer.material = GetMaterial(color);
            _rubberBandRenderer.startColor = color;
            _rubberBandRenderer.endColor = color;
            Vector3 objectPosition = mapper.NormalizedToLocalUnclamped(trackingObject.x, trackingObject.z, 0f);
            Vector3 rubberBandEnd = new Vector3(objectPosition.x, objectPosition.y, 0f);
            if ((_lastRubberBandEnd - rubberBandEnd).sqrMagnitude < 0.00000001f)
            {
                return;
            }

            _lastRubberBandEnd = rubberBandEnd;
            _rubberBandRenderer.SetPosition(0, Vector3.zero);
            _rubberBandRenderer.SetPosition(1, rubberBandEnd);
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

            renderer.sharedMaterial = GetMaterial(color);
        }

        private Material GetMaterial(Color color)
        {
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

            return material;
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

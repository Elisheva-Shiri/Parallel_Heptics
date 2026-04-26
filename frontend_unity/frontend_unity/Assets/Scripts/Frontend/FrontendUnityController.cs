using System.Collections.Generic;
using UnityEngine;

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
        private static readonly Color DarkBarColor = new Color32(64, 64, 64, 255);
        private static readonly Color LightGreenColor = new Color32(144, 238, 144, 255);
        private static readonly Color GreenColor = new Color32(0, 200, 0, 255);

        [Header("Backend protocol")]
        [SerializeField] private string backendHost = "localhost";
        [SerializeField] private int udpListenPort = 12346;
        [SerializeField] private int tcpBackendPort = 12344;
        [SerializeField] private bool bindUdpLoopbackOnly;

        [Header("Panel")]
        [SerializeField] private FlatPanelMapper mapper = new FlatPanelMapper();
        [SerializeField] private Vector3 panelWorldPosition = new Vector3(0f, 1.45f, 5.2f);
        [SerializeField] private Vector3 panelEulerAngles = Vector3.zero;
        [SerializeField] private int maxFingerObjects = 10;
        [SerializeField] private float fingerDiameter = 0.08f;
        [SerializeField] private float holdSeconds = 1.0f;
        [SerializeField] private bool showDebugStatus;

        private readonly List<GameObject> _fingerObjects = new List<GameObject>();
        private readonly Dictionary<Color, Material> _materials = new Dictionary<Color, Material>();
        private ExperimentUdpReceiver _receiver;
        private ExperimentControlClient _controlClient;
        private HoldButtonSelector _holdSelector;
        private Transform _panelRoot;
        private Transform _dynamicRoot;
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
        private ExperimentState _lastState = (ExperimentState)999;
        private float _lastPacketTime = -1000f;
        private long _lastMalformedLogFrame = -9999;

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
            BuildSceneGraph();
            SetStateVisibility(ExperimentState.Start);
        }

        private void Update()
        {
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

            _trackingObject.transform.localPosition = mapper.NormalizedToLocal(trackingObject.x, trackingObject.z, -0.05f);
            Vector2 objectSize = mapper.PixelsToPanelSize(trackingObject.size, trackingObject.size);
            _trackingObject.transform.localScale = new Vector3(objectSize.x, objectSize.y, 0.035f);
            Color color = trackingObject.isPinched ? (trackingObject.pairIndex == 0 ? FirstColor : SecondColor) : GrayColor;
            SetMaterial(_trackingObject, color);

            const float barWidth = 2.0f;
            const float barHeight = 0.16f;
            float halfWidth = barWidth * 0.5f;
            _progressBackground.transform.localScale = new Vector3(barWidth, barHeight, 1f);
            SetBar(_outboundProgress, Mathf.Clamp01(trackingObject.progress) * halfWidth, barHeight, -barWidth * 0.25f);
            SetBar(_returnProgress, Mathf.Clamp01(trackingObject.returnProgress) * halfWidth, barHeight, barWidth * 0.25f);
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

            const float holdBarHeight = 0.08f;
            SetBar(_leftHoldFill, _leftButton.transform.localScale.x * _holdSelector.LeftProgress, holdBarHeight, 0f, true, _leftHoldBackground.transform);
            SetBar(_rightHoldFill, _rightButton.transform.localScale.x * _holdSelector.RightProgress, holdBarHeight, 0f, true, _rightHoldBackground.transform);
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
                    _fingerObjects[i].transform.localPosition = mapper.NormalizedToLocal(landmark.x, landmark.z, -0.08f);
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
            _panelRoot.SetParent(transform, false);
            _panelRoot.position = panelWorldPosition;
            _panelRoot.eulerAngles = panelEulerAngles;

            _dynamicRoot = new GameObject("Dynamic Experiment Elements").transform;
            _dynamicRoot.SetParent(_panelRoot, false);

            CreateQuad("Black Panel", _panelRoot, new Vector3(0f, 0f, 0.02f), new Vector3(mapper.PanelWidth, mapper.PanelHeight, 1f), PanelColor);
            _trackingObject = CreateCube("Tracking Object", _dynamicRoot, Vector3.zero, Vector3.one, GrayColor);

            float progressY = mapper.PanelHeight * 0.5f - 0.25f;
            _progressBackground = CreateQuad("Progress Background", _dynamicRoot, new Vector3(0f, progressY, -0.06f), Vector3.one, DarkBarColor);
            _outboundProgress = CreateQuad("Outbound Progress", _dynamicRoot, new Vector3(-0.5f, progressY, -0.07f), Vector3.one, LightGreenColor);
            _returnProgress = CreateQuad("Return Progress", _dynamicRoot, new Vector3(0.5f, progressY, -0.07f), Vector3.one, GreenColor);
            _counterText = CreateText("Cycle Counter", _dynamicRoot, new Vector3(mapper.PanelWidth * 0.5f - 0.5f, progressY, -0.09f), 0.033f, TextAnchor.MiddleCenter);

            Vector2 buttonSize = mapper.PixelsToPanelSize(100f, 100f);
            float buttonInset = mapper.PixelsToPanelSize(140f, 0f).x;
            _leftButton = CreateQuad("Question Left Button", _dynamicRoot, new Vector3(-mapper.PanelWidth * 0.5f + buttonInset, 0f, -0.06f), new Vector3(buttonSize.x, buttonSize.y, 1f), FirstColor);
            _rightButton = CreateQuad("Question Right Button", _dynamicRoot, new Vector3(mapper.PanelWidth * 0.5f - buttonInset, 0f, -0.06f), new Vector3(buttonSize.x, buttonSize.y, 1f), SecondColor);
            _leftHoldBackground = CreateQuad("Left Hold Background", _dynamicRoot, _leftButton.transform.localPosition + new Vector3(0f, -buttonSize.y * 0.65f, -0.01f), new Vector3(buttonSize.x, 0.08f, 1f), DarkBarColor);
            _rightHoldBackground = CreateQuad("Right Hold Background", _dynamicRoot, _rightButton.transform.localPosition + new Vector3(0f, -buttonSize.y * 0.65f, -0.01f), new Vector3(buttonSize.x, 0.08f, 1f), DarkBarColor);
            _leftHoldFill = CreateQuad("Left Hold Fill", _leftHoldBackground.transform, Vector3.zero, new Vector3(0f, 0.08f, 1f), GreenColor);
            _rightHoldFill = CreateQuad("Right Hold Fill", _rightHoldBackground.transform, Vector3.zero, new Vector3(0f, 0.08f, 1f), GreenColor);

            _titleText = CreateText("Title Text", _dynamicRoot, new Vector3(0f, mapper.PanelHeight * 0.5f - 0.5f, -0.09f), 0.039f, TextAnchor.MiddleCenter);
            _subtitleText = CreateText("Subtitle Text", _dynamicRoot, new Vector3(0f, -0.2f, -0.09f), 0.027f, TextAnchor.MiddleCenter);
            _statusText = CreateText("Status Text", _panelRoot, new Vector3(0f, -mapper.PanelHeight * 0.5f + 0.18f, -0.09f), 0.0135f, TextAnchor.MiddleCenter);
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
            Destroy(go.GetComponent<Collider>());
            SetMaterial(go, color);
            return go;
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

        private void SetBar(GameObject bar, float width, float height, float centerX, bool alignLeft = false, Transform parent = null)
        {
            width = Mathf.Max(0f, width);
            bar.transform.localScale = new Vector3(width, height, 1f);
            float x = alignLeft && parent != null ? -parent.localScale.x * 0.5f + width * 0.5f : centerX;
            bar.transform.localPosition = new Vector3(x, 0f, -0.01f);
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

                _materials.Add(color, material);
            }

            renderer.sharedMaterial = material;
        }
    }
}

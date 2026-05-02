using NUnit.Framework;
using ParallelHeptics.FrontendUnity;
using System.Reflection;
using Unity.XR.CoreUtils;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem.XR;
#endif

namespace ParallelHeptics.FrontendUnity.Tests
{
    public sealed class FrontendUnityTests
    {
        private static GameObject FindDynamicChild(GameObject panelRoot, string childName)
        {
            Transform child = panelRoot.transform.Find($"Dynamic Experiment Elements/{childName}");
            return child == null ? null : child.gameObject;
        }

        [Test]
        public void ExperimentPacketJsonMatchesBackendShape()
        {
            const string json = "{\"stateData\":{\"state\":1,\"pauseTime\":0},\"landmarks\":[{\"x\":0.25,\"z\":0.75}],\"trackingObject\":{\"x\":0.5,\"z\":0.4,\"size\":40.0,\"isPinched\":true,\"progress\":0.25,\"returnProgress\":0.5,\"cycleCount\":1,\"targetCycleCount\":2,\"pairIndex\":0},\"playWhiteNoise\":true,\"isDebug\":false}";

            ExperimentPacket packet = JsonUtility.FromJson<ExperimentPacket>(json);

            Assert.NotNull(packet);
            Assert.AreEqual((int)ExperimentState.Comparison, packet.stateData.state);
            Assert.AreEqual(1, packet.landmarks.Count);
            Assert.AreEqual(0.25f, packet.landmarks[0].x, 0.0001f);
            Assert.IsTrue(packet.trackingObject.isPinched);
            Assert.AreEqual(0.5f, packet.trackingObject.returnProgress, 0.0001f);
            Assert.IsTrue(packet.playWhiteNoise);
            Assert.IsFalse(packet.isDebug);
        }

        [Test]
        public void ExperimentControlJsonMatchesBackendShape()
        {
            string json = JsonUtility.ToJson(new ExperimentControl { questionInput = (int)QuestionInput.Right });

            Assert.AreEqual("{\"questionInput\":1}", json);
        }

        [Test]
        public void FlatPanelMapperMatchesPygameNormalizedCoordinateConvention()
        {
            var mapper = new FlatPanelMapper();

            Vector2 center = mapper.NormalizedToPanel2D(0.5f, 0.5f);
            Vector2 upperLeft = mapper.NormalizedToPanel2D(0f, 0f);

            Assert.AreEqual(2.0f, mapper.PanelWidth, 0.0001f);
            Assert.AreEqual(2.0f, mapper.PanelHeight, 0.0001f);
            Assert.AreEqual(Vector2.zero, center);
            Assert.Less(upperLeft.x, 0f);
            Assert.Less(upperLeft.y, 0f);
        }

        [Test]
        public void FlatPanelMapperCanBeConfiguredToPhysicalTableSize()
        {
            var mapper = new FlatPanelMapper();

            mapper.SetPanelSizeMeters(0.2f, 0.2f);

            Assert.AreEqual(new Vector2(-0.1f, -0.1f), mapper.NormalizedToPanel2D(0f, 0f));
            Assert.AreEqual(new Vector2(0.1f, 0.1f), mapper.NormalizedToPanel2D(1f, 1f));
        }

        [Test]
        public void HoldButtonSelectorSendsOncePerContinuousHold()
        {
            var selector = new HoldButtonSelector(1f);

            Assert.IsNull(selector.Update(leftTouched: true, rightTouched: false, deltaTime: 0.5f));
            Assert.AreEqual(QuestionInput.Left, selector.Update(leftTouched: true, rightTouched: false, deltaTime: 0.5f));
            Assert.IsNull(selector.Update(leftTouched: true, rightTouched: false, deltaTime: 1f));
            Assert.IsNull(selector.Update(leftTouched: false, rightTouched: false, deltaTime: 0.1f));
            Assert.IsNull(selector.Update(leftTouched: true, rightTouched: false, deltaTime: 0.5f));
            Assert.AreEqual(QuestionInput.Left, selector.Update(leftTouched: true, rightTouched: false, deltaTime: 0.5f));
        }

        [Test]
        public void FrontendUnitySceneContainsConfiguredController()
        {
            EditorSceneManager.OpenScene("Assets/Scenes/FrontendUnity.unity");

            var app = GameObject.Find("FrontendUnityApp");

            Assert.NotNull(app);
            Assert.NotNull(app.GetComponent<FrontendUnityController>());
            Assert.NotNull(app.GetComponent<ExperimentUdpReceiver>());
            Assert.NotNull(app.GetComponent<ExperimentControlClient>());
        }

        [Test]
        public void FrontendUnitySceneDefaultsToHorizontalLargeTabletop()
        {
            EditorSceneManager.OpenScene("Assets/Scenes/FrontendUnity.unity");

            var controller = GameObject.Find("FrontendUnityApp").GetComponent<FrontendUnityController>();
            var serialized = new SerializedObject(controller);
            SerializedProperty mapperProperty = serialized.FindProperty("mapper");

            Assert.AreEqual(2.0f, mapperProperty.FindPropertyRelative("panelWidth").floatValue, 0.0001f);
            Assert.AreEqual(2.0f, mapperProperty.FindPropertyRelative("panelHeight").floatValue, 0.0001f);
            Assert.AreEqual(-90f, serialized.FindProperty("panelEulerAngles").vector3Value.x, 0.0001f);
            Assert.AreEqual(0.1f, serialized.FindProperty("fingerDiameter").floatValue, 0.0001f);
            Assert.IsTrue(serialized.FindProperty("enableKeyboardCalibration").boolValue);
            Assert.IsTrue(serialized.FindProperty("calibrationActive").boolValue);
            Assert.IsNull(serialized.FindProperty("playWhiteNoise"));
            Assert.AreEqual(0.15f, serialized.FindProperty("whiteNoiseVolume").floatValue, 0.0001f);
            Assert.AreEqual(44100, serialized.FindProperty("whiteNoiseSampleRate").intValue);
        }

        [Test]
        public void RuntimePanelIsWorldRootAndRestoresVisibleExperimentObjects()
        {
            GameObject controllerObject = new GameObject("Runtime Panel Test Controller");
            GameObject panelRoot = null;

            try
            {
                var controller = controllerObject.AddComponent<FrontendUnityController>();
                MethodInfo buildSceneGraph = typeof(FrontendUnityController).GetMethod("BuildSceneGraph", BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.NotNull(buildSceneGraph);
                buildSceneGraph.Invoke(controller, null);

                panelRoot = GameObject.Find("Flat VR Frontend Panel");
                Assert.NotNull(panelRoot);
                Assert.IsNull(panelRoot.transform.parent, "The tabletop panel must be a scene root, not a child of the XR Origin or camera.");
                Transform dynamicRoot = panelRoot.transform.Find("Dynamic Experiment Elements");
                Assert.NotNull(dynamicRoot);

                Assert.NotNull(dynamicRoot.Find("Outbound Edge Cue"));
                Assert.NotNull(dynamicRoot.Find("Center Return Point"));
                Assert.NotNull(dynamicRoot.Find("Progress Background"));
                Assert.NotNull(dynamicRoot.Find("Outbound Progress"));
                Assert.NotNull(dynamicRoot.Find("Return Progress"));
                Assert.NotNull(FindDynamicChild(panelRoot, "Question Left Button"));
                Assert.NotNull(FindDynamicChild(panelRoot, "Question Right Button"));

                var counter = FindDynamicChild(panelRoot, "Cycle Counter").GetComponent<TextMesh>();
                Assert.NotNull(counter);
                Assert.AreEqual(0f, counter.transform.localEulerAngles.z, 0.0001f, "Counter text orientation must come from a Y-scale flip, not an in-plane rotation that would reverse reading order.");
                Assert.AreEqual(-1f, counter.transform.localScale.y, 0.0001f, "Counter text needs Y-scale -1 so the horizontal tabletop reads right-side up without mirroring the reading direction.");
                Assert.AreEqual(0.025f, counter.characterSize, 0.0001f, "Counter should be half of the previous 0.05 m tabletop text size.");
            }
            finally
            {
                if (panelRoot != null)
                {
                    Object.DestroyImmediate(panelRoot);
                }

                Object.DestroyImmediate(controllerObject);
            }
        }

        [Test]
        public void RenderPacketWithWhiteNoiseStillShowsComparisonObjects()
        {
            GameObject controllerObject = new GameObject("White Noise Render Test Controller");
            GameObject panelRoot = null;

            try
            {
                var controller = controllerObject.AddComponent<FrontendUnityController>();
                MethodInfo buildSceneGraph = typeof(FrontendUnityController).GetMethod("BuildSceneGraph", BindingFlags.Instance | BindingFlags.NonPublic);
                MethodInfo renderPacket = typeof(FrontendUnityController).GetMethod("RenderPacket", BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.NotNull(buildSceneGraph);
                Assert.NotNull(renderPacket);
                Assert.IsNull(controllerObject.GetComponent<AudioSource>(), "The test must start without scene-authored audio setup.");
                buildSceneGraph.Invoke(controller, null);

                var packet = new ExperimentPacket
                {
                    stateData = new StateData { state = (int)ExperimentState.Comparison, pauseTime = 0 },
                    landmarks = new System.Collections.Generic.List<FingerPosition>
                    {
                        new FingerPosition { x = 0.25f, z = 0.75f }
                    },
                    trackingObject = new TrackingObject
                    {
                        x = 0.5f,
                        z = 0.4f,
                        size = 40f,
                        isPinched = true,
                        progress = 0.25f,
                        returnProgress = 0.5f,
                        cycleCount = 1,
                        targetCycleCount = 2,
                        pairIndex = 0
                    },
                    playWhiteNoise = true
                };

                renderPacket.Invoke(controller, new object[] { packet });

                panelRoot = GameObject.Find("Flat VR Frontend Panel");
                Assert.NotNull(panelRoot);
                Assert.IsTrue(FindDynamicChild(panelRoot, "Tracking Object").activeSelf);
                Assert.IsFalse(FindDynamicChild(panelRoot, "Outbound Edge Cue").activeSelf);
                Assert.IsTrue(FindDynamicChild(panelRoot, "Center Return Point").activeSelf);
                Assert.IsTrue(FindDynamicChild(panelRoot, "Progress Background").activeSelf);
                Assert.IsTrue(FindDynamicChild(panelRoot, "Return Progress").activeSelf);
                Assert.IsTrue(FindDynamicChild(panelRoot, "Finger Landmark 1").activeSelf);

                AudioSource audioSource = controllerObject.GetComponent<AudioSource>();
                Assert.NotNull(audioSource, "White-noise playback must create its own AudioSource at runtime.");
                Assert.NotNull(audioSource.clip, "White-noise playback must generate a procedural clip at runtime.");
                Assert.IsTrue(audioSource.loop);
                Assert.AreEqual(0f, audioSource.spatialBlend, 0.0001f);
                Assert.AreEqual(1f, audioSource.volume, 0.0001f);
            }
            finally
            {
                if (panelRoot != null)
                {
                    Object.DestroyImmediate(panelRoot);
                }

                Object.DestroyImmediate(controllerObject);
            }
        }

        [Test]
        public void ComparisonCuesSwitchDirectionAndHideOutsideComparison()
        {
            GameObject controllerObject = new GameObject("Comparison Cue Test Controller");
            GameObject panelRoot = null;

            try
            {
                var controller = controllerObject.AddComponent<FrontendUnityController>();
                MethodInfo buildSceneGraph = typeof(FrontendUnityController).GetMethod("BuildSceneGraph", BindingFlags.Instance | BindingFlags.NonPublic);
                MethodInfo renderPacket = typeof(FrontendUnityController).GetMethod("RenderPacket", BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.NotNull(buildSceneGraph);
                Assert.NotNull(renderPacket);
                buildSceneGraph.Invoke(controller, null);

                var outboundPacket = new ExperimentPacket
                {
                    stateData = new StateData { state = (int)ExperimentState.Comparison, pauseTime = 0 },
                    landmarks = new System.Collections.Generic.List<FingerPosition>(),
                    trackingObject = new TrackingObject
                    {
                        x = 0.55f,
                        z = 0.5f,
                        size = 40f,
                        isPinched = true,
                        progress = 0.4f,
                        returnProgress = 0f,
                        cycleCount = 0,
                        targetCycleCount = 2,
                        pairIndex = 0
                    },
                    playWhiteNoise = false
                };

                renderPacket.Invoke(controller, new object[] { outboundPacket });

                panelRoot = GameObject.Find("Flat VR Frontend Panel");
                Assert.NotNull(panelRoot);
                GameObject outboundCue = FindDynamicChild(panelRoot, "Outbound Edge Cue");
                Assert.IsTrue(outboundCue.activeSelf);
                Assert.IsFalse(FindDynamicChild(panelRoot, "Center Return Point").activeSelf);
                Assert.IsTrue(FindDynamicChild(panelRoot, "Progress Background").activeSelf);
                Assert.IsTrue(FindDynamicChild(panelRoot, "Outbound Progress").activeSelf);

                var outboundRenderer = outboundCue.GetComponent<LineRenderer>();
                Assert.NotNull(outboundRenderer);
                float initialRadius = outboundRenderer.GetPosition(0).x;
                outboundPacket.trackingObject.progress = 0.8f;
                renderPacket.Invoke(controller, new object[] { outboundPacket });
                Assert.AreEqual(initialRadius, outboundRenderer.GetPosition(0).x, 0.0001f, "The outbound cue is a fixed max-range target ring; progress changes only the bar.");

                var questionPacket = new ExperimentPacket
                {
                    stateData = new StateData { state = (int)ExperimentState.Question, pauseTime = 0 },
                    landmarks = new System.Collections.Generic.List<FingerPosition>(),
                    trackingObject = outboundPacket.trackingObject,
                    playWhiteNoise = false
                };

                renderPacket.Invoke(controller, new object[] { questionPacket });

                Assert.IsFalse(FindDynamicChild(panelRoot, "Outbound Edge Cue").activeSelf);
                Assert.IsFalse(FindDynamicChild(panelRoot, "Center Return Point").activeSelf);
                Assert.IsFalse(FindDynamicChild(panelRoot, "Progress Background").activeSelf);
                Assert.IsFalse(FindDynamicChild(panelRoot, "Outbound Progress").activeSelf);
                Assert.IsFalse(FindDynamicChild(panelRoot, "Return Progress").activeSelf);
            }
            finally
            {
                if (panelRoot != null)
                {
                    Object.DestroyImmediate(panelRoot);
                }

                Object.DestroyImmediate(controllerObject);
            }
        }

        [Test]
        public void RuntimeBootstrapCreatesXrOriginWithTrackedHeadset()
        {
            EditorSceneManager.OpenScene("Assets/Scenes/FrontendUnity.unity");
            Assert.NotNull(Camera.main, "FrontendUnity.unity must contain a MainCamera for the XR rig bootstrap to attach to.");

            GameObject controllerObject = new GameObject("Runtime Xr Rig Test Controller");
            XROrigin createdOrigin = null;

            try
            {
                var controller = controllerObject.AddComponent<FrontendUnityController>();
                MethodInfo ensureXrCameraRig = typeof(FrontendUnityController).GetMethod("EnsureXrCameraRig", BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.NotNull(ensureXrCameraRig, "EnsureXrCameraRig must exist so passthrough+head tracking are wired up before the scene graph is built.");
                ensureXrCameraRig.Invoke(controller, null);

                createdOrigin = Object.FindAnyObjectByType<XROrigin>();
                Assert.NotNull(createdOrigin, "Bootstrap must add an XR Origin so the headset moves the camera through a stable world rather than dragging the world along.");

                Camera rigCamera = createdOrigin.Camera;
                Assert.NotNull(rigCamera, "XR Origin must point at the camera it is supposed to drive.");
                Assert.AreEqual("MainCamera", rigCamera.tag, "The XR Origin must claim the scene's MainCamera-tagged camera so head pose drives the user's view, not a stray camera.");
                Assert.IsTrue(rigCamera.transform.IsChildOf(createdOrigin.transform), "The driven camera must be reparented under the XR Origin hierarchy so HMD pose moves it.");
#if ENABLE_INPUT_SYSTEM
                var driver = rigCamera.GetComponent<TrackedPoseDriver>();
                Assert.NotNull(driver, "The driven camera must have a TrackedPoseDriver so HMD pose updates the camera transform every frame.");
                Assert.IsTrue(driver.positionInput.action != null && driver.positionInput.action.bindings.Count > 0, "TrackedPoseDriver position input must be bound to an XR HMD control.");
                Assert.IsTrue(driver.rotationInput.action != null && driver.rotationInput.action.bindings.Count > 0, "TrackedPoseDriver rotation input must be bound to an XR HMD control.");
#endif
            }
            finally
            {
                if (createdOrigin != null)
                {
                    Object.DestroyImmediate(createdOrigin.gameObject);
                }
                Object.DestroyImmediate(controllerObject);
            }
        }
    }
}

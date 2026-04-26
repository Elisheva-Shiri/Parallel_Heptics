using NUnit.Framework;
using ParallelHeptics.FrontendUnity;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace ParallelHeptics.FrontendUnity.Tests
{
    public sealed class FrontendUnityTests
    {
        [Test]
        public void ExperimentPacketJsonMatchesBackendShape()
        {
            const string json = "{\"stateData\":{\"state\":1,\"pauseTime\":0},\"landmarks\":[{\"x\":0.25,\"z\":0.75}],\"trackingObject\":{\"x\":0.5,\"z\":0.4,\"size\":40.0,\"isPinched\":true,\"progress\":0.25,\"returnProgress\":0.5,\"cycleCount\":1,\"targetCycleCount\":2,\"pairIndex\":0}}";

            ExperimentPacket packet = JsonUtility.FromJson<ExperimentPacket>(json);

            Assert.NotNull(packet);
            Assert.AreEqual((int)ExperimentState.Comparison, packet.stateData.state);
            Assert.AreEqual(1, packet.landmarks.Count);
            Assert.AreEqual(0.25f, packet.landmarks[0].x, 0.0001f);
            Assert.IsTrue(packet.trackingObject.isPinched);
            Assert.AreEqual(0.5f, packet.trackingObject.returnProgress, 0.0001f);
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

                Assert.NotNull(GameObject.Find("Outbound Progress"));
                Assert.NotNull(GameObject.Find("Return Progress"));
                Assert.NotNull(GameObject.Find("Question Left Button"));
                Assert.NotNull(GameObject.Find("Question Right Button"));

                var counter = GameObject.Find("Cycle Counter").GetComponent<TextMesh>();
                Assert.NotNull(counter);
                Assert.AreEqual(0f, counter.transform.localEulerAngles.z, 0.0001f, "Counter text should not be mirrored by a 180-degree local rotation.");
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
    }
}

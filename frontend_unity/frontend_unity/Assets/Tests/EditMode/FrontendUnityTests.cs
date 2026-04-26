using NUnit.Framework;
using ParallelHeptics.FrontendUnity;
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

            Assert.AreEqual(Vector2.zero, center);
            Assert.Less(upperLeft.x, 0f);
            Assert.Greater(upperLeft.y, 0f);
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
    }
}

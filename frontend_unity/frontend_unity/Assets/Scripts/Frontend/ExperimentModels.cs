using System;
using System.Collections.Generic;

namespace ParallelHeptics.FrontendUnity
{
    public enum ExperimentState
    {
        Start = 0,
        Comparison = 1,
        Question = 2,
        Pause = 3,
        Break = 4,
        End = -1
    }

    public enum QuestionInput
    {
        Left = 0,
        Right = 1
    }

    [Serializable]
    public sealed class FingerPosition
    {
        public float x;
        public float z;
    }

    [Serializable]
    public sealed class TrackingObject
    {
        public float x;
        public float z;
        public float size;
        public bool isPinched;
        public float progress;
        public float returnProgress;
        public int cycleCount;
        public int targetCycleCount;
        public int pairIndex;
    }

    [Serializable]
    public sealed class StateData
    {
        public int state;
        public int pauseTime;
    }

    [Serializable]
    public sealed class ExperimentPacket
    {
        public StateData stateData;
        public List<FingerPosition> landmarks;
        public TrackingObject trackingObject;
        public bool playWhiteNoise;
    }

    [Serializable]
    public sealed class ExperimentControl
    {
        public int questionInput;
    }
}

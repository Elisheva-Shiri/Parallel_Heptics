using UnityEngine;

namespace ParallelHeptics.FrontendUnity
{
    /// <summary>
    /// Pygame-equivalent hold-to-select logic driven by backend-provided landmarks.
    /// Uses seconds instead of frame counts so timing is stable at 72/90/120 Hz.
    /// </summary>
    public sealed class HoldButtonSelector
    {
        private readonly float _holdSeconds;
        private float _leftTimer;
        private float _rightTimer;
        private bool _leftSent;
        private bool _rightSent;

        public HoldButtonSelector(float holdSeconds)
        {
            _holdSeconds = Mathf.Max(0.01f, holdSeconds);
        }

        public float LeftProgress => Mathf.Clamp01(_leftTimer / _holdSeconds);
        public float RightProgress => Mathf.Clamp01(_rightTimer / _holdSeconds);

        public QuestionInput? Update(bool leftTouched, bool rightTouched, float deltaTime)
        {
            QuestionInput? answer = null;

            if (leftTouched)
            {
                _leftTimer = Mathf.Min(_holdSeconds, _leftTimer + deltaTime);
                if (!_leftSent && _leftTimer >= _holdSeconds)
                {
                    _leftSent = true;
                    answer = QuestionInput.Left;
                }
            }
            else
            {
                _leftTimer = 0f;
                _leftSent = false;
            }

            if (rightTouched)
            {
                _rightTimer = Mathf.Min(_holdSeconds, _rightTimer + deltaTime);
                if (!_rightSent && _rightTimer >= _holdSeconds)
                {
                    _rightSent = true;
                    answer = QuestionInput.Right;
                }
            }
            else
            {
                _rightTimer = 0f;
                _rightSent = false;
            }

            return answer;
        }

        public void Reset()
        {
            _leftTimer = 0f;
            _rightTimer = 0f;
            _leftSent = false;
            _rightSent = false;
        }
    }
}

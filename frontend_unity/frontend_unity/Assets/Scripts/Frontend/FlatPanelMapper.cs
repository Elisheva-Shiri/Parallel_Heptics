using UnityEngine;

namespace ParallelHeptics.FrontendUnity
{
    [System.Serializable]
    public sealed class FlatPanelMapper
    {
        [SerializeField] private float panelWidth = 2.0f;
        [SerializeField] private float panelHeight = 2.0f;
        [SerializeField] private float backendWidthPixels = 640f;
        [SerializeField] private float backendHeightPixels = 480f;

        public float PanelWidth => panelWidth;
        public float PanelHeight => panelHeight;

        public void SetPanelSizeMeters(float widthMeters, float heightMeters)
        {
            panelWidth = Mathf.Max(0.01f, widthMeters);
            panelHeight = Mathf.Max(0.01f, heightMeters);
        }

        public Vector3 NormalizedToLocal(float x, float z, float depthOffset = -0.02f)
        {
            return new Vector3((Mathf.Clamp01(x) - 0.5f) * panelWidth, (Mathf.Clamp01(z) - 0.5f) * panelHeight, depthOffset);
        }

        public Vector2 NormalizedToPanel2D(float x, float z)
        {
            return new Vector2((Mathf.Clamp01(x) - 0.5f) * panelWidth, (Mathf.Clamp01(z) - 0.5f) * panelHeight);
        }

        public Vector2 PixelsToPanelSize(float widthPixels, float heightPixels)
        {
            float width = backendWidthPixels <= 0f ? widthPixels : widthPixels / backendWidthPixels * panelWidth;
            float height = backendHeightPixels <= 0f ? heightPixels : heightPixels / backendHeightPixels * panelHeight;
            return new Vector2(width, height);
        }
    }
}

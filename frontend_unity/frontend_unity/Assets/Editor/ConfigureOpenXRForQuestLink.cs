using System.IO;
using UnityEditor;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEditor.XR.OpenXR.Features;
using UnityEngine;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features;

namespace ParallelHeptics.FrontendUnity.Editor
{
    public static class ConfigureOpenXRForQuestLink
    {
        private const string XrFolder = "Assets/XR";
        private const string GeneralSettingsPath = XrFolder + "/XRGeneralSettingsPerBuildTarget.asset";
        private const string OpenXrLoaderTypeName = "UnityEngine.XR.OpenXR.OpenXRLoader";
        private const string MetaOpenXrLifeCycleFeatureId = "MetaOpenXR-OpenXRLifeCycle";
        private static readonly string[] QuestLinkPassthroughFeatureIds =
        {
            "com.unity.openxr.feature.arfoundation-meta-session",
            "com.unity.openxr.feature.arfoundation-meta-camera",
            "com.unity.openxr.feature.meta-display-utilities"
        };
        private static readonly string[] QuestLinkControllerFeatureIds =
        {
            "com.unity.openxr.feature.input.oculustouch",
            "com.unity.openxr.feature.input.metaquestplus",
            "com.unity.openxr.feature.input.metaquestpro",
            "com.unity.openxr.feature.input.khrsimpleprofile"
        };

        [MenuItem("Parallel Heptics/Configure OpenXR for Quest Link")]
        public static void Configure()
        {
            EnsureFolder(XrFolder);

            if (!EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey, out XRGeneralSettingsPerBuildTarget perBuildTargetSettings))
            {
                perBuildTargetSettings = AssetDatabase.LoadAssetAtPath<XRGeneralSettingsPerBuildTarget>(GeneralSettingsPath);
                if (perBuildTargetSettings == null)
                {
                    perBuildTargetSettings = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
                    AssetDatabase.CreateAsset(perBuildTargetSettings, GeneralSettingsPath);
                }

                EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey, perBuildTargetSettings, true);
            }

            BuildTargetGroup buildTargetGroup = BuildTargetGroup.Standalone;
            if (!perBuildTargetSettings.HasSettingsForBuildTarget(buildTargetGroup))
            {
                perBuildTargetSettings.CreateDefaultSettingsForBuildTarget(buildTargetGroup);
            }

            if (!perBuildTargetSettings.HasManagerSettingsForBuildTarget(buildTargetGroup))
            {
                perBuildTargetSettings.CreateDefaultManagerSettingsForBuildTarget(buildTargetGroup);
            }

            XRGeneralSettings generalSettings = perBuildTargetSettings.SettingsForBuildTarget(buildTargetGroup);
            generalSettings.InitManagerOnStart = true;
            generalSettings.AssignedSettings.automaticLoading = true;
            generalSettings.AssignedSettings.automaticRunning = true;

            bool assigned = XRPackageMetadataStore.AssignLoader(generalSettings.AssignedSettings, OpenXrLoaderTypeName, buildTargetGroup);
            int enabledFeatureCount = EnableQuestLinkFeatures(buildTargetGroup);
            EditorUtility.SetDirty(perBuildTargetSettings);
            EditorUtility.SetDirty(generalSettings);
            EditorUtility.SetDirty(generalSettings.AssignedSettings);
            AssetDatabase.SaveAssets();

            Debug.Log(assigned
                ? $"Configured OpenXR for PC/Standalone Quest Link and enabled {enabledFeatureCount} Quest Link feature(s), including controller input profiles. Restart Play Mode if it was already running."
                : $"OpenXR loader was already configured or could not be reassigned; enabled {enabledFeatureCount} Quest Link feature(s), including controller input profiles. Check Project Settings > XR Plug-in Management > PC, Mac & Linux Standalone.");
        }

        private static int EnableQuestLinkFeatures(BuildTargetGroup buildTargetGroup)
        {
            OpenXRSettings settings = OpenXRSettings.GetSettingsForBuildTargetGroup(buildTargetGroup);
            if (settings == null)
            {
                Debug.LogWarning("OpenXR settings were not available; Quest Link passthrough/controller features could not be enabled.");
                return 0;
            }

            FeatureHelpers.RefreshFeatures(buildTargetGroup);
            int enabledPassthroughCount = 0;
            int enabledControllerCount = 0;
            OpenXRFeature[] features = FeatureHelpers.GetFeaturesWithIdsForBuildTarget(buildTargetGroup, QuestLinkPassthroughFeatureIds);
            foreach (OpenXRFeature feature in features)
            {
                enabledPassthroughCount += SetFeatureEnabledForStandalone(feature) ? 1 : 0;
            }

            OpenXRFeature[] controllerFeatures = FeatureHelpers.GetFeaturesWithIdsForBuildTarget(buildTargetGroup, QuestLinkControllerFeatureIds);
            foreach (OpenXRFeature feature in controllerFeatures)
            {
                enabledControllerCount += SetFeatureEnabledForStandalone(feature) ? 1 : 0;
            }

            SetFeatureEnabledForStandalone(FeatureHelpers.GetFeatureWithIdForBuildTarget(buildTargetGroup, MetaOpenXrLifeCycleFeatureId));

            if (enabledPassthroughCount < QuestLinkPassthroughFeatureIds.Length)
            {
                Debug.LogWarning($"Enabled {enabledPassthroughCount}/{QuestLinkPassthroughFeatureIds.Length} Meta Quest Link passthrough features. If passthrough is unavailable, confirm com.unity.xr.meta-openxr is installed and Meta Quest Link developer passthrough is enabled.");
            }
            if (enabledControllerCount < QuestLinkControllerFeatureIds.Length)
            {
                Debug.LogWarning($"Enabled {enabledControllerCount}/{QuestLinkControllerFeatureIds.Length} Quest Link controller input profiles. Controller triggers require the Oculus/Meta Quest controller profile for the active headset.");
            }

            EditorUtility.SetDirty(settings);
            return enabledPassthroughCount + enabledControllerCount;
        }

        private static bool SetFeatureEnabledForStandalone(OpenXRFeature feature)
        {
            if (feature == null)
            {
                return false;
            }

            // Meta OpenXR 2.1 calls its Android lifecycle refresh even when only
            // configuring Standalone/Quest Link. Machines without the Android
            // build module can then throw during OpenXRFeature.enabled's setter.
            // Setting the serialized field directly keeps the Standalone setting
            // deterministic without forcing an unrelated Android install.
            SerializedObject serializedFeature = new SerializedObject(feature);
            SerializedProperty enabledProperty = serializedFeature.FindProperty("m_enabled");
            if (enabledProperty == null)
            {
                return false;
            }

            enabledProperty.boolValue = true;
            serializedFeature.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(feature);
            return true;
        }

        private static void EnsureFolder(string assetFolder)
        {
            if (AssetDatabase.IsValidFolder(assetFolder))
            {
                return;
            }

            string parent = Path.GetDirectoryName(assetFolder)?.Replace('\\', '/');
            string folderName = Path.GetFileName(assetFolder);
            if (string.IsNullOrEmpty(parent))
            {
                parent = "Assets";
            }

            AssetDatabase.CreateFolder(parent, folderName);
        }
    }
}

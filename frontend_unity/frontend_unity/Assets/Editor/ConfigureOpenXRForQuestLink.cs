using System.IO;
using UnityEditor;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEngine;
using UnityEngine.XR.Management;

namespace ParallelHeptics.FrontendUnity.Editor
{
    public static class ConfigureOpenXRForQuestLink
    {
        private const string XrFolder = "Assets/XR";
        private const string GeneralSettingsPath = XrFolder + "/XRGeneralSettingsPerBuildTarget.asset";
        private const string OpenXrLoaderTypeName = "UnityEngine.XR.OpenXR.OpenXRLoader";

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
            EditorUtility.SetDirty(perBuildTargetSettings);
            EditorUtility.SetDirty(generalSettings);
            EditorUtility.SetDirty(generalSettings.AssignedSettings);
            AssetDatabase.SaveAssets();

            Debug.Log(assigned
                ? "Configured OpenXR for PC/Standalone Quest Link. Restart Play Mode if it was already running."
                : "OpenXR loader was already configured or could not be reassigned. Check Project Settings > XR Plug-in Management > PC, Mac & Linux Standalone.");
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

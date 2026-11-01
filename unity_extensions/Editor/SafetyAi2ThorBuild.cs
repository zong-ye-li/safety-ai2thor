using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class SafetyAi2ThorBuild {
    public static void Build() {
        if (!Application.isBatchMode || Application.unityVersion != "2020.3.25f1")
            throw new InvalidOperationException("Use upstream Unity 2020.3.25f1 in batch mode");
        string output = Environment.GetEnvironmentVariable("SAFETY_AI2THOR_EXE");
        if (string.IsNullOrWhiteSpace(output) || !Path.IsPathRooted(output)
            || Path.GetExtension(output).ToLowerInvariant() != ".exe")
            throw new InvalidOperationException("SAFETY_AI2THOR_EXE requires an absolute new executable path");
        output = Path.GetFullPath(output);
        if (Directory.Exists(Path.GetDirectoryName(output)))
            throw new InvalidOperationException("Use a new output directory");
        string[] scenes = {"Assets/Scenes/FloorPlan25_physics.unity", "Assets/Scenes/FloorPlan26_physics.unity"};
        foreach (string scene in scenes)
            if (!File.Exists(scene)) throw new FileNotFoundException("Missing upstream scene", scene);
        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone, ScriptingImplementation.Mono2x);
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = scenes, locationPathName = output,
            target = BuildTarget.StandaloneWindows64, options = BuildOptions.None
        });
        if (report.summary.result != BuildResult.Succeeded)
            throw new InvalidOperationException("Player build failed: " + report.summary.result);
        Debug.Log("safety-ai2thor player built: " + output);
    }
}

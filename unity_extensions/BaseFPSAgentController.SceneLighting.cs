using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public abstract partial class BaseFPSAgentController {
        private sealed class SceneLightingSnapshot {
            public Light light;
            public Color color;
            public float intensity;
            public float range;
            public float shadowStrength;
            public bool enabled;
        }

        private readonly Dictionary<int, SceneLightingSnapshot> sceneLightingSnapshots =
            new Dictionary<int, SceneLightingSnapshot>();

        private Light[] CollectSceneLights(bool includeInactive) {
            // Exclude project assets, prefab previews and other loaded scenes.
            var scene = gameObject.scene;
            return Resources.FindObjectsOfTypeAll<Light>()
                .Where(light => light != null && light.gameObject.scene == scene
                    && (includeInactive || light.gameObject.activeInHierarchy))
                .OrderBy(light => light.GetInstanceID())
                .ToArray();
        }

        private static string SceneLightPath(Transform transform) {
            var names = new List<string>();
            while (transform != null) {
                names.Add(transform.name);
                transform = transform.parent;
            }
            names.Reverse();
            return string.Join("/", names);
        }

        private static Dictionary<string, object> DescribeSceneLight(Light light) {
#if UNITY_EDITOR
            string bakeType = light.lightmapBakeType.ToString();
            string bakeTypeSource = "editor_authoring_settings";
#else
            string bakeType = light.bakingOutput.lightmapBakeType.ToString();
            string bakeTypeSource = "runtime_baking_output";
#endif
            return new Dictionary<string, object> {
                ["instanceId"] = light.GetInstanceID(),
                ["path"] = SceneLightPath(light.transform),
                ["type"] = light.type.ToString(),
                ["bakeType"] = bakeType,
                ["bakeTypeSource"] = bakeTypeSource,
                ["color"] = new Dictionary<string, float> {
                    ["r"] = light.color.r, ["g"] = light.color.g, ["b"] = light.color.b
                },
                ["intensity"] = light.intensity,
                ["range"] = light.range,
                ["shadowStrength"] = light.shadowStrength,
                ["enabled"] = light.enabled,
                ["activeInHierarchy"] = light.gameObject.activeInHierarchy
            };
        }

        private Dictionary<string, object> DescribeSceneLighting(Light[] lights) {
            return new Dictionary<string, object> {
                ["scene"] = gameObject.scene.name,
                ["lightCount"] = lights.Length,
                ["lights"] = lights.Select(DescribeSceneLight).ToArray()
            };
        }

        public void GetSceneLighting(bool includeInactive = true) {
            actionFinished(success: true,
                actionReturn: DescribeSceneLighting(CollectSceneLights(includeInactive)));
        }

        private static void ValidateSceneLightNumber(string name, float value, float minimum,
            float maximum = float.PositiveInfinity, bool allowMinimum = true) {
            if (float.IsNaN(value) || float.IsInfinity(value)
                || (allowMinimum ? value < minimum : value <= minimum) || value > maximum) {
                throw new ArgumentOutOfRangeException(name, value, "Invalid scene-light parameter.");
            }
        }

        public void SetSceneLighting(Color? color = null, float? intensity = null,
            float? intensityMultiplier = null, float? range = null,
            float? shadowStrength = null, bool? enabled = null, bool includeInactive = true) {
            // Validate everything before saving state or changing any light.
            if (!color.HasValue && !intensity.HasValue && !intensityMultiplier.HasValue
                && !range.HasValue && !shadowStrength.HasValue && !enabled.HasValue) {
                throw new ArgumentException("Provide at least one scene-light parameter.");
            }
            if (intensity.HasValue && intensityMultiplier.HasValue) {
                throw new ArgumentException("Choose intensity or intensityMultiplier, not both.");
            }
            if (color.HasValue) {
                ValidateSceneLightNumber("color.r", color.Value.r, 0, 1);
                ValidateSceneLightNumber("color.g", color.Value.g, 0, 1);
                ValidateSceneLightNumber("color.b", color.Value.b, 0, 1);
            }
            if (intensity.HasValue) ValidateSceneLightNumber("intensity", intensity.Value, 0);
            if (intensityMultiplier.HasValue)
                ValidateSceneLightNumber("intensityMultiplier", intensityMultiplier.Value, 0);
            if (range.HasValue) ValidateSceneLightNumber("range", range.Value, 0, allowMinimum: false);
            if (shadowStrength.HasValue)
                ValidateSceneLightNumber("shadowStrength", shadowStrength.Value, 0, 1);

            var lights = CollectSceneLights(includeInactive);
            // Discard destroyed objects left over from a previous scene load.
            foreach (var key in sceneLightingSnapshots.Where(item => item.Value.light == null)
                .Select(item => item.Key).ToArray()) {
                sceneLightingSnapshots.Remove(key);
            }
            foreach (var light in lights) {
                int id = light.GetInstanceID();
                if (!sceneLightingSnapshots.ContainsKey(id)) {
                    sceneLightingSnapshots[id] = new SceneLightingSnapshot {
                        light = light, color = light.color, intensity = light.intensity,
                        range = light.range, shadowStrength = light.shadowStrength,
                        enabled = light.enabled
                    };
                }
                if (color.HasValue) {
                    var value = color.Value;
                    light.color = new Color(value.r, value.g, value.b, 1);
                }
                if (intensity.HasValue) light.intensity = intensity.Value;
                if (intensityMultiplier.HasValue)
                    light.intensity = sceneLightingSnapshots[id].intensity * intensityMultiplier.Value;
                if (range.HasValue) light.range = range.Value;
                if (shadowStrength.HasValue) light.shadowStrength = shadowStrength.Value;
                if (enabled.HasValue) light.enabled = enabled.Value;
            }
            var result = DescribeSceneLighting(lights);
            result["appliedCount"] = lights.Length;
            actionFinished(success: true, actionReturn: result);
        }

        public void ResetSceneLighting() {
            int restored = 0;
            foreach (var snapshot in sceneLightingSnapshots.Values) {
                if (snapshot.light == null) continue;
                var light = snapshot.light;
                light.color = snapshot.color;
                light.intensity = snapshot.intensity;
                light.range = snapshot.range;
                light.shadowStrength = snapshot.shadowStrength;
                light.enabled = snapshot.enabled;
                restored++;
            }
            sceneLightingSnapshots.Clear();
            var result = DescribeSceneLighting(CollectSceneLights(true));
            result["restoredCount"] = restored;
            actionFinished(success: true, actionReturn: result);
        }
    }
}

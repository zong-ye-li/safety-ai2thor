using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnityStandardAssets.Characters.FirstPerson {
    // Material changes exist only inside this agent's RGB camera render. Nested
    // depth/ID/other cameras use original materials, and physics is never changed.
    public sealed class SafetyHeldObjectVisualOverride : MonoBehaviour {
        private sealed class MaterialState {
            public Renderer renderer;
            public Material[] originals;
            public Material[] copies;
        }
        private sealed class CameraState {
            public Camera camera;
            public bool wasApplied;
        }
        private BaseFPSAgentController owner;
        private GameObject held;
        private readonly List<MaterialState> materials = new List<MaterialState>();
        private readonly Stack<CameraState> cameras = new Stack<CameraState>();
        private float requestedOpacity = 1;
        private float lastRgbOpacity = 1;
        private bool applied;
        private string lastError = "";

        public void Bind(BaseFPSAgentController controller) {
            owner = controller;
            RefreshHeld();
        }

        private void OnEnable() {
            Camera.onPreCull += BeforeCamera;
            Camera.onPostRender += AfterCamera;
        }

        private void OnDisable() {
            Camera.onPreCull -= BeforeCamera;
            Camera.onPostRender -= AfterCamera;
            RestoreMaterials();
            DisposeCopies();
            cameras.Clear();
        }

        private void LateUpdate() {
            // Also repair an interrupted camera render before the next frame.
            if (cameras.Count > 0) {
                RestoreMaterials();
                cameras.Clear();
            }
            RefreshHeld();
        }

        private void RefreshHeld() {
            GameObject next = owner != null ? owner.ItemInHand : null;
            if (ReferenceEquals(next, held)) return;
            RestoreMaterials();
            DisposeCopies();
            held = next;
            lastRgbOpacity = 1;
            lastError = "";
        }

        private static void ValidateRenderers(Renderer[] renderers) {
            if (GraphicsSettings.currentRenderPipeline != null)
                throw new InvalidOperationException("Held opacity currently supports the built-in rendering pipeline");
            if (renderers.Length == 0)
                throw new InvalidOperationException("Held object has no renderers");
            foreach (var renderer in renderers) {
                foreach (var material in renderer.sharedMaterials) {
                    if (material == null) continue;
                    foreach (string property in new[] { "_Color", "_SrcBlend", "_DstBlend", "_ZWrite", "_Mode" }) {
                        if (!material.HasProperty(property))
                            throw new InvalidOperationException("Unsupported held shader: " + material.shader.name + " lacks " + property);
                    }
                }
            }
        }

        public void Configure(float opacity) {
            if (float.IsNaN(opacity) || float.IsInfinity(opacity) || opacity < 0 || opacity > 1)
                throw new ArgumentOutOfRangeException("opacity", "Opacity must be finite and between zero and one");
            RefreshHeld();
            RestoreMaterials();
            if (held != null && opacity < 1) ValidateRenderers(held.GetComponentsInChildren<Renderer>(true));
            requestedOpacity = opacity;
            lastError = "";
            if (opacity == 1) {
                DisposeCopies();
                lastRgbOpacity = 1;
            }
        }

        private void DisposeCopies() {
            foreach (var state in materials) {
                foreach (var copy in state.copies) {
                    if (copy == null) continue;
                    if (Application.isPlaying) UnityEngine.Object.Destroy(copy);
                    else UnityEngine.Object.DestroyImmediate(copy);
                }
            }
            materials.Clear();
        }

        private void RestoreMaterials() {
            if (!applied) return;
            foreach (var state in materials) {
                if (state.renderer != null) state.renderer.sharedMaterials = state.originals;
            }
            applied = false;
        }

        private void PrepareCopies() {
            var renderers = held.GetComponentsInChildren<Renderer>(true);
            ValidateRenderers(renderers); // Prevalidate every slot before any assignment.
            bool rebuild = materials.Count != renderers.Length;
            if (!rebuild) {
                for (int index = 0; index < renderers.Length; index++) {
                    if (materials[index].renderer != renderers[index]
                        || !materials[index].originals.SequenceEqual(renderers[index].sharedMaterials)) {
                        rebuild = true;
                        break;
                    }
                }
            }
            if (rebuild) {
                DisposeCopies();
                foreach (var renderer in renderers) {
                    var originals = renderer.sharedMaterials;
                    materials.Add(new MaterialState {
                        renderer = renderer, originals = originals,
                        copies = originals.Select(material => material == null ? null : new Material(material) {
                            name = material.name + " [temporary held RGB]", hideFlags = HideFlags.DontSave
                        }).ToArray()
                    });
                }
            }
            foreach (var state in materials) {
                for (int index = 0; index < state.copies.Length; index++) {
                    var copy = state.copies[index];
                    if (copy == null) continue;
                    var original = state.originals[index];
                    copy.shader = original.shader;
                    copy.CopyPropertiesFromMaterial(original);
                    var color = original.GetColor("_Color");
                    copy.SetColor("_Color", new Color(color.r, color.g, color.b, color.a * requestedOpacity));
                    copy.SetFloat("_Mode", 2); // Standard Fade, not just a color alpha.
                    copy.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
                    copy.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
                    copy.SetInt("_ZWrite", 0);
                    copy.SetOverrideTag("RenderType", "Transparent");
                    copy.DisableKeyword("_ALPHATEST_ON");
                    copy.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                    copy.EnableKeyword("_ALPHABLEND_ON");
                    copy.renderQueue = (int)RenderQueue.Transparent;
                }
            }
        }

        private void ApplyMaterials() {
            RefreshHeld();
            if (held == null || requestedOpacity == 1 || applied) return;
            PrepareCopies();
            foreach (var state in materials) state.renderer.sharedMaterials = state.copies;
            applied = true;
            lastRgbOpacity = requestedOpacity;
            lastError = "";
        }

        private void BeforeCamera(Camera camera) {
            if (owner == null || (requestedOpacity == 1 && !applied)) return;
            cameras.Push(new CameraState { camera = camera, wasApplied = applied });
            try {
                if (camera == owner.m_Camera) ApplyMaterials();
                else RestoreMaterials();
            } catch (Exception exception) {
                RestoreMaterials();
                lastError = exception.Message;
            }
        }

        private void AfterCamera(Camera camera) {
            if (cameras.Count == 0) return;
            var previous = cameras.Pop();
            if (previous.camera != camera) {
                RestoreMaterials();
                cameras.Clear();
                lastError = "Camera callback order mismatch; original materials restored";
                return;
            }
            try {
                if (previous.wasApplied) ApplyMaterials();
                else RestoreMaterials();
            } catch (Exception exception) {
                RestoreMaterials();
                lastError = exception.Message;
            }
        }

        private static Dictionary<string, float> Vector(Vector3 value) {
            return new Dictionary<string, float> { ["x"] = value.x, ["y"] = value.y, ["z"] = value.z };
        }

        public Dictionary<string, object> Describe() {
            RefreshHeld();
            bool supported = true;
            string supportError = "";
            if (held != null) {
                try { ValidateRenderers(held.GetComponentsInChildren<Renderer>(true)); }
                catch (Exception exception) { supported = false; supportError = exception.Message; }
            }
            var physicsObject = held == null ? null : held.GetComponent<SimObjPhysics>();
            return new Dictionary<string, object> {
                ["renderMode"] = "own_rgb_camera_only",
                ["requestedOpacity"] = requestedOpacity, ["lastRgbOpacity"] = lastRgbOpacity,
                ["heldObjectId"] = physicsObject == null ? null : physicsObject.objectID,
                ["supported"] = supported, ["supportError"] = supportError, ["lastError"] = lastError,
                ["position"] = held == null ? null : Vector(held.transform.position),
                ["colliders"] = held == null ? new object[0] : held.GetComponentsInChildren<Collider>(true)
                    .Select(collider => (object)new Dictionary<string, object> {
                        ["id"] = collider.GetInstanceID(), ["enabled"] = collider.enabled,
                        ["isTrigger"] = collider.isTrigger
                    }).ToArray(),
                ["renderers"] = held == null ? new object[0] : held.GetComponentsInChildren<Renderer>(true)
                    .Select(renderer => (object)new Dictionary<string, object> {
                        ["id"] = renderer.GetInstanceID(), ["enabled"] = renderer.enabled,
                        ["materialIds"] = renderer.sharedMaterials.Select(material => material == null ? 0 : material.GetInstanceID()).ToArray(),
                        ["shaders"] = renderer.sharedMaterials.Select(material => material == null ? "" : material.shader.name).ToArray()
                    }).ToArray()
            };
        }
    }
}

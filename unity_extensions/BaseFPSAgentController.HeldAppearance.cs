using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public abstract partial class BaseFPSAgentController {
        private SafetyHeldObjectVisualOverride HeldAppearance() {
            var appearance = GetComponent<SafetyHeldObjectVisualOverride>();
            if (appearance == null) appearance = gameObject.AddComponent<SafetyHeldObjectVisualOverride>();
            appearance.Bind(this);
            return appearance;
        }

        public void GetHeldObjectAppearance() {
            actionFinished(success: true, actionReturn: HeldAppearance().Describe());
        }

        public void SetHeldObjectOpacity(float opacity) {
            var appearance = HeldAppearance();
            appearance.Configure(opacity);
            actionFinished(success: true, actionReturn: appearance.Describe());
        }

        public void ResetHeldObjectAppearance() {
            var appearance = HeldAppearance();
            appearance.Configure(1);
            actionFinished(success: true, actionReturn: appearance.Describe());
        }
    }
}

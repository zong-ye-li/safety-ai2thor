using System.Collections.Generic;
using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public abstract partial class BaseFPSAgentController {
        private SafetyEnvironmentTimeClock safetyEnvironmentClock;
        // Read by the guarded common timeScale branch in this custom build.
        private bool SafetyOwnsEnvironmentTime {
            get { return safetyEnvironmentClock != null && safetyEnvironmentClock.Owned; }
        }
        private SafetyEnvironmentTimeClock EnvironmentClock() {
            if (safetyEnvironmentClock == null) {
                safetyEnvironmentClock = gameObject.AddComponent<SafetyEnvironmentTimeClock>();
            }
            return safetyEnvironmentClock;
        }
        public void GetEnvironmentClock() {
            actionFinished(success:true,actionReturn:EnvironmentClock().Describe());
        }
        public void SetEnvironmentClockPaused(bool paused) {
            var clock=EnvironmentClock();clock.Configure(paused);
            actionFinished(success:true,actionReturn:clock.Describe());
        }
    }

}

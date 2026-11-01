using System.Collections.Generic;
using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    // Opt-in single-player scaled-time ownership. Network IO/rendering continue;
    // scaled Update/coroutine/animation timers stop during model waits. Physics
    // autoSimulation is separately owned by the existing clock actions.
    public sealed class SafetyEnvironmentTimeClock : MonoBehaviour {
        private static SafetyEnvironmentTimeClock activeOwner;
        public bool Owned {get;private set;}
        private bool paused;
        public void Configure(bool pause) {
            if (activeOwner != null && activeOwner != this)
                throw new System.InvalidOperationException("Environment time already owned by another agent");
            activeOwner=this;Owned=true;paused=pause;Time.timeScale=pause?0:1;
        }
        public Dictionary<string,object> Describe() {
            return new Dictionary<string,object> {
                ["version"]="unity-scaled-environment-time-v1",["owned"]=Owned,
                ["paused"]=paused,["timeScale"]=Time.timeScale,["scaledTime"]=Time.time,
                ["physicsAutoSimulation"]=Physics.autoSimulation,
                ["scope"]="Scaled Unity timers only;network/render IO continues;not exact simulation-time calibration"
            };
        }
        private void OnDisable() {
            if (activeOwner == this) {Time.timeScale=1;activeOwner=null;}
            Owned=false;
        }
    }
}

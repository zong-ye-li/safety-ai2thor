"""Keep initial survey views independent of the policy's moving image window."""


class InitialSurvey:
    def __init__(self):
        self.frames = []
        self.views = []

    def capture(self, env):
        frames = env.get_recent_frames(1)
        if not frames:
            return
        info = getattr(env, "frame_info", [])
        view = {key: info[-1][key] for key in ("frame", "yaw", "horizon", "observation_id",
                "captured_utc", "monotonic_seconds", "position", "rotation", "physical_time") if info and key in info[-1]}
        # A complete rotation returns to the first direction; don't send that view twice.
        pose = (view.get("yaw"), view.get("horizon"))
        if pose[0] is not None and any((old.get("yaw"), old.get("horizon")) == pose for old in self.views):
            return
        if len(self.frames) >= 4:
            return
        self.frames.append(frames[-1].copy())
        self.views.append({"image_index": len(self.frames)-1, **view,
                           "objects": env.get_discovery_summary() if hasattr(env, "get_discovery_summary") else {"objects": []}})

    def description(self):
        return {"source": "initial_survey", "views": self.views,
                "coverage": "four_yaw_samples" if len({row.get("yaw") for row in self.views if row.get("yaw") is not None}) == 4
                            else "partial_or_unknown",
                "scope": "Views were captured at different times; four yaw samples do not prove complete room coverage."}

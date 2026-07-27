import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class ButtonPlanOracle(PlanOracle):
    def __init__(self, gripper_always_closed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gripper_always_closed = gripper_always_closed

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["press_start"] = self.above(plan_input["button"], 0.06)
        poses["press"] = self.above(plan_input["button"], -0.025)
        poses["press_end"] = poses["press_start"]
        poses["final"] = plan_input["effector_goal"]

        # Times.
        times = {}
        distance = np.linalg.norm(
            poses["initial"].translation() - poses["press_start"].translation()
        )
        times["initial"] = 0.0
        times["press_start"] = times["initial"] + self._dt * (0.5 + distance * 4)
        times["press"] = times["press_start"] + self._dt * 0.8
        times["press_end"] = times["press"] + self._dt * 0.8
        times["final"] = times["press_end"] + self._dt * 1.25
        self.jitter_times(times)

        # Grasps.
        grasps = {}
        if self._gripper_always_closed:
            g = 1.0
        else:
            g = 0.0
        for name in times.keys():
            if not self._gripper_always_closed:
                if name in {"press_start", "final"}:
                    g = 1.0 - g
            grasps[name] = g

        return times, poses, grasps

    def reset(self, ob, info):
        if "privileged_target_button_top_pos" in info:
            target_button_top_pos = info["privileged_target_button_top_pos"]
        else:
            # In task mode, read from the target button's site
            target_button_idx = 0
            target_button_top_pos = self._env.unwrapped._data.site_xpos[
                self._env.unwrapped._button_site_ids[target_button_idx]
            ].copy()

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*self._env.unwrapped._arm_sampling_bounds),
                yaw=0.0,
            ),
            "button": self.to_pose(
                pos=target_button_top_pos,
                yaw=target_button_top_pos[0],
            ),
        }

        self.finalize_plan(plan_input, info)

import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class ButtonPlanOracle(PlanOracle):
    def __init__(self, object_id=0, gripper_always_closed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id
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
        env = self._env.unwrapped
        i = self._object_id
        if f"heca_target_button_{i}_pos_ee" in info:
            target_button_top_pos = info[f"heca_target_button_{i}_pos_ee"]
        else:
            btn = env.get_object(f"button_{i}")
            target_button_top_pos = env._data.site_xpos[btn._site_ids[0]].copy()

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "button": self.to_pose(
                pos=target_button_top_pos,
                yaw=0.0,
            ),
        }

        self.finalize_plan(plan_input, info)

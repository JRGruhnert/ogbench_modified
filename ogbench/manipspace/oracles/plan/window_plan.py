import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class WindowPlanOracle(PlanOracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        window_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["window_initial"]),
            translation=plan_input["window_initial"].translation(),
            n=2,
        )
        window_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["window_initial"]),
            translation=plan_input["window_goal"].translation(),
            n=2,
        )
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(window_initial, 0.06)
        poses["grasp_start"] = window_initial
        poses["grasp_end"] = window_initial
        poses["move"] = window_goal
        poses["release"] = window_goal
        poses["clearance"] = self.above(window_goal, 0.06)
        poses["final"] = plan_input["effector_goal"]

        # Times.
        times = {}
        times["initial"] = 0.0
        times["approach"] = times["initial"] + self._dt
        times["grasp_start"] = times["approach"] + self._dt * 0.5
        times["grasp_end"] = times["grasp_start"] + self._dt * 0.5
        times["move"] = times["grasp_end"] + self._dt * 0.5
        times["release"] = times["move"] + self._dt * 0.5
        times["clearance"] = times["release"] + self._dt * 0.5
        times["final"] = times["clearance"] + self._dt
        self.jitter_times(times)

        # Grasps.
        grasps = self.build_grasps(times, {"grasp_end", "release"})
        times, poses, grasps = self.hold_after(times, poses, grasps, "release", duration=0.4)

        return times, poses, grasps

    def reset(self, ob, info):
        if "privileged_target_window_handle_pos" in info:
            target_handle_pos = info["privileged_target_window_handle_pos"]
        else:
            target_handle_pos = self._env.unwrapped._data.site_xpos[
                self._env.unwrapped._window_target_site_id
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
            "window_initial": self.to_pose(
                pos=info["privileged_window_handle_pos"],
                yaw=info["privileged_window_handle_yaw"][0],
            ),
            "window_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info["privileged_window_handle_yaw"][0],
            ),
        }

        self.finalize_plan(plan_input, info)

import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class LeverPlanOracle(PlanOracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        lever_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["lever_initial"]),
            translation=plan_input["lever_initial"].translation(),
            n=2,
        )
        lever_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["lever_initial"]),
            translation=plan_input["lever_goal"].translation(),
            n=2,
        )
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(lever_initial, 0.08)
        poses["grasp_start"] = lever_initial
        poses["grasp_end"] = lever_initial
        poses["move"] = lever_goal
        poses["release"] = lever_goal
        poses["clearance"] = self.above(lever_goal, 0.08)
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
        for time in times.keys():
            if time != "initial":
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.1

        # Grasps.
        grasps = {}
        g = 0.0
        for name in times.keys():
            if name in {"grasp_end", "release"}:
                g = 1.0 - g
            grasps[name] = g

        return times, poses, grasps

    def reset(self, ob, info):
        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*self._env.unwrapped._arm_sampling_bounds),
                yaw=np.random.uniform(-np.pi, np.pi),
            ),
            "lever_initial": self.to_pose(
                pos=info["privileged_lever_handle_pos"],
                yaw=info["privileged_lever_handle_yaw"][0],
            ),
            "lever_goal": self.to_pose(
                pos=info["privileged_target_lever_handle_pos"],
                yaw=info["privileged_lever_handle_yaw"][0],
            ),
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        poses = [poses[name] for name in times.keys()]
        grasps = [grasps[name] for name in times.keys()]
        times = list(times.values())

        self._t_init = info["time"][0]
        self._t_max = times[-1]
        self._done = False
        self._plan = self.compute_plan(times, poses, grasps)

import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class WindowPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
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
        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(window_initial, 0.08)
        poses["down"] = window_initial
        poses["grasp"] = window_initial
        poses["move"] = window_goal
        poses["release"] = window_goal
        poses["leave"] = self.above(window_goal, 0.08)
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = self.distance(poses["initial"], poses["approach"])
        distance2 = self.distance(poses["grasp"], poses["move"])
        distance3 = self.distance(poses["leave"], poses["final"])
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.8 + distance * 4)
        times["down"] = self._dt * 0.5
        times["grasp"] = self._dt * 0.5
        times["move"] = self._dt * (0.8 + distance2 * 4)
        times["release"] = self._dt * 0.5
        times["leave"] = self._dt * 0.5
        times["final"] = self._dt * (0.8 + distance3 * 4)

        # Grasps
        grasps = self.build_grasps(times, {"grasp", "release"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["approach", "grasp", "release", "leave"],
        )

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        target_handle_pos = info[f"heca_target_window{i}_pos"]

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "window_initial": self.to_pose(
                pos=info[f"heca_window{i}_pos"],
                yaw=info[f"heca_window{i}_yaw"][0],
            ),
            "window_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info[f"heca_window{i}_yaw"][0],
            ),
        }

        self.finalize_plan(plan_input, info)

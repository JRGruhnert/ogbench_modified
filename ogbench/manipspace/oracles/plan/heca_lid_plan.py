import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class LidPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        grab_yaw = self.get_yaw(plan_input["lid_initial"]) + np.pi / 2
        place_yaw = self.get_yaw(plan_input["lid_goal"]) + np.pi / 2
        lid_initial = self.to_pose(
            pos=plan_input["lid_initial"].translation(),
            yaw=grab_yaw,
        )
        lid_goal = self.to_pose(
            pos=plan_input["lid_goal"].translation(),
            yaw=place_yaw,
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(lid_initial, 0.12)
        poses["pick-start"] = lid_initial
        poses["pick"] = lid_initial
        poses["pick-end"] = poses["approach"]
        poses["approach2"] = self.above(lid_goal, 0.12)
        poses["place-start"] = lid_goal
        poses["place"] = lid_goal
        poses["place-end"] = poses["approach2"]
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance1 = np.linalg.norm(
            poses["initial"].translation() - poses["approach"].translation()
        )
        distance2 = np.linalg.norm(
            poses["approach"].translation() - poses["approach2"].translation()
        )
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.5 + distance1 * 4)
        times["pick-start"] = self._dt * 1.5
        times["pick"] = self._dt
        times["pick-end"] = self._dt
        times["approach2"] = self._dt * (0.5 + distance1 * 4)
        times["place-start"] = self._dt * 1.5
        times["place"] = self._dt
        times["place-end"] = self._dt
        times["final"] = self._dt

        # Grasp
        grasps = self.build_grasps(times, {"pick", "place"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["pick", "place"],
        )
        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        target_pos = info[f"heca_target_lid_{i}_pos"]
        target_yaw = info[f"heca_target_lid_{i}_yaw"][0]

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "lid_initial": self.to_pose(
                pos=info[f"heca_lid_{i}_pos"],
                yaw=info[f"heca_lid_{i}_yaw"][0],
            ),
            "lid_goal": self.to_pose(
                pos=target_pos,
                yaw=target_yaw,
            ),
        }

        self.finalize_plan(plan_input, info)

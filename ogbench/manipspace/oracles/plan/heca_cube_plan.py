import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class CubePlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        # Pick
        block_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["block_initial"]),
            translation=plan_input["block_initial"].translation(),
        )
        # Place
        block_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(block_initial),
            obj_yaw=self.get_yaw(plan_input["block_goal"]),
            translation=plan_input["block_goal"].translation(),
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(block_initial, 0.1)
        poses["pick-start"] = block_initial
        poses["pick"] = block_initial
        poses["pick-end"] = block_initial
        poses["leave"] = poses["approach"]
        poses["approach2"] = self.above(block_goal, 0.1)
        poses["place-start"] = block_goal
        poses["place"] = block_goal
        poses["place-end"] = block_goal
        poses["leave2"] = poses["approach2"]
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = self.distance(poses["initial"], poses["approach"])
        distance2 = self.distance(poses["leave"], poses["approach2"])
        distance3 = self.distance(poses["approach2"], poses["final"])
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.8 + distance * 4)
        times["pick-start"] = self._dt
        times["pick"] = self._dt
        times["pick-end"] = self._dt
        times["leave"] = self._dt
        times["approach2"] = self._dt * (0.8 + distance2 * 4)
        times["place-start"] = self._dt
        times["place"] = self._dt
        times["place-end"] = self._dt
        times["leave2"] = self._dt
        times["final"] = self._dt * (0.8 + distance3 * 4)

        # Grasp
        grasps = self.build_grasps(times, {"pick", "place"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["approach", "pick", "leave", "approach2", "place", "leave"],
        )
        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        target_block_pos = info[f"heca_target_cube{i}_pos"]
        target_block_yaw = info[f"heca_target_cube{i}_yaw"]

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "block_initial": self.to_pose(
                pos=info[f"heca_cube{i}_pos"],
                yaw=info[f"heca_cube{i}_yaw"][0],
            ),
            "block_goal": self.to_pose(
                pos=target_block_pos,
                yaw=target_block_yaw[0],
            ),
        }

        self.finalize_plan(plan_input, info)

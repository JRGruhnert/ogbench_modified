import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class CubePlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        # Pick
        block_initial = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["block_initial"]),
            translation=plan_input["block_initial"].translation(),
        )
        # Place
        block_goal = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["block_goal"]),
            translation=plan_input["block_goal"].translation(),
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(block_initial, 0.1)
        poses["grasp"] = block_initial
        poses["leave"] = poses["approach"]
        poses["approach2"] = self.above(block_goal, 0.1)
        poses["release"] = block_goal
        poses["leave2"] = poses["approach2"]
        poses["final"] = plan_input["effector_goal"]

        # Times
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt
        times["grasp"] = self._dt * 0.8
        times["leave"] = self._dt * 0.8
        times["approach2"] = self._dt
        times["release"] = self._dt * 0.8
        times["leave2"] = self._dt * 0.8
        times["final"] = self._dt

        # Grasp
        grasps = self.build_grasps(times, {"grasp", "release"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["grasp", "release"],
        )
        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        if f"heca_target_cube_{i}_pos" in info:
            target_block_pos = info[f"heca_target_cube_{i}_pos"]
            target_block_yaw = info[f"heca_target_cube_{i}_yaw"]
        else:
            cube = env.get_object(f"cube_{i}")
            target_block_pos = env._data.mocap_pos[cube._target_mocap_id].copy()
            target_block_yaw = np.array([0.0])

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
                pos=info[f"heca_cube_{i}_pos_base"],
                yaw=info[f"heca_cube_{i}_yaw"][0],
            ),
            "block_goal": self.to_pose(
                pos=target_block_pos,
                yaw=target_block_yaw[0],
            ),
        }

        self.finalize_plan(plan_input, info)

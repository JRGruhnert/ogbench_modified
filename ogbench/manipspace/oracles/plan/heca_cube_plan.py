import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class CubePlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}

        # Pick.
        block_initial = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["block_initial"]),
            translation=plan_input["block_initial"].translation(),
        )
        poses["initial"] = plan_input["effector_initial"]
        poses["pick"] = self.above(block_initial, 0.1 + np.random.uniform(0, 0.1))
        poses["pick_start"] = block_initial
        poses["pick_end"] = block_initial
        poses["postpick"] = poses["pick"]

        # Place.
        block_goal = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["block_goal"]),
            translation=plan_input["block_goal"].translation(),
        )
        poses["place"] = self.above(block_goal, 0.1 + np.random.uniform(0, 0.1))
        poses["place_start"] = block_goal
        poses["place_end"] = block_goal
        poses["postplace"] = poses["place"]
        poses["final"] = plan_input["effector_goal"]

        # Clearance.
        midway = lie.interpolate(poses["postpick"], poses["place"])
        poses["clearance"] = lie.SE3.from_rotation_and_translation(
            rotation=midway.rotation(),
            translation=np.array(
                [*midway.translation()[:2], poses["initial"].translation()[-1]]
            )
            + np.random.uniform([-0.1, -0.1, 0], [0.1, 0.1, 0.2]),
        )

        # Times.
        times = {}
        times["initial"] = 0.0
        times["pick"] = times["initial"] + self._dt
        times["pick_start"] = times["pick"] + self._dt * 1.5
        times["pick_end"] = times["pick_start"] + self._dt
        times["postpick"] = times["pick_end"] + self._dt
        times["clearance"] = times["postpick"] + self._dt
        times["place"] = times["clearance"] + self._dt
        times["place_start"] = times["place"] + self._dt * 1.5
        times["place_end"] = times["place_start"] + self._dt
        times["postplace"] = times["place_end"] + self._dt
        times["final"] = times["postplace"] + self._dt
        times = self.jitter_times(times, factor=0.2)
        times, poses = self.add_neutral_yaw_prephase(poses["initial"], times, poses)

        # Grasps.
        grasps = self.build_grasps(times, {"pick_end", "place_end"})
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

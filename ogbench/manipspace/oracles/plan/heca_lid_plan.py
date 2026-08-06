import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class LidPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        grab_yaw = self.get_yaw(plan_input["lid_initial"]) + np.pi / 2
        lid_initial = self.to_pose(
            pos=plan_input["lid_initial"].translation(),
            yaw=grab_yaw,
        )
        lid_goal = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["lid_goal"]),
            translation=plan_input["lid_goal"].translation(),
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["pick"] = self.above(lid_initial, 0.12)
        poses["pick_start"] = lid_initial
        poses["pick_end"] = lid_initial
        poses["postpick"] = poses["pick"]
        poses["place"] = self.above(lid_goal, 0.12)
        poses["place_start"] = lid_goal
        poses["place_end"] = lid_goal
        poses["postplace"] = poses["place"]
        poses["final"] = plan_input["effector_goal"]

        # Times
        times = {}
        times["initial"] = 0.0
        times["pick"] = times["initial"] + self._dt
        times["pick_start"] = times["pick"] + self._dt * 2.0
        times["pick_end"] = times["pick_start"] + self._dt
        times["postpick"] = times["pick_end"] + self._dt
        times["place"] = times["postpick"] + self._dt
        times["place_start"] = times["place"] + self._dt * 1.5
        times["place_end"] = times["place_start"] + self._dt
        times["postplace"] = times["place_end"] + self._dt
        times["final"] = times["postplace"] + self._dt
        times = self.jitter_times(times, factor=0.2)

        # Grasps
        grasps = self.build_grasps(times, {"pick_end", "place_end"})

        # Postprocess
        times, poses, grasps = self.hold_after_multiple(
            times,
            poses,
            grasps,
            names=["grasp_end", "release_end"],
        )
        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        if f"heca_target_lid_{i}_pos" in info:
            target_pos = info[f"heca_target_lid_{i}_pos"]
            target_yaw = info[f"heca_target_lid_{i}_yaw"][0]
        else:
            lid = env.get_object(f"lid_{i}")
            target_pos = env._data.mocap_pos[lid._target_mocap_id].copy()
            target_yaw = 0.0

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
                pos=info[f"heca_lid_{i}_pos_ee"],
                yaw=info[f"heca_lid_{i}_yaw"][0],
            ),
            "lid_goal": self.to_pose(
                pos=target_pos,
                yaw=target_yaw,
            ),
        }

        self.finalize_plan(plan_input, info)

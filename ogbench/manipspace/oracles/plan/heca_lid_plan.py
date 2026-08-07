import numpy as np

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
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt
        times["pick-start"] = self._dt * 1.5
        times["pick"] = self._dt
        times["pick-end"] = self._dt
        times["approach2"] = self._dt
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

import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class LeverPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        # Poses.
        poses = {}
        lever_initial = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["lever_initial"]),
            translation=plan_input["lever_initial"].translation(),
        )
        lever_goal = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["lever_initial"]),
            translation=plan_input["lever_goal"].translation(),
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
        times = self.jitter_times(times)
        times, poses = self.add_neutral_yaw_prephase(poses["initial"], times, poses)

        # Grasps.
        grasps = self.build_grasps(times, {"grasp_end", "release"})

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        if f"heca_target_lever_{i}_pos_ee" in info:
            target_handle_pos = info[f"heca_target_lever_{i}_pos_ee"]
        else:
            target_handle_pos = env._data.site_xpos[
                env.get_object(f"lever_{i}")._target_site_id
            ].copy()

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "lever_initial": self.to_pose(
                pos=info[f"heca_lever_{i}_pos_ee"],
                yaw=info[f"heca_lever_{i}_yaw"][0],
            ),
            "lever_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info[f"heca_lever_{i}_yaw"][0],
            ),
        }

        self.finalize_plan(plan_input, info)

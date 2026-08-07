import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class DrawerPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):

        drawer_initial = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["drawer_initial"]),
            translation=plan_input["drawer_initial"].translation(),
        )
        drawer_goal = self.equal_yaw(
            obj_yaw=self.get_yaw(plan_input["drawer_initial"]),
            translation=plan_input["drawer_goal"].translation(),
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(drawer_initial, 0.12)
        poses["grasp-start"] = drawer_initial
        poses["grasp-end"] = drawer_initial
        poses["move"] = drawer_goal
        poses["release-start"] = drawer_goal
        poses["release-end"] = self.above(drawer_goal, 0.12)
        poses["final"] = plan_input["effector_goal"]

        # Times
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt
        times["grasp-start"] = self._dt * 0.5
        times["grasp-end"] = self._dt * 0.5
        times["move"] = self._dt * 0.5
        times["release-start"] = self._dt * 0.5
        times["release-end"] = self._dt * 0.5
        times["final"] = self._dt

        # Grasps
        grasps = self.build_grasps(times, {"grasp-end", "release-start"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["grasp-end", "release-start"],
        )

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        if f"heca_target_drawer_{i}_pos_ee" in info:
            target_handle_pos = info[f"heca_target_drawer_{i}_pos_ee"]
        else:
            target_handle_pos = env._data.site_xpos[
                env.get_object(f"drawer_{i}")._target_site_id
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
            "drawer_initial": self.to_pose(
                pos=info[f"heca_drawer_{i}_pos_ee"],
                yaw=info[f"heca_drawer_{i}_yaw"][0],
            ),
            "drawer_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info[f"heca_drawer_{i}_yaw"][0],
            ),
        }

        self.finalize_plan(plan_input, info)

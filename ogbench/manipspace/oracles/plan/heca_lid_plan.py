import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class LidPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        lid_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["lid_initial"]) + np.pi / 2,
            translation=plan_input["lid_initial"].translation(),
            n=2,
        )
        lid_goal_base = self.shortest_yaw(
            eff_yaw=self.get_yaw(lid_initial),
            obj_yaw=self.get_yaw(plan_input["lid_goal"]) + np.pi / 2,
            translation=plan_input["lid_goal"].translation(),
            n=2,
        )
        offset = np.asarray(plan_input["handle_offset"], dtype=float)
        lid_goal = self.to_pose(
            pos=lid_goal_base.translation() + offset,
            yaw=lid_goal_base.rotation().compute_yaw_radians(),
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(lid_initial, 0.12)
        poses["pick-start"] = lid_initial
        poses["pick"] = lid_initial
        poses["pick-end"] = lid_initial
        poses["leave"] = poses["approach"]
        poses["approach2"] = self.above(lid_goal, 0.12)
        poses["place-start"] = lid_goal
        poses["place"] = lid_goal
        poses["place-end"] = lid_goal
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
        target_pos = info[f"heca_target_lid{i}_pos"]
        target_yaw = info[f"heca_target_lid{i}_yaw"][0]

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
                pos=info[f"heca_lid{i}_pos"],
                yaw=info[f"heca_lid{i}_yaw"][0],
            ),
            "lid_goal": self.to_pose(
                pos=target_pos,
                yaw=target_yaw,
            ),
            # Local offset from the lid's base to its handle site; the plan
            # grasps the handle, so goal positions must be shifted by it.
            "handle_offset": np.asarray(
                env.get_object(f"lid{i}").handle_offset, dtype=float
            ),
        }

        self.finalize_plan(plan_input, info)

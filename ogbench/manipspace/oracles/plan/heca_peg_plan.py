import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class PegPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        peg_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["peg_initial"]) + np.pi / 2,
            translation=plan_input["peg_initial"].translation(),
            n=2,
        )
        grab_yaw = self.get_yaw(peg_initial)
        handle_pos = plan_input["peg_initial"].translation()
        ring_center = plan_input["ring_center"]
        offset = handle_pos - ring_center
        place_pos = plan_input["peg_goal"].translation() + offset
        peg_goal = self.to_pose(pos=place_pos, yaw=grab_yaw)

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(peg_initial, 0.08)
        poses["pick-start"] = peg_initial
        poses["pick"] = peg_initial
        poses["pick-end"] = peg_initial
        poses["leave"] = poses["approach"]
        poses["approach2"] = self.above(peg_goal, 0.08)
        poses["place-start"] = peg_goal
        poses["place"] = peg_goal
        poses["place-end"] = peg_goal
        poses["leave2"] = poses["approach2"]
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = self.distance(poses["initial"], poses["approach"])
        distance2 = self.distance(poses["leave"], poses["approach2"])
        distance3 = self.distance(poses["approach2"], poses["final"])
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.8 + distance * 4)
        times["pick-start"] = self._dt * 1.5
        times["pick"] = self._dt
        times["pick-end"] = self._dt
        times["leave"] = self._dt * 1.5
        times["approach2"] = self._dt * (0.8 + distance2 * 4)
        times["place-start"] = self._dt * 1.5
        times["place"] = self._dt
        times["place-end"] = self._dt
        times["leave2"] = self._dt * 1.5
        times["final"] = self._dt * (0.8 + distance3 * 4)

        # Grasp
        grasps = self.build_grasps(times, {"pick", "place"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["approach", "pick", "leave", "approach2", "place", "leave2"],
        )
        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        target_pos = info[f"heca_target_peg{i}_pos"]
        target_yaw = info[f"heca_target_peg{i}_yaw"][0]

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "peg_initial": self.to_pose(
                pos=info[f"heca_peg{i}_pos"],
                yaw=info[f"heca_peg{i}_yaw"][0],
            ),
            "peg_goal": self.to_pose(
                pos=target_pos,
                yaw=target_yaw,
            ),
            "ring_center": info[f"heca_peg{i}_pos_base"],
        }

        self.finalize_plan(plan_input, info)

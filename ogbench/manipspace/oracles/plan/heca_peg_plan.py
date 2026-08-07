import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class PegPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        grab_yaw = self.get_yaw(plan_input["peg_initial"]) + np.pi / 2
        peg_initial = self.to_pose(
            pos=plan_input["peg_initial"].translation(),
            yaw=grab_yaw,
        )
        handle_pos = plan_input["peg_initial"].translation()
        ring_center = plan_input["ring_center"]
        offset = handle_pos - ring_center
        place_pos = plan_input["peg_goal"].translation() + offset
        peg_goal = self.to_pose(pos=place_pos, yaw=grab_yaw)

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(peg_initial, 0.1)
        poses["pick-start"] = peg_initial
        poses["pick"] = peg_initial
        poses["pick-end"] = poses["approach"]
        poses["approach2"] = self.above(peg_goal, 0.1)
        poses["place-start"] = peg_goal
        poses["place"] = peg_goal
        poses["place-end"] = poses["approach2"]
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = np.linalg.norm(
            poses["initial"].translation() - poses["approach"].translation()
        )
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.5 + distance * 4)
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
        if f"heca_target_peg_{i}_pos" in info:
            target_pos = info[f"heca_target_peg_{i}_pos"]
            target_yaw = info[f"heca_target_peg_{i}_yaw"][0]
        else:
            peg = env.get_object(f"peg_{i}")
            target_pos = env._data.mocap_pos[peg._target_mocap_id].copy()
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
            "peg_initial": self.to_pose(
                pos=info[f"heca_peg_{i}_pos_ee"],
                yaw=info[f"heca_peg_{i}_yaw"][0],
            ),
            "peg_goal": self.to_pose(
                pos=target_pos,
                yaw=target_yaw,
            ),
            "ring_center": info[f"heca_peg_{i}_pos_base"],
        }

        self.finalize_plan(plan_input, info)

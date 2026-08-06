import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class PegPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        poses = {}

        # Pick — go above the peg handle, descend, grab.
        grab_yaw = self.get_yaw(plan_input["peg_initial"]) + np.pi / 2
        peg_initial = self.to_pose(
            pos=plan_input["peg_initial"].translation(),
            yaw=grab_yaw,
        )
        poses["initial"] = plan_input["effector_initial"]
        poses["pick"] = self.above(peg_initial, 0.12 + np.random.uniform(0, 0.08))
        poses["pick_start"] = peg_initial
        poses["pick_end"] = peg_initial
        poses["postpick"] = poses["pick"]

        # Place — go above target + offset so the ring lands on target.
        handle_pos = plan_input["peg_initial"].translation()
        ring_center = plan_input["ring_center"]
        offset = handle_pos - ring_center
        place_pos = plan_input["peg_goal"].translation() + offset
        peg_goal = self.to_pose(pos=place_pos, yaw=grab_yaw)
        poses["place"] = self.above(peg_goal, 0.12 + np.random.uniform(0, 0.08))
        poses["place_start"] = peg_goal
        poses["place_end"] = peg_goal
        poses["postplace"] = poses["place"]
        poses["final"] = plan_input["effector_goal"]

        # Times.
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
        times, poses = self.add_neutral_yaw_prephase(poses["initial"], times, poses)

        grasps = self.build_grasps(times, {"pick_end", "place_end"})

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

import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class PegPlanOracle(PlanOracle):
    """Plan oracle for picking up the assembly peg and placing it at the target."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.add_neutral_yaw_prephase(poses["initial"], times, poses)

        self.add_dwell("postpick_dwell", "postpick", times, poses, 0.4)
        self.add_dwell("place_end_dwell", "place_end", times, poses, 0.4)

        for time in times.keys():
            if time != "initial" and not time.endswith("_dwell"):
                times[time] += np.random.uniform(-1, 1) * self._dt * 0.2

        grasps = {}
        g = 0.0
        for name in times.keys():
            if name in {"pick_end", "place_end"}:
                g = 1.0 - g
            grasps[name] = g

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        if "privileged_target_peg_pos" in info:
            target_pos = info["privileged_target_peg_pos"]
            target_yaw = info["privileged_target_peg_yaw"][0]
        else:
            peg = env.get_object("peg")
            target_pos = env._data.mocap_pos[peg._target_mocap_ids[0]].copy()
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
                pos=info["privileged_peg_0_handle_pos"],
                yaw=info["privileged_peg_0_yaw"][0],
            ),
            "peg_goal": self.to_pose(
                pos=target_pos,
                yaw=target_yaw,
            ),
            "ring_center": info["privileged_peg_0_pos"],
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        poses = [poses[name] for name in times.keys()]
        grasps = [grasps[name] for name in times.keys()]
        times = list(times.values())

        self._t_init = info["time"][0]
        self._t_max = times[-1]
        self._done = False
        self._plan = self.compute_plan(times, poses, grasps)

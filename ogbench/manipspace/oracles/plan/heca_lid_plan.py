import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class LidPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        poses = {}

        # Pick — go above the lid handle, descend, grab.
        # Rotate gripper 90° relative to handle so fingers close across the handle.
        grab_yaw = self.get_yaw(plan_input["lid_initial"]) + np.pi / 2
        lid_initial = self.to_pose(
            pos=plan_input["lid_initial"].translation(),
            yaw=grab_yaw,
        )
        poses["initial"] = plan_input["effector_initial"]
        poses["pick"] = self.above(lid_initial, 0.12 + np.random.uniform(0, 0.08))
        poses["pick_start"] = lid_initial
        poses["pick_end"] = lid_initial
        poses["postpick"] = poses["pick"]

        # Place — go above target, descend, release.
        lid_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(poses["postpick"]),
            obj_yaw=self.get_yaw(plan_input["lid_goal"]),
            translation=plan_input["lid_goal"].translation(),
            n=2,
        )
        poses["place"] = self.above(lid_goal, 0.12 + np.random.uniform(0, 0.08))
        poses["place_start"] = lid_goal
        poses["place_end"] = lid_goal
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

        # Grasps.
        g = 0.0
        grasps = {}
        for name in times.keys():
            if name in {"pick_end", "place_end"}:
                g = 1.0 - g
            grasps[name] = g

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

        times, poses, grasps = self.compute_keyframes(plan_input)
        poses = [poses[name] for name in times.keys()]
        grasps = [grasps[name] for name in times.keys()]
        times = list(times.values())

        self._t_init = info["time"][0]
        self._t_max = times[-1]
        self._done = False
        self._plan = self.compute_plan(times, poses, grasps)

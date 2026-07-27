import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class FaucetPlanOracle(PlanOracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_keyframes(self, plan_input):
        faucet_initial = plan_input["faucet_initial"]
        faucet_goal = plan_input["faucet_goal"]
        init_knob_angle = plan_input["init_knob_angle"]
        goal_knob_angle = plan_input["goal_knob_angle"]
        center = plan_input["faucet_center"]
        radius = 0.105  # 0.175 * 0.6 scale

        # Push: close gripper, approach from outside the arc, push handle along.
        n_arc = 6
        arc_angles = np.linspace(init_knob_angle, goal_knob_angle, n_arc)
        arc_poses = []
        for angle in arc_angles:
            xy = center[:2] + radius * np.array([np.sin(angle), -np.cos(angle)])
            pos = np.array([xy[0], xy[1], faucet_initial.translation()[2]])
            arc_poses.append(self.to_pose(pos=pos, yaw=self.get_yaw(faucet_initial)))

        push_offset = 0.08
        # Approach from the side opposite to the push direction.
        # Tangential CCW: [cos(angle), sin(angle)], Tangential CW: [-cos(angle), -sin(angle)]
        delta = goal_knob_angle - init_knob_angle
        if delta >= 0:
            # Pushing CCW, approach from CW side.
            init_dir = np.array([-np.cos(init_knob_angle), -np.sin(init_knob_angle)])
        else:
            # Pushing CW, approach from CCW side.
            init_dir = np.array([np.cos(init_knob_angle), np.sin(init_knob_angle)])
        approach_xy = faucet_initial.translation()[:2] + init_dir * push_offset
        handle_z = faucet_initial.translation()[2]
        approach_pos = np.array([approach_xy[0], approach_xy[1], handle_z])
        above_init_pos = np.array([approach_xy[0], approach_xy[1], handle_z + 0.10])

        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["above"] = self.to_pose(pos=above_init_pos, yaw=self.get_yaw(faucet_initial))
        poses["approach"] = self.to_pose(pos=approach_pos, yaw=self.get_yaw(faucet_initial))
        for i, p in enumerate(arc_poses):
            poses[f"arc_{i}"] = p
        # Dwell at the last arc point to ensure the turn finishes.
        poses["dwell"] = self.to_pose(
            pos=arc_poses[-1].translation(),
            yaw=self.get_yaw(arc_poses[-1]),
        )
        poses["lift"] = self.to_pose(
            pos=arc_poses[-1].translation() + np.array([0, 0, 0.10]),
            yaw=self.get_yaw(arc_poses[-1]),
        )
        poses["final"] = plan_input["effector_goal"]

        times = {}
        times["initial"] = 0.0
        times["above"] = times["initial"] + self._dt
        times["approach"] = times["above"] + self._dt * 0.5
        t = times["approach"]
        for i in range(n_arc):
            times[f"arc_{i}"] = t + self._dt * 0.4  # longer per step for smoother push
            t = times[f"arc_{i}"]
        times["dwell"] = t + self._dt * 0.5
        times["lift"] = times["dwell"] + self._dt * 0.5
        times["final"] = times["lift"] + self._dt
        self.add_neutral_yaw_prephase(poses["initial"], times, poses)
        for name in times.keys():
            if name != "initial":
                times[name] += np.random.uniform(-1, 1) * self._dt * 0.1

        grasps = {}
        for name in times.keys():
            grasps[name] = 1.0  # gripper always closed

        return times, poses, grasps

    def reset(self, ob, info):
        if "privileged_target_faucet_handle_pos" in info:
            target_handle_pos = info["privileged_target_faucet_handle_pos"]
            target_faucet_yaw = info["privileged_target_faucet_pos"][0]
        else:
            target_handle_pos = self._env._data.site_xpos[
                self._env.unwrapped._faucet_target_site_id
            ].copy()
            target_faucet_yaw = self._env.unwrapped._target_faucet_yaw

        faucet_center = self._env._data.xpos[
            self._env._data.body("faucet_link").id
        ].copy()

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*self._env.unwrapped._arm_sampling_bounds),
                yaw=0.0,
            ),
            "faucet_initial": self.to_pose(
                pos=info["privileged_faucet_handle_pos"],
                yaw=info["privileged_faucet_handle_yaw"][0],
            ),
            "faucet_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info["privileged_faucet_handle_yaw"][0],
            ),
            "faucet_center": faucet_center,
            "init_knob_angle": self._env._data.joint("faucet_knob").qpos[0],
            "goal_knob_angle": target_faucet_yaw,
        }

        times, poses, grasps = self.compute_keyframes(plan_input)
        poses = [poses[name] for name in times.keys()]
        grasps = [grasps[name] for name in times.keys()]
        times = list(times.values())

        self._t_init = info["time"][0]
        self._t_max = times[-1]
        self._done = False
        self._plan = self.compute_plan(times, poses, grasps)

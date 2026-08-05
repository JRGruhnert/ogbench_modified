import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class FaucetPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        faucet_initial = plan_input["faucet_initial"]
        faucet_goal = plan_input["faucet_goal"]
        init_knob_angle = plan_input["init_knob_angle"]
        goal_knob_angle = plan_input["goal_knob_angle"]
        center = plan_input["faucet_center"]
        # Read handle radius from the model (scale-aware, matches XML after scaling).
        radius = abs(float(self._env.unwrapped._model.site("faucet_handle_center").pos[1]))

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
            times[f"arc_{i}"] = t + self._dt * 0.4
            t = times[f"arc_{i}"]
        times["dwell"] = t + self._dt * 0.5
        times["lift"] = times["dwell"] + self._dt * 0.5
        times["final"] = times["lift"] + self._dt
        self.add_neutral_yaw_prephase(poses["initial"], times, poses)
        self.jitter_times(times)

        grasps = {}
        for name in times:
            grasps[name] = 1.0  # gripper always closed

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        if f"heca_target_faucet_{i}_pos_ee" in info:
            target_handle_pos = info[f"heca_target_faucet_{i}_pos_ee"]
            target_faucet_yaw = info[f"heca_target_faucet_{i}_pos"][0]
            faucet = env.get_object(f"faucet_{i}")
        else:
            faucet = env.get_object(f"faucet_{i}")
            target_handle_pos = env._data.site_xpos[faucet._target_site_id].copy()
            target_faucet_yaw = faucet._target_val

        faucet_center = env._data.xpos[faucet._body_id].copy()

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "faucet_initial": self.to_pose(
                pos=info[f"heca_faucet_{i}_pos_ee"],
                yaw=info[f"heca_faucet_{i}_yaw"][0],
            ),
            "faucet_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info[f"heca_faucet_{i}_yaw"][0],
            ),
            "faucet_center": faucet_center,
            "init_knob_angle": env._data.joint(faucet.joint_name).qpos[0],
            "goal_knob_angle": target_faucet_yaw,
        }

        self.finalize_plan(plan_input, info)

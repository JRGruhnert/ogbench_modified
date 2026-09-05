import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class FaucetPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def _arc_poses(
        self,
        faucet_initial,
        center,
        body_xmat,
        init_angle,
        goal_angle,
        push_offset=0.08,
        radius=0.105,
        step_angle=0.15,
    ):
        """Arc poses in the body's local frame (handle at [0, -radius, 0]).

        The number of arc poses scales with the angular sweep so the chord
        approximation stays accurate regardless of how far the knob must turn.
        """
        handle_z = faucet_initial.translation()[2]
        base_yaw = self.get_yaw(faucet_initial)
        local_handle = np.array([0.0, -radius, 0.0])  # fixed in body frame
        n_arc = max(2, int(np.ceil(abs(goal_angle - init_angle) / step_angle)))
        arc_angles = np.linspace(init_angle, goal_angle, n_arc)
        arc_poses = []
        for angle in arc_angles:
            dz = angle - init_angle
            rot_dz = np.array(
                [[np.cos(dz), -np.sin(dz), 0], [np.sin(dz), np.cos(dz), 0], [0, 0, 1]]
            )
            world = body_xmat @ rot_dz @ local_handle
            pos = np.array([center[0] + world[0], center[1] + world[1], handle_z])
            yaw = base_yaw + dz
            arc_poses.append(self.to_pose(pos=pos, yaw=yaw))

        # Tangent approach in local frame
        delta = goal_angle - init_angle
        if delta >= 0:
            local_dir = np.array([-1.0, 0.0, 0.0])  # approach from -x (CW side)
        else:
            local_dir = np.array([1.0, 0.0, 0.0])  # approach from +x (CCW side)
        # Offset the end effector tangentially by rotating the handle point
        # around the knob center, so it stays on the handle's circle (radius)
        # instead of drifting outward.
        phi = push_offset / radius
        angle = local_dir[0] * phi  # local_dir[0] is +1 or -1
        rot_phi = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        approach_local = rot_phi @ local_handle
        approach_world = body_xmat @ approach_local
        approach_xy = center[:2] + approach_world[:2]
        approach_pose = self.to_pose(
            pos=np.array([approach_xy[0], approach_xy[1], handle_z]),
            yaw=base_yaw,
        )

        # The goal EE pose is just the last arc pose (handle at the goal angle);
        # `compute_keyframes` lifts straight up from it.
        return arc_poses, approach_pose

    def compute_keyframes(self, plan_input):

        arc_poses, approach_pose = self._arc_poses(
            plan_input["faucet_initial"],
            plan_input["faucet_center"],
            plan_input["body_xmat"],
            plan_input["init_knob_angle"],
            plan_input["goal_knob_angle"],
            radius=plan_input["handle_radius"],
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(approach_pose, 0.10)
        poses["down"] = approach_pose
        for i, p in enumerate(arc_poses):
            poses[f"arc_{i}"] = p
        poses["lift"] = self.above(poses[f"arc_{len(arc_poses) -1}"], 0.10)
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = self.distance(poses["initial"], poses["approach"])
        distance2 = self.distance(poses["lift"], poses["final"])
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.8 + distance * 4)
        times["down"] = self._dt
        for i in range(len(arc_poses)):
            times[f"arc_{i}"] = self._dt * 0.4
        times["lift"] = self._dt
        times["final"] = self._dt * (0.8 + distance2 * 4)

        # Grasps
        grasps = {}
        for name in times:
            grasps[name] = 1.0

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["approach", "down", f"arc_{len(arc_poses) -1}", "lift"],
        )

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        target_handle_pos = info[f"heca_target_faucet{i}_pos"]
        target_faucet_yaw = info[f"heca_target_faucet{i}_ang"][0]
        faucet = env.get_object(f"faucet{i}")

        faucet_center = env._data.xpos[faucet._body_id].copy()
        body_xmat = env._data.xmat[faucet._body_id].reshape(3, 3).copy()

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
                pos=info[f"heca_faucet{i}_pos"],
                yaw=info[f"heca_faucet{i}_yaw"][0],
            ),
            "faucet_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info[f"heca_faucet{i}_yaw"][0],
            ),
            "faucet_center": faucet_center,
            "body_xmat": body_xmat,
            "handle_radius": faucet.handle_radius,
            "init_knob_angle": env._data.joint(faucet.joint_name).qpos[0],
            "goal_knob_angle": target_faucet_yaw,
        }

        self.finalize_plan(plan_input, info)

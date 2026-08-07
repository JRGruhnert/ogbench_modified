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
        init_angle,
        goal_angle,
        n_arc=6,
        push_offset=0.08,
        radius=0.105,
    ):
        handle_z = faucet_initial.translation()[2]
        base_yaw = self.get_yaw(faucet_initial)
        arc_angles = np.linspace(init_angle, goal_angle, n_arc)
        arc_poses = []
        for angle in arc_angles:
            xy = center[:2] + radius * np.array([np.sin(angle), -np.cos(angle)])
            pos = np.array([xy[0], xy[1], handle_z])
            yaw = base_yaw + (angle - init_angle)
            arc_poses.append(self.to_pose(pos=pos, yaw=yaw))

        delta = goal_angle - init_angle
        if delta >= 0:
            init_dir = np.array([-np.cos(init_angle), -np.sin(init_angle)])
        else:
            init_dir = np.array([np.cos(init_angle), np.sin(init_angle)])
        approach_xy = (
            center[:2]
            + radius * np.array([np.sin(init_angle), -np.cos(init_angle)])
            + init_dir * push_offset
        )
        approach_pose = self.to_pose(
            pos=np.array([approach_xy[0], approach_xy[1], handle_z]),
            yaw=base_yaw,
        )

        return arc_poses, approach_pose

    def compute_keyframes(self, plan_input):

        arc_poses, approach_pose = self._arc_poses(
            plan_input["faucet_initial"],
            plan_input["faucet_center"],
            plan_input["init_knob_angle"],
            plan_input["goal_knob_angle"],
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(approach_pose, 0.10)
        poses["down"] = approach_pose
        for i, p in enumerate(arc_poses):
            poses[f"arc_{i}"] = p
        poses["lift"] = self.above(arc_poses[-1], 0.10)
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = np.linalg.norm(
            poses["initial"].translation() - poses["approach"].translation()
        )
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.5 + distance * 4)
        times["down"] = self._dt * 0.5
        for i in range(len(arc_poses)):
            times[f"arc_{i}"] = self._dt * 0.6
        times["lift"] = self._dt * 0.5
        times["final"] = self._dt

        # Grasps
        grasps = {}
        for name in times:
            grasps[name] = 1.0

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["down", f"arc_{len(arc_poses) -1}"],
        )

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

import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class LeverPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def _arc_poses(
        self, lever_initial, center, init_angle, goal_angle, n_arc=6, radius=0.12
    ):
        """Compute arc poses for the lever handle moving in a vertical arc.

        The lever rotates around the X axis (hinge axis), so the handle moves
        in the YZ plane.

        Returns (arc_poses, approach_pose).
        """
        base_yaw = self.get_yaw(lever_initial)

        arc_angles = np.linspace(init_angle, goal_angle, n_arc)
        arc_poses = []
        for angle in arc_angles:
            y = center[1] - radius * np.cos(angle)
            z = center[2] - radius * np.sin(angle)
            pos = np.array([center[0], y, z])
            arc_poses.append(self.to_pose(pos=pos, yaw=base_yaw))

        # First arc pose (initial handle position).
        approach_pose = arc_poses[0]

        return arc_poses, approach_pose

    def compute_keyframes(self, plan_input):
        arc_poses, approach_pose = self._arc_poses(
            plan_input["lever_initial"],
            plan_input["lever_center"],
            plan_input["init_angle"],
            plan_input["goal_angle"],
        )
        n_arc = len(arc_poses)

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(approach_pose, 0.08)
        poses["grasp"] = approach_pose
        for i, p in enumerate(arc_poses):
            poses[f"arc_{i}"] = p
        poses["release"] = arc_poses[-1]
        poses["leave"] = self.above(arc_poses[-1], 0.08)
        poses["final"] = plan_input["effector_goal"]

        # Times
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt
        times["grasp"] = self._dt * 0.5
        for i in range(n_arc):
            times[f"arc_{i}"] = self._dt * 0.4
        times["release"] = self._dt * 0.5
        times["clearance"] = self._dt * 0.5
        times["final"] = self._dt

        # Grasps
        grasps = self.build_grasps(times, {"grasp", "release"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["grasp", f"arc_{n_arc - 1}"],
        )
        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        if f"heca_target_lever_{i}_pos_ee" in info:
            target_handle_pos = info[f"heca_target_lever_{i}_pos_ee"]
        else:
            target_handle_pos = env._data.site_xpos[
                env.get_object(f"lever_{i}")._target_site_id
            ].copy()

        lever = env.get_object(f"lever_{i}")
        lever_center = env._data.xpos[lever._body_id].copy()

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "lever_initial": self.to_pose(
                pos=info[f"heca_lever_{i}_pos_ee"],
                yaw=info[f"heca_lever_{i}_yaw"][0],
            ),
            "lever_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info[f"heca_lever_{i}_yaw"][0],
            ),
            "lever_center": lever_center,
            "init_angle": env._data.joint(lever.joint_name).qpos[0],
            "goal_angle": lever._target_val,
        }

        self.finalize_plan(plan_input, info)

import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class ButtonPlanOracle(PlanOracle):
    def __init__(self, object_id=0, gripper_always_closed=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id
        self._gripper_always_closed = gripper_always_closed

    def compute_keyframes(self, plan_input):
        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(plan_input["button"], 0.06)
        poses["press"] = self.above(plan_input["button"], -0.02, noise=0.0)
        poses["leave"] = poses["approach"]
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = np.linalg.norm(
            poses["initial"].translation() - poses["approach"].translation()
        )
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.5 + distance * 4)
        times["press"] = self._dt * 0.8
        times["leave"] = self._dt * 0.8
        times["final"] = self._dt * 1.25

        # Grasps
        grasps = self.build_grasps(times, {"approach", "final"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["press"],
        )
        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        target_button_top_pos = info[f"heca_target_button{i}_pos"]

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "button": self.to_pose(
                pos=target_button_top_pos,
                yaw=0.0,
            ),
        }

        self.finalize_plan(plan_input, info)

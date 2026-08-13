import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class SliderPlanOracle(PlanOracle):
    def __init__(self, object_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object_id = object_id

    def compute_keyframes(self, plan_input):
        slider_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["slider_initial"]) + np.pi / 2,
            translation=plan_input["slider_initial"].translation(),
            n=2,
        )
        slider_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(slider_initial),
            obj_yaw=self.get_yaw(plan_input["slider_initial"]) + np.pi / 2,
            translation=plan_input["slider_goal"].translation(),
            n=2,
        )

        # Poses
        poses = {}
        poses["initial"] = plan_input["effector_initial"]
        poses["approach"] = self.above(slider_initial, 0.06)
        poses["grasp-start"] = slider_initial
        poses["grasp-end"] = slider_initial
        poses["move"] = slider_goal
        poses["release-start"] = slider_goal
        poses["release-end"] = self.above(slider_goal, 0.06)
        poses["final"] = plan_input["effector_goal"]

        # Times
        distance = np.linalg.norm(
            poses["initial"].translation() - poses["approach"].translation()
        )
        distance2 = np.linalg.norm(
            poses["approach"].translation() - poses["final"].translation()
        )
        times = {}
        times["initial"] = 0.0
        times["approach"] = self._dt * (0.8 + distance * 4)
        times["grasp-start"] = self._dt * 0.5
        times["grasp-end"] = self._dt * 0.5
        times["move"] = self._dt * (0.8 + distance2 * 4)
        times["release-start"] = self._dt * 0.5
        times["release-end"] = self._dt * 0.5
        times["final"] = self._dt

        # Grasps
        grasps = self.build_grasps(times, {"grasp-end", "release-start"})

        # Postprocess
        times, poses, grasps = self.process_keyframes(
            times,
            poses,
            grasps,
            checkpoints=["grasp-end", "release-start"],
        )

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        i = self._object_id
        target_handle_pos = info[f"heca_target_slider{i}_pos"]

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "slider_initial": self.to_pose(
                pos=info[f"heca_slider{i}_pos"],
                yaw=info[f"heca_slider{i}_yaw"][0],
            ),
            "slider_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info[f"heca_slider{i}_yaw"][0],
            ),
        }

        self.finalize_plan(plan_input, info)

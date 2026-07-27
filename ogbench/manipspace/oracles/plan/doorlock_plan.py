import numpy as np

from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class DoorlockPlanOracle(PlanOracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_keyframes(self, plan_input):
        poses = {}
        doorlock_initial = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["doorlock_initial"]),
            translation=plan_input["doorlock_initial"].translation(),
            n=2,
        )
        doorlock_goal = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input["effector_initial"]),
            obj_yaw=self.get_yaw(plan_input["doorlock_initial"]),
            translation=plan_input["doorlock_goal"].translation(),
            n=2,
        )

        # Doorlock is now a top-opening chest: closing = push down, opening = pull up.
        is_closing = doorlock_goal.translation()[2] < doorlock_initial.translation()[2]

        if is_closing:
            # Push down with open gripper.
            poses["initial"] = plan_input["effector_initial"]
            poses["approach"] = self.above(doorlock_initial, 0.06)
            poses["push"] = doorlock_goal
            poses["retreat"] = self.above(doorlock_goal, 0.10)
            poses["final"] = plan_input["effector_goal"]

            times = {}
            times["initial"] = 0.0
            times["approach"] = times["initial"] + self._dt
            times["push"] = times["approach"] + self._dt * 0.5
            times["retreat"] = times["push"] + self._dt * 0.5
            times["final"] = times["retreat"] + self._dt
            self.jitter_times(times)

            grasps = {}
            for name in times.keys():
                grasps[name] = 0.0  # Gripper open throughout — just push

        else:
            # Opening: grasp handle and pull up.
            poses["initial"] = plan_input["effector_initial"]
            poses["approach"] = self.above(doorlock_initial, 0.06)
            poses["grasp_start"] = doorlock_initial
            poses["grasp_end"] = doorlock_initial
            poses["pull"] = self.above(doorlock_goal, 0.08)
            poses["release"] = self.above(doorlock_goal, 0.08)
            poses["clearance"] = self.above(doorlock_goal, 0.12)
            poses["final"] = plan_input["effector_goal"]

            times = {}
            times["initial"] = 0.0
            times["approach"] = times["initial"] + self._dt
            times["grasp_start"] = times["approach"] + self._dt * 0.5
            times["grasp_end"] = times["grasp_start"] + self._dt * 0.5
            times["pull"] = times["grasp_end"] + self._dt * 0.5
            times["release"] = times["pull"] + self._dt * 0.5
            times["clearance"] = times["release"] + self._dt * 0.5
            times["final"] = times["clearance"] + self._dt
            self.jitter_times(times)

            grasps = {}
            g = 0.0
            for name in times.keys():
                if name in {"grasp_end", "release"}:
                    g = 1.0 - g
                grasps[name] = g

        return times, poses, grasps

    def reset(self, ob, info):
        env = self._env.unwrapped
        if "privileged_target_doorlock_handle_pos" in info:
            target_handle_pos = info["privileged_target_doorlock_handle_pos"]
        else:
            target_handle_pos = env._data.site_xpos[
                env.get_object("doorlock")._target_site_id
            ].copy()

        plan_input = {
            "effector_initial": self.to_pose(
                pos=info["proprio_effector_pos"],
                yaw=info["proprio_effector_yaw"][0],
            ),
            "effector_goal": self.to_pose(
                pos=np.random.uniform(*env._arm_sampling_bounds),
                yaw=0.0,
            ),
            "doorlock_initial": self.to_pose(
                pos=info["privileged_doorlock_handle_pos"],
                yaw=info["privileged_doorlock_handle_yaw"][0],
            ),
            "doorlock_goal": self.to_pose(
                pos=target_handle_pos,
                yaw=info["privileged_doorlock_handle_yaw"][0],
            ),
        }

        self.finalize_plan(plan_input, info)

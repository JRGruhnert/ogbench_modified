import numpy as np
from ogbench.manipspace.envs.objects import DrawerObject, WindowObject, ButtonDoubleObject, CubeObject
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv1(SceneEnvBase):
    """Drawer + 2 buttons + window + 1 cube."""

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        self._objects = [
            CubeObject(count=1),
            ButtonDoubleObject(),
            DrawerObject(pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14)),
            WindowObject(pos=(0.3, 0.3, 0.202)),
        ]

        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

        self._button_locks = {0: "drawer_slide", 1: "window_slide"}

    def _configure_scene(self):
        self._object_sampling_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        self._target_sampling_bounds = self._object_sampling_bounds
        self._drawer_center = np.array([0.33, -0.24, 0.066])
        self._cube_colors = np.array([self._colors["red"], self._colors["blue"]])
        self._cube_success_colors = np.array(
            [self._colors["lightred"], self._colors["lightblue"]]
        )

    # ------------------------------------------------------------------
    # Task definitions
    # ------------------------------------------------------------------
    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
            ),
            dict(
                task_name="task2_unlock_and_lock",
                init=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task3_rearrange_medium",
                init=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button_states=np.array([1, 0]),
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=-0.16,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task4_put_in_drawer",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([1, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task5_rearrange_hard",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == "data_collection":
            self.initialize_arm()
            for i in range(self._num_cubes):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                self._data.joint(f"object_joint_{i}").qpos[:3] = (*xy, 0.02)
                self._data.joint(f"object_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(
                    yaw
                ).wxyz.tolist()
            for i in range(self._num_buttons):
                self._cur_button_states[i] = self.np_random.choice(
                    self._num_button_states
                )
            self._apply_button_states()
            for obj in self.objects:
                obj.randomize(self)
            self.set_new_target(return_info=False)
        else:
            # Task mode.
            perm = (
                self.np_random.permutation(self._num_cubes)
                if self._permute_blocks and self._num_cubes > 0
                else np.arange(self._num_cubes)
            )
            init_xyzs = (
                self.cur_task_info["init"]["block_xyzs"].copy()[perm]
                if self._num_cubes > 0
                else None
            )
            goal_xyzs = (
                self.cur_task_info["goal"]["block_xyzs"].copy()[perm]
                if self._num_cubes > 0
                else None
            )
            init_btn = self.cur_task_info["init"]["button_states"].copy()
            goal_btn = self.cur_task_info["goal"]["button_states"].copy()

            # Goal snapshot.
            saved_qpos, saved_qvel = self._data.qpos.copy(), self._data.qvel.copy()
            self.initialize_arm()
            self._set_cubes(goal_xyzs, goal_xyzs)
            self._cur_button_states = goal_btn.copy()
            self._apply_button_states()
            for obj in self._objects:
                obj.init_to_goal(self, self.cur_task_info)
            mujoco.mj_forward(self._model, self._data)
            for _ in range(2):
                self.step(self.action_space.sample())
            self._cur_goal_ob = (
                self.compute_oracle_observation()
                if self._use_oracle_rep
                else self.compute_ob_info()
            )
            self._cur_goal_rendered = (
                self.get_pixel_observation() if self._render_goal else None
            )

            # Init setup.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for i in range(self._num_cubes):
                p = init_xyzs[i].copy()
                p[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f"object_joint_{i}").qpos[:3] = p
                self._data.joint(f"object_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(
                    self.np_random.uniform(0, 2 * np.pi)
                ).wxyz.tolist()
            self._set_cubes(init_xyzs, goal_xyzs)  # mocap targets = goal positions
            self._cur_button_states = init_btn.copy()
            self._target_button_states = goal_btn.copy()
            self._apply_button_states()
            for obj in self._objects:
                obj.init_to_init(self, self.cur_task_info)

        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()
        self._success = False


    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode == "data_collection"

        available = [
            i
            for i in range(self._num_cubes)
            if not self._is_in_drawer(self._data.joint(f"object_joint_{i}").qpos[:3])
        ]

        # Compute task probabilities.
        probs_dict = self._get_task_probabilities()
        task_list, prob_list = [], []
        for t in self._task_types:
            if t in probs_dict:
                task_list.append(t)
                prob_list.append(probs_dict[t])
        probs = np.array(prob_list, dtype=float)
        probs /= probs.sum()
        self._target_task = self.np_random.choice(task_list, p=probs)

        if self._target_task == "cube":
            block_xyzs = np.array(
                [
                    self._data.joint(f"object_joint_{i}").qpos[:3]
                    for i in range(self._num_cubes)
                ]
            )
            top_blocks = []
            for i in range(self._num_cubes):
                if i not in available:
                    continue
                for j in range(self._num_cubes):
                    if (
                        i != j
                        and block_xyzs[j][2] > block_xyzs[i][2]
                        and (
                            np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02
                        )
                    ):
                        break
                else:
                    top_blocks.append(i)

            self._target_block = self.np_random.choice(top_blocks)
            drawer_open = (
                "drawer" in self._joint_objects
                and self._data.joint(self._joint_objects["drawer"]["joint"]).qpos[0]
                < -0.12
            )
            put_in_drawer = drawer_open and self.np_random.uniform() < 0.3
            stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack

            if put_in_drawer:
                tar_pos = self._drawer_center.copy()
                tar_pos[:2] += self.np_random.uniform(-0.005, 0.005, size=2)
            elif stack:
                other = self.np_random.choice(
                    list(set(top_blocks) - {self._target_block})
                )
                bp = self._data.joint(f"object_joint_{other}").qpos[:3]
                tar_pos = np.array([bp[0], bp[1], bp[2] + 0.04])
            else:
                xy = self.np_random.uniform(*self._target_sampling_bounds)
                tar_pos = (*xy, 0.02)

            yaw = self.np_random.uniform(0, 2 * np.pi)
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            identity_quat = lie.SO3.identity().wxyz.tolist()

            for i in range(self._num_cubes):
                if i == self._target_block:
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
                else:
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
                        identity_quat
                    )
            for i in range(self._num_cubes):
                alpha = (
                    0.2 if (self._visualize_info and i == self._target_block) else 0.0
                )
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = alpha

        elif self._target_task == "button":
            self._target_button = self.np_random.choice(self._num_buttons)
            self._target_button_states[self._target_button] = (
                self._cur_button_states[self._target_button] + 1
            ) % self._num_button_states

        else:
            for obj in self.objects:
                if getattr(obj, "var_prefix", None) == self._target_task:
                    obj.handle_target(self)
                    break
            else:
                if not self._handle_specific_target(self._target_task):
                    raise ValueError(f"Unknown target task: {self._target_task}")

        mujoco.mj_kinematics(self._model, self._data)

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def post_step(self):
        # Numerical stability check (task mode).
        if self._mode == "task":
            is_healthy = True
            for i in range(self._num_cubes):
                p = self._data.joint(f"object_joint_{i}").qpos[:3]
                if np.any(p <= self._workspace_bounds[0] - 0.2) or np.any(
                    p >= self._workspace_bounds[1] + 0.2
                ):
                    is_healthy = False
                    break
            if not is_healthy:
                print(
                    "Numerical instability detected. Resetting cube positions.",
                    flush=True,
                )
                for i in range(self._num_cubes):
                    xy = self.np_random.uniform(*self._object_sampling_bounds)
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    self._data.joint(f"object_joint_{i}").qpos[:3] = (*xy, 0.02)
                    self._data.joint(f"object_joint_{i}").qpos[3:] = (
                        lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    )
                if self._num_cubes > 0:
                    self._data.joint("object_joint_0").qvel[:] = 0.0
                mujoco.mj_forward(self._model, self._data)

        # Update button states.
        for i in range(self._num_buttons):
            prev = self._prev_ob_info[f"privileged_button_{i}_pos"][0]
            cur = self._data.joint(f"buttonbox_joint_{i}").qpos.copy()[0]
            if prev > -0.02 and cur <= -0.02:
                self._cur_button_states[i] = (
                    self._cur_button_states[i] + 1
                ) % self._num_button_states
        self._apply_button_states()

        # Evaluate successes.
        successes = self._compute_successes()
        self._success = all(val for val, _ in successes)

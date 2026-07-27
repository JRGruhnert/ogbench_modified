import mujoco
import numpy as np
from dm_control import mjcf

from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ogbench.manipspace.envs.objects.base import SceneObject


class SceneEnvBase(ManipSpaceEnv):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        self._env_type = env_type
        self._permute_blocks = permute_blocks

        self._derive_config_from_objects()
        super().__init__(*args, **kwargs)

        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        self._target_sampling_bounds = self._object_sampling_bounds
        self._drawer_center = np.array([0.33, -0.24, 0.066])

        if self._num_cubes > 0:
            self._cube_colors = np.array([self._colors["red"], self._colors["blue"]])
            self._cube_success_colors = np.array(
                [self._colors["lightred"], self._colors["lightblue"]]
            )

        self._num_button_states = 2
        self._cur_button_states = np.array([0] * self._num_buttons)
        self._target_task = self._task_types[0]
        self._target_block = 0
        self._target_button = 0
        self._target_button_states = np.array([0] * self._num_buttons)
        self._target_object_pos = {name: 0.0 for name in self._joint_objects}

    @property
    def objects(self) -> list[SceneObject]:
        return self._objects  # type: ignore

    # ------------------------------------------------------------------
    # set_state
    # ------------------------------------------------------------------
    def set_state(self, qpos, qvel, button_states):
        self._cur_button_states = button_states.copy()
        self._apply_button_states()
        super().set_state(qpos, qvel)

    # ------------------------------------------------------------------
    # set_tasks — subclasses override this
    # ------------------------------------------------------------------
    def set_tasks(self):
        raise NotImplementedError("Subclasses must implement set_tasks().")

    # ------------------------------------------------------------------
    # _derive_config_from_objects
    # ------------------------------------------------------------------
    def _derive_config_from_objects(self):
        """Derive configuration from the ``_objects`` list."""
        joint_objects = {}
        task_types = []
        num_cubes = 0
        num_buttons = 0
        buttons_xml = "buttons.xml"

        for obj in self.objects:
            if getattr(obj, "is_joint_object", False):
                joint_objects[obj.var_prefix] = {
                    "joint": obj.joint_name,
                    "site": obj.site_name,
                    "target_site": obj.target_site_name,
                    "material": getattr(obj, "material_name", None),
                    "pos_range": obj.pos_range,
                    "scaler": obj.scaler,
                    "tolerance": obj.tolerance,
                }
                task_types.append(obj.var_prefix)
            elif getattr(obj, "is_free_body", False):
                num_cubes = getattr(obj, "count", 1)
                if obj.var_prefix == "cube":
                    task_types.append("cube")
            elif getattr(obj, "is_button", False):
                num_buttons = obj.count
                buttons_xml = obj.xml_file

        if num_buttons > 0 and "button" not in task_types:
            task_types.append("button")

        self._joint_objects = joint_objects
        self._task_types = task_types
        self._button_locks = {}
        self._num_cubes = num_cubes
        self._num_buttons = num_buttons
        self._buttons_xml = buttons_xml

    # ------------------------------------------------------------------
    # _get_object_instance
    # ------------------------------------------------------------------
    def _get_object_instance(self, name):
        """Return the object instance for a given var_prefix, or None."""
        for obj in self.objects:
            if getattr(obj, "var_prefix", None) == name:
                return obj
        return None

    # ------------------------------------------------------------------
    # add_objects
    # ------------------------------------------------------------------
    def add_objects(self, arena_mjcf):
        # Cubes.
        if self._num_cubes > 0:
            cube_mjcf = mjcf.from_path((self._desc_dir / "cube.xml").as_posix())
            arena_mjcf.include_copy(cube_mjcf)
            self._cube_geoms_list = [
                cube_mjcf.find("body", f"object_{i}").find_all("geom")
                for i in range(self._num_cubes)
            ]
            self._cube_target_geoms_list = [
                cube_mjcf.find("body", f"object_target_{i}").find_all("geom")
                for i in range(self._num_cubes)
            ]

        # Buttons.
        button_mjcf = mjcf.from_path((self._desc_dir / self._buttons_xml).as_posix())
        arena_mjcf.include_copy(button_mjcf)
        self._button_geoms_list = [
            [button_mjcf.find("geom", f"btngeom_{i}")] for i in range(self._num_buttons)
        ]

        # Joint objects and other objects.
        loaded_xmls = set()
        for obj in self.objects:
            xml = getattr(type(obj), "xml_file", None)
            if xml is None:
                continue
            # Skip cube and button XMLs — handled manually above.
            if xml == "cube.xml" or xml.startswith("buttons"):
                continue

            obj_mjcf = mjcf.from_path((self._desc_dir / xml).as_posix())

            # Multi-instance: rename elements.
            if (
                xml in loaded_xmls
                and hasattr(obj, "instance_id")
                and obj.instance_id > 0
            ):
                type(obj).rename_in_xml(obj_mjcf, obj.instance_id)
            loaded_xmls.add(xml)

            # Set object body position / euler if specified.
            body_name = getattr(obj, "body_name", None) or getattr(
                obj, "var_prefix", None
            )
            body = obj_mjcf.find("body", body_name) if body_name else None
            if body is None:
                try:
                    body = obj_mjcf.worldbody.body[0]
                except (AttributeError, IndexError):
                    pass
            if body is not None:
                if getattr(obj, "pos", None) is not None:
                    body.pos = obj.pos
                if getattr(obj, "euler", None) is not None:
                    body.euler = obj.euler

            arena_mjcf.include_copy(obj_mjcf)

        self._add_specific_objects(arena_mjcf)

        # Cameras.
        for name, kwargs in {
            "front": {
                "pos": (1.139, 0.0, 0.821),
                "xyaxes": (0.0, 1.0, 0.0, -0.627, 0.0, 0.779),
            },
            "front_pixels": {
                "pos": (0.905, 0.0, 0.762),
                "xyaxes": (0.0, 1.0, 0.0, -0.771, 0.0, 0.637),
            },
        }.items():
            arena_mjcf.worldbody.add("camera", name=name, **kwargs)

    # ------------------------------------------------------------------
    # post_compilation_objects
    # ------------------------------------------------------------------
    def post_compilation_objects(self):
        if self._num_cubes > 0:
            self._cube_geom_ids_list = [
                [self._model.geom(g.full_identifier).id for g in cube_geoms]
                for cube_geoms in self._cube_geoms_list
            ]
            self._cube_target_mocap_ids = [
                self._model.body(f"object_target_{i}").mocapid[0]
                for i in range(self._num_cubes)
            ]
            self._cube_target_geom_ids_list = [
                [self._model.geom(g.full_identifier).id for g in geoms]
                for geoms in self._cube_target_geoms_list
            ]

        self._button_geom_ids_list = [
            [self._model.geom(g.full_identifier).id for g in btn_geoms]
            for btn_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [
            self._model.site(f"btntop_{i}").id for i in range(self._num_buttons)
        ]

        for obj in self.objects:
            obj.post_compilation(self)

        self._post_compilation_specific()

    # ------------------------------------------------------------------
    # _apply_button_states
    # ------------------------------------------------------------------
    def _apply_button_states(self):
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._colors[
                    "red" if self._cur_button_states[i] == 0 else "white"
                ]

        for obj in self.objects:
            obj.apply_lock(self, self._cur_button_states, self._button_locks)

        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # _set_cubes — helper for placing cube blocks
    # ------------------------------------------------------------------
    def _set_cubes(
        self, block_xyzs, mocap_xyzs=None, hide_others=False, target_idx=None
    ):
        """Place cubes at *block_xyzs*, optionally set mocap targets.

        When *hide_others* is True, non-target mocap bodies are moved out of sight.
        """
        identity_quat = lie.SO3.identity().wxyz.tolist()
        for i in range(self._num_cubes):
            self._data.joint(f"object_joint_{i}").qpos[:3] = block_xyzs[i]
            self._data.joint(f"object_joint_{i}").qpos[3:] = identity_quat
        if mocap_xyzs is not None:
            for i in range(self._num_cubes):
                if hide_others and i != target_idx:
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
                        identity_quat
                    )
                else:
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = mocap_xyzs[i]
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
                        identity_quat
                    )

    # ------------------------------------------------------------------
    # initialize_episode
    # ------------------------------------------------------------------
    def initialize_episode(self):
        if self._num_cubes > 0:
            for i in range(self._num_cubes):
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba = self._cube_colors[i]
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

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

    # ------------------------------------------------------------------
    # _is_in_drawer
    # ------------------------------------------------------------------
    def _is_in_drawer(self, obj_pos):
        if "drawer" not in self._joint_objects:
            return False
        y = self._data.site_xpos[
            self._model.site(self._joint_objects["drawer"]["site"]).id
        ][1]
        return bool(
            np.all([0.21, y - 0.27, 0.0] <= obj_pos)
            and np.all(obj_pos <= [0.45, y - 0.07, 0.15])
        )

    # ------------------------------------------------------------------
    # _is_grabbed
    # ------------------------------------------------------------------
    def _is_grabbed(self, obj_pos):
        close = (
            np.linalg.norm(obj_pos - self._data.site_xpos[self._pinch_site_id]) < 0.04
        )
        z_thresh = (
            self._drawer_center[2] + 0.01 if self._is_in_drawer(obj_pos) else 0.01
        )
        return close and obj_pos[2] > z_thresh

    # ------------------------------------------------------------------
    # set_new_target  (data-collection mode)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # _block_state
    # ------------------------------------------------------------------
    def _block_state(self, block_idx: int) -> int:
        return (
            0
            if self._is_in_drawer(
                self._data.joint(f"object_joint_{block_idx}").qpos[:3]
            )
            else 1
        )

    # ------------------------------------------------------------------
    # default_quaternion
    # ------------------------------------------------------------------
    def default_quaternion(self) -> np.ndarray:
        return np.array(lie.SO3.identity().wxyz.tolist())

    # ------------------------------------------------------------------
    # pre_step
    # ------------------------------------------------------------------
    def pre_step(self):
        self._prev_button_states = self._cur_button_states.copy()
        super().pre_step()

    # ------------------------------------------------------------------
    # _compute_successes
    # ------------------------------------------------------------------
    def _compute_successes(self):
        successes = []
        for obj in self.objects:
            result = obj.compute_success(self)
            if result is not None:
                successes.append(result)
        return successes

    # ------------------------------------------------------------------
    # post_step
    # ------------------------------------------------------------------
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
        success_lookup = {task_type: val for val, task_type in successes}
        if self._mode == "data_collection":
            self._success = success_lookup.get(self._target_task, True)
        else:
            self._success = all(val for val, _ in successes)

        # Adjust cube colours.
        cube_s = [
            bool(
                np.linalg.norm(
                    self._data.joint(f"object_joint_{i}").qpos[:3]
                    - self._data.mocap_pos[self._cube_target_mocap_ids[i]]
                )
                <= 0.04
            )
            for i in range(self._num_cubes)
        ]
        for i in range(self._num_cubes):
            vis = self._visualize_info and (
                self._mode == "task" or i == self._target_block
            )
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[3] = 0.2 if vis else 0.0
            for gid in self._cube_geom_ids_list[i]:
                if self._visualize_info and cube_s[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
                else:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

    # ------------------------------------------------------------------
    # add_object_info
    # ------------------------------------------------------------------
    def add_object_info(self, ob_info: dict):
        for obj in self.objects:
            ob_info.update(obj.get_info(self))

        # Data-collection target info.
        if self._mode == "data_collection":
            ob_info["privileged_target_task"] = self._target_task

            for obj in self.objects:
                ob_info.update(obj.get_target_info(self))

        ob_info["prev_button_states"] = self._prev_button_states.copy()
        ob_info["button_states"] = self._cur_button_states.copy()

    # ------------------------------------------------------------------
    # _button_state
    # ------------------------------------------------------------------
    def _button_state(self, button_idx: int) -> int:
        return 0 if self._cur_button_states[button_idx] == 0 else 1

    # ------------------------------------------------------------------
    # compute_observation
    # ------------------------------------------------------------------
    def compute_observation(self):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()

        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        ob_info = self.compute_ob_info()
        ob = [
            ob_info["proprio_joint_pos"],
            ob_info["proprio_joint_vel"],
            (ob_info["proprio_effector_pos"] - xyz_center) * xyz_scaler,
            np.cos(ob_info["proprio_effector_yaw"]),
            np.sin(ob_info["proprio_effector_yaw"]),
            ob_info["proprio_gripper_opening"] * 3.0,
            ob_info["proprio_gripper_contact"],
        ]

        for obj in self.objects:
            obj.add_observation(self, ob, ob_info)

        self._add_specific_observations(ob, ob_info, xyz_center, xyz_scaler)
        return np.concatenate(ob)

    # ------------------------------------------------------------------
    # compute_oracle_observation
    # ------------------------------------------------------------------
    def compute_oracle_observation(self):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        ob_info = self.compute_ob_info()
        ob = []

        for obj in self.objects:
            obj.add_oracle_obs(self, ob, ob_info)

        self._add_specific_oracle_obs(ob, ob_info, xyz_center, xyz_scaler)
        return np.concatenate(ob)

    # ------------------------------------------------------------------
    # compute_reward
    # ------------------------------------------------------------------
    def compute_reward(self):
        if self._reward_task_id is None:
            return super().compute_reward()

        successes = [val for val, _ in self._compute_successes()]
        return float(sum(successes) - len(successes))

    # ==================================================================
    # Default hook implementations (subclasses override as needed)
    # ==================================================================

    def _get_task_probabilities(self):
        """Return a dict mapping task_type → raw probability."""
        probs = {}
        available = sum(
            1
            for i in range(self._num_cubes)
            if not self._is_in_drawer(self._data.joint(f"object_joint_{i}").qpos[:3])
        )
        if "cube" in self._task_types:
            probs["cube"] = 1.0 if available > 0 else 0.0
        if "button" in self._task_types:
            probs["button"] = 1.0
        for obj in self.objects:
            prob = obj.get_task_probability(self)
            if prob is not None:
                prefix = getattr(obj, "var_prefix", None)
                if prefix is not None:
                    probs[prefix] = prob
        return probs

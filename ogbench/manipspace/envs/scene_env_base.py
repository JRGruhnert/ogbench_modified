import mujoco
import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ogbench.manipspace.envs.objects.base import SceneObject


class SceneEnvBase(ManipSpaceEnv):
    def __init__(self, env_type, objects=None, permute_blocks=True, *args, **kwargs):
        self._env_type = env_type
        self._objects = objects or []
        self._permute_blocks = permute_blocks
        super().__init__(*args, **kwargs)
        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])
        self._oracle_just_done = False
        self._task_selection_counts = {}
        self._cur_goal_ob = None
        self._cur_goal_rendered = None
        self._render_goal = False

    def set_tasks(self):
        self.task_infos = []

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        is_collection = self._mode in ("data_collection", "collection", "randomized")

        if is_collection:
            self.initialize_arm()
            for obj in self.objects:
                obj.randomize(self)
            self._apply_button_states()
            self.set_new_target(return_info=False)
        else:
            if self.cur_task_info is None:
                self.cur_task_id = 1
                self.cur_task_info = self.task_infos[0]

            saved_qpos, saved_qvel = self._data.qpos.copy(), self._data.qvel.copy()
            self.initialize_arm()
            for obj in self._objects:
                obj.init_to_goal(self, self.cur_task_info)
            self._apply_button_states()
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

            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for obj in self._objects:
                obj.init_to_init(self, self.cur_task_info)
            self._apply_button_states()

        self.pre_step()
        self.post_step()
        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode in ("data_collection", "collection", "randomized")
        self._oracle_just_done = True

        probs = self._get_task_probabilities()
        task_list, prob_list = [], []
        for obj in self.objects:
            if obj.name in probs:
                task_list.append(obj.name)
                prob_list.append(probs[obj.name])

        # Keep only tasks that are currently available (positive probability).
        available = [(n, w) for n, w in zip(task_list, prob_list) if w > 0]
        if not available:
            if return_info:
                return self.compute_observation(), self.get_reset_info()
            return

        names = [n for n, _ in available]
        raw = np.array([w for _, w in available], dtype=float)

        # Inverse-frequency balancing: make under-selected tasks more likely while
        # still respecting each object's availability weight.
        counts = np.array(
            [self._task_selection_counts.get(n, 0) for n in names], dtype=float
        )
        weights = raw / (counts + 1.0)
        weights /= weights.sum()

        self._target_task = self.np_random.choice(names, p=weights)
        self._task_selection_counts[self._target_task] = (
            self._task_selection_counts.get(self._target_task, 0) + 1
        )

        for obj in self.objects:
            if obj.name == self._target_task:
                obj.randomize(self)
                obj.handle_target(self)
                break

        mujoco.mj_kinematics(self._model, self._data)

        # Compute the goal observation (target state) for goal-conditioned data.
        self._cur_goal_ob = self._compute_goal_observation()
        self._cur_goal_rendered = (
            self.get_pixel_observation() if self._render_goal else None
        )

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def _set_target_object_to_target(self):
        """Move the target object to its current target state."""
        for obj in self.objects:
            if obj.name != self._target_task:
                continue
            if hasattr(obj, "_target_mocap_id"):
                pos = self._data.mocap_pos[obj._target_mocap_id].copy()
                quat = self._data.mocap_quat[obj._target_mocap_id].copy()
                self._data.joint(obj.joint_name).qpos[:3] = pos
                self._data.joint(obj.joint_name).qpos[3:] = quat
            elif hasattr(obj, "_target_val"):
                self._data.joint(obj.joint_name).qpos[0] = obj._target_val
            elif hasattr(obj, "_target_button_states"):
                obj._cur_state[0] = obj._target_button_states[0]
            break

    def _compute_goal_observation(self):
        """Compute the observation of the target state (used as the goal)."""
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        saved_oracle_just_done = self._oracle_just_done

        # Some objects store discrete state outside of qpos/qvel (e.g. buttons
        # track `_cur_state`). Save and restore it so computing the goal does
        # not corrupt the current episode state.
        saved_cur_states = {}
        for obj in self.objects:
            if hasattr(obj, "_cur_state"):
                saved_cur_states[obj.name] = obj._cur_state.copy()

        self._set_target_object_to_target()
        self._apply_button_states()

        goal_ob = (
            self.compute_oracle_observation()
            if self._use_oracle_rep
            else self.compute_ob_info()
        )

        self._data.qpos[:] = saved_qpos
        self._data.qvel[:] = saved_qvel
        for obj in self.objects:
            if obj.name in saved_cur_states:
                obj._cur_state[:] = saved_cur_states[obj.name]
        self._oracle_just_done = saved_oracle_just_done

        # Restore colors/locks to match the restored current state.
        self._apply_button_states()
        return goal_ob

    def _get_task_probabilities(self):
        probs = {}
        for obj in self.objects:
            prob = obj.get_task_probability(self)
            if prob is not None:
                probs[obj.name] = prob
        return probs

    def _apply_button_states(self):
        for obj in self.objects:
            obj.apply_colors_and_locks(self)
        mujoco.mj_forward(self._model, self._data)

    def add_objects(self, arena_mjcf):
        for obj in self.objects:
            obj.load(arena_mjcf, self._desc_dir)
        self.add_cameras(arena_mjcf)

    def add_cameras(self, arena_mjcf):
        # Add cameras.
        cameras = {
            "front": {
                "pos": (1.139, 0.000, 0.821),
                "xyaxes": (0.000, 1.000, 0.000, -0.627, 0.000, 0.779),
            },
            "front_pixels": {
                "pos": (0.905, 0.000, 0.762),
                "xyaxes": (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add("camera", name=camera_name, **camera_kwargs)

    @property
    def objects(self) -> list[SceneObject]:
        return self._objects

    def get_object(self, name):
        """Find a SceneObject by name."""
        for obj in self._objects:
            if obj.name == name:
                return obj
        return None

    def set_state(self, qpos, qvel):
        for obj in self.objects:
            obj.apply_lock(self._model)

        mujoco.mj_forward(self._model, self._data)  # type: ignore
        super().set_state(qpos, qvel)

    def post_compilation_objects(self):
        for obj in self.objects:
            obj.post_compilation(self)

    def default_quaternion(self) -> np.ndarray:
        return np.array(lie.SO3.identity().wxyz.tolist())

    def pre_step(self):
        for obj in self.objects:
            obj.pre_step()
        super().pre_step()

    def _compute_successes(self):
        successes = []
        for obj in self.objects:
            result = obj.compute_success(self)
            if result is not None:
                successes.append(result)
        return successes

    def post_step(self):
        successes = self._compute_successes()
        self._success = all(val for val, _ in successes)

        for obj in self.objects:
            obj.post_step(self)
            obj.health_check_and_colors(self, successes)

        self._apply_button_states()

    def add_object_info(self, ob_info: dict):
        for obj in self.objects:
            ob_info.update(obj.get_info(self))

        if self._mode in ("data_collection", "collection", "randomized"):
            ob_info["privileged_target_task"] = self._target_task
            ob_info["oracle_done"] = float(self._oracle_just_done)
            self._oracle_just_done = False
            for obj in self.objects:
                ob_info.update(obj.get_info_target(self))
            # Oracle success: is the current target object at its goal?
            ob_info["oracle_success"] = float(
                any(
                    val
                    for val, name in self._compute_successes()
                    if name == self._target_task
                )
            )
            # Workspace normalization metadata (same values as compute_observation)
            ob_info["meta_xyz_center"] = np.array([0.425, 0.0, 0.0])
            ob_info["meta_xyz_scaler"] = np.array([10.0])
            ob_info["meta_gripper_scaler"] = np.array([3.0])
            ob_info["meta_prismatic_max"] = np.array([3.0])

    def get_reset_info(self):
        reset_info = super().get_reset_info()
        if self._mode in ("data_collection", "collection", "randomized"):
            reset_info["goal"] = self._cur_goal_ob
            if self._render_goal and self._cur_goal_rendered is not None:
                reset_info["goal_rendered"] = self._cur_goal_rendered
        return reset_info

    def get_step_info(self):
        ob_info = super().get_step_info()
        if self._mode in ("data_collection", "collection", "randomized"):
            ob_info["goal"] = self._cur_goal_ob
        return ob_info

    def _append_object_state(self, ob: list):
        """Append each object's goal-relevant state to the observation list."""
        for obj in self.objects:
            if hasattr(obj, "_target_mocap_id"):
                # Free body: position (3D).
                ob.append(self._data.joint(obj.joint_name).qpos[:3].copy())
            elif hasattr(obj, "_target_val"):
                # Articulated joint: joint value (1D).
                ob.append(np.array([self._data.joint(obj.joint_name).qpos[0]]))
            elif hasattr(obj, "_target_button_states"):
                # Button: discrete state (1D).
                ob.append(np.array([obj._cur_state[0]], dtype=np.float64))
            # Passive containers (shelf/box) contribute no state.

    def compute_observation(self):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()

        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        ob_info = self.compute_ob_info()
        ob = [
            ob_info["proprio_joint_pos"],
            ob_info["proprio_joint_vel"],
            (ob_info["proprio_effector_pos"] - xyz_center) * xyz_scaler,
            np.cos(ob_info["proprio_effector_yaw"]),
            np.sin(ob_info["proprio_effector_yaw"]),
            ob_info["proprio_gripper_opening"] * gripper_scaler,
            ob_info["proprio_gripper_contact"],
        ]
        self._append_object_state(ob)
        return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        ob = []
        self._append_object_state(ob)
        return np.concatenate(ob)

    def compute_reward(self):
        successes = [val for val, _ in self._compute_successes()]
        return float(all(successes))

    def set_scene_state(self, state_dict: dict):
        for name, value in state_dict.items():
            obj = self.get_object(name)
            if obj is None:
                continue
            if not obj.can_set_state(self, value):
                return False

        for name, value in state_dict.items():
            obj = self.get_object(name)
            if obj is None:
                continue
            obj.set_state(self, value)

        self._apply_button_states()
        return True

    def _object_state_from_info(self, obj, info_dict):
        if hasattr(obj, "_target_button_states"):
            key = f"heca_{obj.name}_ste"
            if key in info_dict:
                return int(round(float(np.asarray(info_dict[key]).ravel()[0])))
            return None

        if hasattr(obj, "_target_val"):
            for suffix in ("_ang", "_sca"):
                key = f"heca_{obj.name}{suffix}"
                if key in info_dict:
                    return float(np.asarray(info_dict[key]).ravel()[0])
            return None

        if hasattr(obj, "_target_mocap_id"):
            base_key = f"heca_{obj.name}_pos_base"
            pos_key = f"heca_{obj.name}_pos"
            if base_key in info_dict:
                pos = np.asarray(info_dict[base_key], dtype=float).ravel()
            elif pos_key in info_dict:
                pos = np.asarray(info_dict[pos_key], dtype=float).ravel()
                # Some free bodies only expose the handle site as `_pos` (e.g.
                # the lid). Recover the base by removing the fixed local offset.
                handle_offset = getattr(obj, "handle_offset", None)
                if handle_offset is not None:
                    pos = pos - np.asarray(handle_offset, dtype=float)
            else:
                return None
            rot_key = f"heca_{obj.name}_rot"
            yaw_key = f"heca_{obj.name}_yaw"
            if rot_key in info_dict:
                quat = np.asarray(info_dict[rot_key], dtype=float).ravel()
            elif yaw_key in info_dict:
                quat = lie.SO3.from_z_radians(
                    float(np.asarray(info_dict[yaw_key]).ravel()[0])
                ).wxyz
            else:
                quat = lie.SO3.identity().wxyz
            return (pos, quat)

        return None

    def _noisy_value(self, obj, value, noise_scale):
        if noise_scale <= 0.0:
            return value

        # Free body: value is (pos, quat).
        if isinstance(value, tuple):
            pos, quat = value
            pos = np.asarray(pos, dtype=float)
            # Perturb only x/y (the axes with sampling bounds); keep the
            # requested height so the object stays on its surface/stack.
            pos[:2] += self.np_random.normal(0.0, noise_scale, size=2)
            bounds = getattr(obj, "_sampling_bounds", None)
            if bounds is None:
                bounds = getattr(self, "_object_sampling_bounds", None)
            if bounds is not None:
                bounds = np.asarray(bounds, dtype=float)
                # Bounds convention is [[xlo, ylo], [xhi, yhi]], so the per-axis
                # lower limits are `bounds[0]` and upper limits `bounds[1]`.
                if bounds.shape == (2, 2):
                    pos[:2] = np.clip(pos[:2], bounds[0], bounds[1])
                elif bounds.shape == (2, 3):
                    pos = np.clip(pos, bounds[0], bounds[1])
            return (pos, quat)

        # Discrete button state: no continuous noise.
        if isinstance(value, (int, np.integer)):
            return value

        # Continuous joint value: clip to the object's valid range.
        val = float(value) + self.np_random.normal(0.0, noise_scale)
        pos_range = getattr(obj, "pos_range", None)
        if pos_range is not None:
            val = float(np.clip(val, pos_range[0], pos_range[1]))
        return val

    def step_scene(
        self,
        info_dict,
        terminate=0.3,
        skip=0.1,
        random=0.6,
        noise_scale=0.2,
    ):
        mode_probs = np.array([terminate, skip, random], dtype=float)
        if np.any(mode_probs < 0):
            raise ValueError("Failure mode probabilities must be non-negative.")
        if mode_probs.sum() <= 0:
            raise ValueError(
                "At least one failure mode probability must be positive; "
                f"got terminate={terminate}, skip={skip}, random={random}."
            )
        mode = self.np_random.choice(
            ["terminate", "skip", "random"], p=mode_probs / mode_probs.sum()
        )

        self.pre_step()

        # Parse requested updates and detect locked objects.
        targets = {}
        for obj in self.objects:
            value = self._object_state_from_info(obj, info_dict)
            if value is not None:
                targets[obj.name] = (obj, value)

        locked = {
            name
            for name, (obj, value) in targets.items()
            if not obj.can_set_state(self, value)
        }

        if mode == "terminate":
            if locked:
                self._success = False
                ob = self.compute_observation()
                info = self.get_step_info()
                info["success"] = False
                info["failure_mode"] = mode
                return ob, self.compute_reward(), True, False, info
        elif mode == "skip":
            if locked:
                ob = self.compute_observation()
                info = self.get_step_info()
                info["success"] = bool(self._success)
                info["failure_mode"] = mode
                return (
                    ob,
                    self.compute_reward(),
                    self.terminate_episode(),
                    self.truncate_episode(),
                    info,
                )

        # Apply the updates.
        for name, (obj, value) in targets.items():
            if mode == "random":
                # Locked values can never change; only unlocked objects are
                # resampled (randomly perturbed) instead of applied exactly.
                if name in locked:
                    continue
                value = self._noisy_value(obj, value, noise_scale)
            obj.set_state(self, value)

            joint_name = getattr(obj, "joint_name", None)
            if joint_name is not None:
                self._data.joint(joint_name).qvel[:] = 0.0

        self._apply_button_states()
        self._success = all(val for val, _ in self._compute_successes())

        ob = self.compute_observation()
        info = self.get_step_info()
        info["success"] = bool(self._success)
        info["failure_mode"] = mode
        reward = self.compute_reward()
        terminated = self.terminate_episode()
        truncated = self.truncate_episode()
        return ob, reward, terminated, truncated, info

    def set_start(self, info_dict, return_info=True):
        """Teleport the scene to a start configuration (ignores locks).

        Entities in `info_dict` are set exactly to their requested values.
        All other entities are randomized, and their goal is pinned to their
        (random) current state so start == goal for them and they never block
        success. Pair with `set_goal` for evaluation.
        """
        # Parse requested updates.
        targets = {}
        for obj in self.objects:
            value = self._object_state_from_info(obj, info_dict)
            if value is not None:
                targets[obj.name] = (obj, value)

        # Apply every requested state exactly (locks are ignored: this is a
        # reset-like teleport to a desired start configuration).
        for name, (obj, value) in targets.items():
            obj.set_state(self, value)

            joint_name = getattr(obj, "joint_name", None)
            if joint_name is not None:
                self._data.joint(joint_name).qvel[:] = 0.0

        # Entities not part of the start configuration: randomize and pin the
        # goal to the randomized current state (start == goal for them).
        for obj in self.objects:
            if obj.name in targets:
                continue
            if not (
                hasattr(obj, "_target_mocap_id")
                or hasattr(obj, "_target_val")
                or hasattr(obj, "_target_button_states")
            ):
                continue  # Passive containers (shelf/box).
            obj.randomize(self)
            joint_name = getattr(obj, "joint_name", None)
            if joint_name is not None:
                self._data.joint(joint_name).qvel[:] = 0.0
            self._pin_target_to_current(obj)

        self._apply_button_states()
        self._success = all(val for val, _ in self._compute_successes())

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def _set_object_target(self, obj, value):
        """Set only the *goal* (target) of an object, leaving its current state.

        Free bodies: target mocap pose. Articulated objects: `_target_val`
        (plus the visual goal site where the object exposes `_set_site`).
        Buttons: `_target_button_states`.
        """
        if hasattr(obj, "_target_mocap_id"):
            pos, quat = value
            self._data.mocap_pos[obj._target_mocap_id] = pos
            self._data.mocap_quat[obj._target_mocap_id] = quat
        elif hasattr(obj, "_target_val"):
            val = float(np.asarray(value).ravel()[0])
            obj._target_val = val
            set_site = getattr(obj, "_set_site", None)
            if set_site is not None:
                set_site(self, val)
        elif hasattr(obj, "_target_button_states"):
            obj._target_button_states[0] = int(
                round(float(np.asarray(value).ravel()[0]))
            )

    def _pin_target_to_current(self, obj):
        """Set the object's goal to its current state (start == goal)."""
        if hasattr(obj, "_target_mocap_id"):
            pos = self._data.joint(obj.joint_name).qpos[:3].copy()
            quat = self._data.joint(obj.joint_name).qpos[3:].copy()
            self._data.mocap_pos[obj._target_mocap_id] = pos
            self._data.mocap_quat[obj._target_mocap_id] = quat
        elif hasattr(obj, "_target_val"):
            self._set_object_target(obj, self._data.joint(obj.joint_name).qpos[0])
        elif hasattr(obj, "_target_button_states"):
            obj._target_button_states[0] = int(obj._cur_state[0])

    def set_goal(self, info_dict, return_info=True):
        """Set only the goal (target) state of the entities in `info_dict`.

        The current scene is untouched — only each object's goal changes
        (target mocap pose, `_target_val`, or `_target_button_states`).
        Combine with `set_start`: after the model reaches the goals,
        `info["success"]` / `compute_reward()` return True. The goal
        observation (`reset_info["goal"]`) is recomputed accordingly.
        """
        # Parse requested goals.
        targets = {}
        for obj in self.objects:
            value = self._object_state_from_info(obj, info_dict)
            if value is not None:
                targets[obj.name] = (obj, value)

        # Set only the goals (targets), never the current state.
        for name, (obj, value) in targets.items():
            self._set_object_target(obj, value)

        # Recompute the goal observation with all goal objects at their goals.
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        saved_cur_states = {}
        for obj in self.objects:
            if hasattr(obj, "_cur_state"):
                saved_cur_states[obj.name] = obj._cur_state.copy()

        for name, (obj, value) in targets.items():
            obj.set_state(self, value)
        self._apply_button_states()
        self._cur_goal_ob = (
            self.compute_oracle_observation()
            if self._use_oracle_rep
            else self.compute_ob_info()
        )
        self._cur_goal_rendered = (
            self.get_pixel_observation() if self._render_goal else None
        )

        # Restore the current state (goals persist: mocap / _target_val /
        # _target_button_states were set by _set_object_target).
        self._data.qpos[:] = saved_qpos
        self._data.qvel[:] = saved_qvel
        for obj in self.objects:
            if obj.name in saved_cur_states:
                obj._cur_state[:] = saved_cur_states[obj.name]
        self._apply_button_states()

        self._success = all(val for val, _ in self._compute_successes())

        if return_info:
            return self.compute_observation(), self.get_reset_info()

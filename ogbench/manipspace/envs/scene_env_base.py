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
                ob.append(
                    np.array([self._data.joint(obj.joint_name).qpos[0]])
                )
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
        return float(sum(successes) - len(successes))

    def set_scene_state(self, state_dict: dict):
        """Teleport the scene to a given state. Respects lock rules — if any object
        refuses, no state changes are made.

        Args:
            state_dict: dict mapping object name → value.
                        Joint objects: float (joint position).
                        Free-body objects: (pos, quat) tuple.
                        Buttons: int (0 or 1).
        Returns:
            True if state was set, False if any object refused (locked).
        """
        # Phase 1: check all objects.
        for name, value in state_dict.items():
            obj = self.get_object(name)
            if obj is None:
                continue
            if not obj.can_set_state(self, value):
                return False

        # Phase 2: apply all changes.
        for name, value in state_dict.items():
            obj = self.get_object(name)
            if obj is None:
                continue
            obj.set_state(self, value)

        self._apply_button_states()
        return True

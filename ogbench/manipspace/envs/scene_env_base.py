import mujoco
import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ogbench.manipspace.envs.objects.base import SceneObject


class SceneEnvBase(ManipSpaceEnv):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        self._env_type = env_type
        self._permute_blocks = permute_blocks
        super().__init__(*args, **kwargs)
        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])


    def set_tasks(self):
        raise NotImplementedError

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        is_collection = self._mode in ("data_collection", "collection")

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
                self.compute_oracle_observation() if self._use_oracle_rep else self.compute_ob_info()
            )
            self._cur_goal_rendered = self.get_pixel_observation() if self._render_goal else None

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
        assert self._mode in ("data_collection", "collection")

        probs = self._get_task_probabilities()
        task_list, prob_list = [], []
        for obj in self.objects:
            if obj.name in probs:
                task_list.append(obj.name)
                prob_list.append(probs[obj.name])
        probs = np.array(prob_list, dtype=float)
        probs /= probs.sum()
        self._target_task = self.np_random.choice(task_list, p=probs)

        for obj in self.objects:
            if obj.name == self._target_task:
                obj.handle_target(self)
                break

        mujoco.mj_kinematics(self._model, self._data)
        if return_info:
            return self.compute_observation(), self.get_reset_info()

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

    def get_object(self, name, instance_id=0):
        """Find a SceneObject by name and optional instance_id."""
        for obj in self._objects:
            if obj.name == name and obj.instance_id == instance_id:
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

        if self._mode in ("data_collection", "collection"):
            ob_info["privileged_target_task"] = self._target_task
            for obj in self.objects:
                ob_info.update(obj.get_info_target(self))

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
        for obj in self.objects:
            obj.add_observation(self, ob, ob_info)
        return np.concatenate(ob)

    def compute_oracle_observation(self):
        ob_info = self.compute_ob_info()
        ob = [ob_info["proprio_joint_pos"]]
        for obj in self.objects:
            obj.add_oracle_obs(self, ob, ob_info)
        return np.concatenate(ob)

    def compute_reward(self):
        successes = [val for val, _ in self._compute_successes()]
        return float(sum(successes) - len(successes))

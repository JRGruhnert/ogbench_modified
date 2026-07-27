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
        self._configure_scene()

    def set_tasks(self):
        raise NotImplementedError

    def _configure_scene(self):
        raise NotImplementedError

    def initialize_episode(self):
        raise NotImplementedError

    def set_new_target(self, return_info=True, p_stack=0.5):
        raise NotImplementedError

    def add_objects(self, arena_mjcf):
        # Add objects to scene.
        for obj in self.objects:
            obj.load(arena_mjcf,self._desc_dir)

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
        return self._objects  # type: ignore

    def set_state(self, qpos, qvel):
        for obj in self.objects:
            obj.apply_lock(self, self._cur_button_states, self._button_locks)

        mujoco.mj_forward(self._model, self._data)
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
        for obj in self.objects:
            obj.post_step()
        successes = self._compute_successes()
        self._success = all(val for val, _ in successes)

    def add_object_info(self, ob_info: dict):
        for obj in self.objects:
            ob_info.update(obj.get_info(self))

        if self._mode == "data_collection":
            for obj in self.objects:
                ob_info.update(obj.get_target_info(self))


    def compute_observation(self):
        return self.get_pixel_observation()


    def compute_oracle_observation(self):
        ob_info = self.compute_ob_info()
        ob = []

        for obj in self.objects:
            obj.add_oracle_obs(self, ob, ob_info)

        return np.concatenate(ob)

    def compute_reward(self):
        successes = [val for val, _ in self._compute_successes()]
        return float(sum(successes) - len(successes))

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

import numpy as np
from dm_control import mjcf
class SceneObject:
    xml_file: str = ""
    name: str = ""
    scale: float = 1.0

    def __init__(self, instance_id: int, pos: np.ndarray, euler: np.ndarray):
        self.instance_id = instance_id
        self.pos = pos
        self.euler = euler

    @property
    def suffix(self):
        return f"{self.name}_{self.instance_id}"

    def load(self, arena_mjcf, desc_dir):
        self._mjcf = mjcf.from_path((desc_dir / self.xml_file).as_posix())
        arena_mjcf.include_copy(self._mjcf)


    def default_quaternion(self) -> np.ndarray:
        return np.array(lie.SO3.identity().wxyz.tolist())

    def post_compilation(self, env):
        pass

    def randomize(self, env):
        """Randomize this object's state (data_collection mode)."""
        pass


    def compute_success(self, env):
        """Return (success_bool, task_type_string) or None if not applicable."""
        return None

    def get_info(self, env):
        """Return dict of privileged_{name}_* key-value pairs."""
        return {}

    def get_target_info(self, env):
        """Return dict of privileged_target_{name}_* key-value pairs (data_collection)."""
        return {}

    def add_observation(self, env, ob, ob_info):
        """Append observation entries to the ob list."""
        pass

    def add_oracle_obs(self, env, ob, ob_info):
        """Append oracle observation entries to the ob list."""
        pass

    def get_task_probability(self, env):
        """Return probability for this object's task type (data_collection)."""
        return 1.0

    def handle_target(self, env):
        """Set a new random target for this object (data_collection)."""
        pass

    def get_target_from_task(self, task_info):
        """Extract this object's target value from a task dict."""
        return None

    def apply_lock(self, model):
        raise NotImplementedError

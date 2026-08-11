import numpy as np
from dm_control import mjcf

from ogbench.manipspace import lie

COLORS = dict(
    red=np.array([0.96, 0.26, 0.33, 1.0]),
    orange=np.array([1.0, 0.69, 0.21, 1.0]),
    yellow=np.array([0.76, 0.96, 0.04, 1.0]),
    green=np.array([0.06, 0.74, 0.21, 1.0]),
    blue=np.array([0.35, 0.55, 0.91, 1.0]),
    purple=np.array([0.61, 0.28, 0.82, 1.0]),
    magenta=np.array([0.82, 0.28, 0.61, 1.0]),
    lightred=np.array([0.99, 0.85, 0.86, 1.0]),
    lightorange=np.array([1.0, 0.94, 0.84, 1.0]),
    lightyellow=np.array([0.95, 0.99, 0.8, 1.0]),
    lightgreen=np.array([0.77, 0.95, 0.81, 1.0]),
    lightblue=np.array([0.86, 0.9, 0.98, 1.0]),
    lightpurple=np.array([0.91, 0.84, 0.96, 1.0]),
    lightmagenta=np.array([0.96, 0.84, 0.91, 1.0]),
    white=np.array([0.9, 0.9, 0.9, 1.0]),
    lightgray=np.array([0.7, 0.7, 0.7, 1.0]),
    gray=np.array([0.5, 0.5, 0.5, 1.0]),
    darkgray=np.array([0.3, 0.3, 0.3, 1.0]),
    black=np.array([0.1, 0.1, 0.1, 1.0]),
)


class SceneObject:
    xml_file: str = ""
    name: str = ""
    joint_name: str = None

    def __init__(self, id: int = 0, pos=None, euler=None):
        self.id = id
        self.pos = pos
        self.euler = euler
        self.name = f"{self.name}{id}"

    @property
    def _suf(self):
        return f"_{self.id}" if self.id > 0 else ""

    def _jname(self, base):
        return f"{base}{self._suf}"

    def load(self, arena_mjcf, desc_dir):
        self._mjcf = mjcf.from_path((desc_dir / self.xml_file).as_posix())
        if self.id > 0:
            self._rename_elements(self._mjcf, self.id)
        # Apply pos/euler to the root body if provided.
        if self.pos is not None or self.euler is not None:
            for body in self._mjcf.worldbody.find_all("body"):
                if self.pos is not None:
                    body.pos = self.pos
                if self.euler is not None:
                    body.euler = self.euler
                break  # Only the first body
        arena_mjcf.include_copy(self._mjcf)

    @staticmethod
    def _rename_elements(mjcf_model, suffix):
        # Rename all named elements to avoid duplicates when loading the same XML twice.
        TAGS = (
            "body",
            "joint",
            "site",
            "geom",
            "material",
            "texture",
            "mesh",
            "actuator",
            "sensor",
            "camera",
            "light",
            "equality",
            "tendon",
        )
        for tag in TAGS:
            for el in mjcf_model.find_all(tag):
                if el.name:
                    el.name = f"{el.name}_{suffix}"
        for el in mjcf_model.find_all("default"):
            if el.dclass:
                el.dclass = f"{el.dclass}_{suffix}"

    def default_quaternion(self) -> np.ndarray:
        return np.array(lie.SO3.identity().wxyz.tolist())

    def post_compilation(self, env):
        pass

    def randomize(self, env):
        pass

    def init_to_goal(self, env, task_info):
        pass

    def init_to_init(self, env, task_info):
        pass

    def compute_success(self, env):
        return None

    def get_info(self, env) -> dict:
        return {}

    def get_info_target(self, env) -> dict:
        return {}

    def get_task_probability(self, env) -> float:
        return 1.0

    def handle_target(self, env):
        pass

    def get_target_from_task(self, task_info):
        return None

    def apply_lock(self, model):
        pass

    def pre_step(self):
        pass

    def post_step(self, env):
        pass

    def apply_colors_and_locks(self, env):
        pass

    def get_state(self):
        """Current state of this object (used by other objects for lock rules)."""
        return 0

    def can_set_state(self, env, value):
        """Can this object be set to the given value right now?"""
        return not self._is_locked(env)

    def set_state(self, env, value):
        """Directly set this object's state (teleport)."""
        pass

    def _is_locked(self, env):
        """Locked when any rule object's state doesn't match its unlock state."""
        for name, unlock_state in getattr(self, "_lock_rule", {}).items():
            for obj in env._objects:
                if obj.name == name and obj.get_state() != unlock_state:
                    return True
        return False

    def health_check_and_colors(self, env, successes):
        pass

    def add_observation(self, env, ob: list, ob_info: dict):
        pass

    def add_oracle_obs(self, env, ob: list, ob_info: dict):
        pass

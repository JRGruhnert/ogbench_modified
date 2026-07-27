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
    count: int = 0

    def __init__(self, instance_id: int = 0, pos=None, euler=None):
        self.instance_id = instance_id
        self.pos = pos
        self.euler = euler

    def load(self, arena_mjcf, desc_dir):
        self._mjcf = mjcf.from_path((desc_dir / self.xml_file).as_posix())
        if self.instance_id > 0:
            self._rename_elements(self._mjcf, self.instance_id)
        arena_mjcf.include_copy(self._mjcf)

    @staticmethod
    def _rename_elements(mjcf_model, suffix):
        for tag in ("body", "joint", "site", "geom", "material"):
            for el in mjcf_model.find_all(tag):
                if el.name:
                    el.name = f"{el.name}_{suffix}"

    def default_quaternion(self) -> np.ndarray:
        return np.array(lie.SO3.identity().wxyz.tolist())

    # ---- lifecycle hooks (no default) ----
    def post_compilation(self, env):
        pass

    def randomize(self, env):
        pass

    def init_to_goal(self, env, task_info):
        pass

    def init_to_init(self, env, task_info):
        pass

    # ---- queries ----
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

    # ---- per-step ----
    def apply_lock(self, model):
        pass

    def pre_step(self):
        pass

    def post_step(self, env):
        pass

    def apply_colors_and_locks(self, env):
        pass

    def health_check_and_colors(self, env, successes):
        pass

    # ---- observation ----
    def add_observation(self, env, ob: list, ob_info: dict):
        pass

    def add_oracle_obs(self, env, ob: list, ob_info: dict):
        pass

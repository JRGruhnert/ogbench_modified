from ogbench.manipspace.envs.objects.base import SceneObject


class ShelfObject(SceneObject):
    """Shelf — a static fixture providing a goal site for cubes."""

    xml_file = "shelf.xml"
    var_prefix = "shelf"
    is_free_body = False
    has_target = False

    body_name = "shelf"
    site_name = "shelf_goal"

    default_pos = None
    default_euler = None

    def __init__(self, instance_id=0, pos=None, euler=None):
        super().__init__(instance_id, pos, euler)
        self.var_prefix = self._suffix("shelf")
        if instance_id > 0:
            self.site_name = self._suffix(self.site_name)
            self.body_name = self._suffix(self.body_name)
        if pos is None:
            self.pos = self.default_pos
        if euler is None:
            self.euler = self.default_euler

    # -- Backward-compat helpers -----------------------------------------

    @staticmethod
    def _suffix_static(name, i):
        return f"{name}_{i}"

    @classmethod
    def rename_in_xml(cls, mjcf_model, suffix):
        for element_type in ["body", "joint", "site", "geom", "material"]:
            for element in mjcf_model.find_all(element_type):
                if hasattr(element, "name") and element.name is not None:
                    try:
                        element.name = f"{element.name}_{suffix}"
                    except Exception:
                        pass

    def target_site_pos(self, env, name):
        return None

    # -- State helpers (static fixture) ----------------------------------

    def is_closed(self, env):
        return True

    def get_state(self, env):
        return 1

    def get_target_value(self, env):
        return None

    def set_target_in_model(self, env, val):
        pass

    # -- SceneObject interface -------------------------------------------

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

    def get_info(self, env):
        return {}

    def get_target_info(self, env):
        return {}

    def add_observation(self, env, ob, ob_info):
        pass

    def add_oracle_obs(self, env, ob, ob_info):
        pass

    def get_task_probability(self, env):
        available = sum(
            1
            for i in range(env.unwrapped._num_cubes)
            if not env.unwrapped._is_in_drawer(
                env._data.joint(f"object_joint_{i}").qpos[:3]
            )
        )
        return 1.0 if available > 0 else 0.0

    def handle_target(self, env):
        pass

    def get_target_from_task(self, task_info):
        return None

    def apply_lock(self, env, button_states, button_locks):
        pass

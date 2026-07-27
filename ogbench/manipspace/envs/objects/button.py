import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class ButtonSingleObject(SceneObject):
    """Single button — ``buttons_single.xml``."""

    xml_file = "buttons_single.xml"
    var_prefix = "button"
    is_free_body = False
    is_button = True
    has_target = True

    # Default placement
    default_pos = None
    default_euler = None

    def __init__(self):
        super().__init__()
        self.count = 1

    # -- Backward-compat helpers (still used by old env code) ------------

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

    @staticmethod
    def is_closed(env):
        return True

    @staticmethod
    def get_state(env):
        return 0

    @staticmethod
    def get_target_value(env):
        return None

    @staticmethod
    def set_target_in_model(env, val):
        pass

    # -- SceneObject interface -------------------------------------------

    def post_compilation(self, env):
        pass

    def randomize(self, env):
        for i in range(self.count):
            env._cur_button_states[i] = env.np_random.choice(2)

    def init_to_goal(self, env, task_info):
        pass

    def init_to_init(self, env, task_info):
        pass

    def compute_success(self, env):
        successes = [
            (env._cur_button_states[i] == env._target_button_states[i])
            for i in range(self.count)
        ]
        return (all(successes), "button")

    def get_info(self, env):
        info = {}
        for i in range(self.count):
            site_id = env._button_site_ids[i]
            info[f"privileged_button_{i}_state"] = (
                0 if env._cur_button_states[i] == 0 else 1
            )
            info[f"privileged_button_{i}_pos_full"] = env._data.site_xpos[
                site_id
            ].copy()
            info[f"privileged_button_{i}_pos"] = env._data.joint(
                f"buttonbox_joint_{i}"
            ).qpos.copy()
            info[f"privileged_button_{i}_vel"] = env._data.joint(
                f"buttonbox_joint_{i}"
            ).qvel.copy()
            info[f"privileged_button_{i}_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        return info

    def get_target_info(self, env):
        info = {}
        if env._mode == "data_collection":
            info["privileged_target_button"] = env._target_button
            info["privileged_target_button_state"] = env._target_button_states[
                env._target_button
            ]
            info["privileged_target_button_top_pos"] = env._data.site_xpos[
                env._button_site_ids[env._target_button]
            ].copy()
            info["privileged_target_button_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        return info

    def add_observation(self, env, ob, ob_info):
        for i in range(self.count):
            button_state = np.eye(2)[env._cur_button_states[i]]
            ob.extend(
                [
                    button_state,
                    ob_info[f"privileged_button_{i}_pos"] * 120.0,
                    ob_info[f"privileged_button_{i}_vel"],
                ]
            )

    def add_oracle_obs(self, env, ob, ob_info):
        ob.append(env._cur_button_states.astype(np.float64))

    def get_task_probability(self, env):
        return 1.0

    def handle_target(self, env):
        env._target_button = env.np_random.choice(self.count)
        env._target_button_states[env._target_button] = (
            env._cur_button_states[env._target_button] + 1
        ) % env._num_button_states

    def get_target_from_task(self, task_info):
        return None

    def apply_lock(self, env, button_states, button_locks):
        pass


class ButtonDoubleObject(ButtonSingleObject):
    """Two buttons — ``buttons.xml``."""

    xml_file = "buttons.xml"

    def __init__(self):
        super().__init__()
        self.count = 2


class ButtonTripleObject(ButtonSingleObject):
    """Three buttons — ``buttons_triple.xml``."""

    xml_file = "buttons_triple.xml"

    def __init__(self):
        super().__init__()
        self.count = 3

import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class FaucetObject(SceneObject):
    """Faucet — a rotary knob joint.

    Closed at qpos = -1.57, open at qpos = 1.57.
    """

    xml_file = "faucet.xml"
    var_prefix = "faucet"
    is_free_body = False
    is_joint_object = True
    has_target = True

    body_name = "faucet"
    joint_name = "faucet_knob"
    site_name = "faucet_handle_center"
    target_site_name = "faucet_handle_center_target"
    material_name = "faucet_handle"
    pos_range = (-1.57, 1.57)
    scaler = 4.0
    tolerance = 0.15

    default_pos = None
    default_euler = None

    def __init__(self, instance_id=0, pos=None, euler=None):
        super().__init__(instance_id, pos, euler)
        self.name = self._suffix("faucet")
        if instance_id > 0:
            self.joint_name = self._suffix(self.joint_name)
            self.site_name = self._suffix(self.site_name)
            self.target_site_name = self._suffix(self.target_site_name)
            if self.material_name:
                self.material_name = self._suffix(self.material_name)
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

    # -- State helpers ---------------------------------------------------

    def is_closed(self, env):
        """True if the faucet is closed (qpos <= -0.3)."""
        return bool(env._data.joint(self.joint_name).qpos[0] <= -0.3)

    def get_state(self, env):
        """1 = closed, 0 = open."""
        return 1 if self.is_closed(env) else 0

    # -- SceneObject interface -------------------------------------------

    def post_compilation(self, env):
        self._site_id = env._model.site(self.site_name).id
        self._target_site_id = env._model.site(self.target_site_name).id

    def randomize(self, env):
        lo, hi = self.pos_range
        env._data.joint(self.joint_name).qpos[0] = env.np_random.uniform(lo, hi)

    def init_to_goal(self, env, task_info):
        env._data.joint(self.joint_name).qpos[0] = task_info["goal"][
            f"{self.name}_pos"
        ]

    def init_to_init(self, env, task_info):
        lo, hi = self.pos_range
        val = task_info["init"][f"{self.name}_pos"]
        env._data.joint(self.joint_name).qpos[0] = float(
            np.clip(val + env.np_random.uniform(-0.01, 0.01), lo, hi)
        )

    def compute_success(self, env):
        cur = env._data.joint(self.joint_name).qpos[0]
        target = env._target_object_pos.get(self.name, 0)
        success = bool(np.abs(cur - target) <= self.tolerance)
        return (success, self.name)

    def get_info(self, env):
        from ogbench.manipspace import lie

        site_id = self._site_id
        return {
            f"privileged_{self.name}_pos": env._data.joint(
                self.joint_name
            ).qpos.copy(),
            f"privileged_{self.name}_vel": env._data.joint(
                self.joint_name
            ).qvel.copy(),
            f"privileged_{self.name}_handle_pos": env._data.site_xpos[
                site_id
            ].copy(),
            f"privileged_{self.name}_handle_state": self.get_state(env),
            f"privileged_{self.name}_handle_yaw": np.array(
                [
                    lie.SO3.from_matrix(
                        env._data.site_xmat[site_id].reshape(3, 3)
                    ).compute_yaw_radians()
                ]
            ),
            f"privileged_{self.name}_handle_quat": np.array(
                lie.SO3.from_matrix(
                    env._data.site_xmat[site_id].reshape(3, 3)
                ).wxyz.copy()
            ),
        }

    def get_target_info(self, env):
        return {
            f"privileged_target_{self.name}_pos": np.array(
                [env._target_object_pos.get(self.name, 0)]
            ),
            f"privileged_target_{self.name}_handle_pos": env._data.site_xpos[
                self._target_site_id
            ].copy(),
        }

    def add_observation(self, env, ob, ob_info):
        ob.extend(
            [
                ob_info[f"privileged_{self.name}_pos"] * self.scaler,
                ob_info[f"privileged_{self.name}_vel"],
            ]
        )

    def add_oracle_obs(self, env, ob, ob_info):
        ob.append(ob_info[f"privileged_{self.name}_pos"] * self.scaler)

    def get_task_probability(self, env):
        for btn_idx, jname in env._button_locks.items():
            if jname == self.joint_name:
                if env._cur_button_states[btn_idx] == 0:
                    return 0.25
        return 1.0

    def handle_target(self, env):
        target_val = self._get_target_value(env)
        env._target_object_pos[self.name] = target_val
        self._set_target(env, target_val)

    def get_target_from_task(self, task_info):
        return task_info.get(f"{self.name}_pos", None)

    def apply_lock(self, env, button_states, button_locks):
        for btn_idx, jname in button_locks.items():
            if jname == self.joint_name:
                if button_states[btn_idx] == 0:
                    env._model.joint(self.joint_name).damping[0] = 1e6
                else:
                    env._model.joint(self.joint_name).damping[0] = 2.0
                if self.material_name:
                    env._model.material(self.material_name).rgba = env._colors["white"]

    # -- Internal helpers ------------------------------------------------

    def _get_target_value(self, env):
        """Return the target joint value (invert: closed → open, open → closed)."""
        lo, hi = self.pos_range
        if self.is_closed(env):
            return hi  # closed → target open
        else:
            return lo  # open → target closed

    def _set_target(self, env, val):
        """Set target site via trig using the scale-aware handle radius."""
        radius = 0.175 * self.scale
        env._model.site(self.target_site_name).pos[0] = radius * np.sin(val)
        env._model.site(self.target_site_name).pos[1] = radius * (1.0 - np.cos(val))

    # -- Backward-compat aliases -----------------------------------------

    def get_target_value(self, env):
        return self._get_target_value(env)

    def set_target_in_model(self, env, val):
        self._set_target(env, val)

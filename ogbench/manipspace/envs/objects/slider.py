import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class SliderObject(SceneObject):
    """Custom prismatic object with a graspable handle.

    The handle slides along the local X axis. `pos_range` controls how far the
    prismatic joint can travel.
    """

    xml_file = "heca_slider.xml"
    name = "slider"
    joint_name = "slider_slide"
    site_name = "slider_handle_center"
    target_site_name = "slider_handle_center_target"
    scaler = 1.0
    tolerance = 0.02

    def __init__(self, id=0, pos=(0, 0, 0), euler=(0, 0, 0), locks=None, pos_range=(0, 0.2)):
        super().__init__(id, pos, euler)
        if id > 0:
            self.joint_name = f"{self.joint_name}_{id}"
            self.site_name = f"{self.site_name}_{id}"
            self.target_site_name = f"{self.target_site_name}_{id}"
        self._lock_rule = locks or []
        self._target_val = 0.0
        self.pos_range = pos_range

    def is_closed(self, env):
        lo, hi = self.pos_range
        return bool(env._data.joint(self.joint_name).qpos[0] <= (lo + hi) / 2)

    def does_lock(self, env, value):
        """Return True when the slider's current displacement is close to `value`."""
        disp = env._data.joint(self.joint_name).qpos[0]
        return bool(np.isclose(disp, value, atol=0.01))

    def post_compilation(self, env):
        self._site_id = env._model.site(self.site_name).id
        self._target_site_id = env._model.site(self.target_site_name).id

        # Apply the configured prismatic limits to the actual joint range.
        env._model.joint(self.joint_name).range[:] = self.pos_range

        # Resize the rail so it matches the configured travel range.
        lo, hi = self.pos_range
        center_x = (lo + hi) / 2
        half_len = (hi - lo) / 2 + 0.02
        for geom_name in ("slider_rail", "slider_rail_col"):
            gid = env._model.geom(self._jname(geom_name)).id
            env._model.geom_pos[gid][0] = center_x
            env._model.geom_size[gid][0] = half_len

    def randomize(self, env):
        lo, hi = self.pos_range
        env._data.joint(self.joint_name).qpos[0] = env.np_random.uniform(lo, hi)

    def init_to_goal(self, env, task_info):
        val = task_info["goal"][f"{self.name}_pos"]
        env._data.joint(self.joint_name).qpos[0] = val
        self._target_val = val
        self._set_site(env, val)

    def init_to_init(self, env, task_info):
        lo, hi = self.pos_range
        val = task_info["init"][f"{self.name}_pos"]
        env._data.joint(self.joint_name).qpos[0] = float(
            np.clip(val + env.np_random.uniform(-0.01, 0.01), lo, hi)
        )

    def compute_success(self, env):
        cur = env._data.joint(self.joint_name).qpos[0]
        return (bool(np.abs(cur - self._target_val) <= self.tolerance), self.name)

    def get_info(self, env):
        from ogbench.manipspace import lie

        sid = self._site_id
        quat = np.array(
            lie.SO3.from_matrix(env._data.site_xmat[sid].reshape(3, 3)).wxyz.copy()
        )
        return {
            f"heca_{self.name}_pos": env._data.site_xpos[sid].copy(),
            f"heca_{self.name}_rot": quat,
            f"heca_{self.name}_yaw": np.array(
                [
                    lie.SO3.from_matrix(
                        env._data.site_xmat[sid].reshape(3, 3)
                    ).compute_yaw_radians()
                ]
            ),
            f"heca_{self.name}_ste": np.array([0]),
            f"heca_{self.name}_ste_min": np.array([0]),
            f"heca_{self.name}_ste_max": np.array([0]),
            f"heca_{self.name}_sca": env._data.joint(self.joint_name).qpos.copy(),
            f"heca_{self.name}_sca_min": np.array([self.pos_range[0]]),
            f"heca_{self.name}_sca_max": np.array([self.pos_range[1]]),
        }

    def get_info_target(self, env):
        return {
            f"heca_target_{self.name}_sca": np.array([self._target_val]),
            f"heca_target_{self.name}_pos": env._data.site_xpos[
                self._target_site_id
            ].copy(),
        }

    def get_task_probability(self, env):
        if self._is_locked(env):
            return 0.0
        return 1.0

    def handle_target(self, env):
        lo, hi = self.pos_range
        cur = env._data.joint(self.joint_name).qpos[0]
        target = env.np_random.choice([lo, hi])
        # If the current position already satisfies the chosen goal, use the
        # opposite end so the task isn't already solved.
        if abs(cur - target) <= self.tolerance:
            target = hi if target == lo else lo
        self._target_val = target
        self._set_site(env, target)

    def set_state(self, env, value):
        env._data.joint(self.joint_name).qpos[0] = value
        self._target_val = value
        self._set_site(env, value)

    def get_target_from_task(self, task_info):
        return task_info.get(f"{self.name}_pos")

    def apply_lock(self, model):
        model.joint(self.joint_name).damping[0] = 2.0

    def apply_colors_and_locks(self, env):
        if self._is_locked(env):
            env._model.joint(self.joint_name).damping[0] = 1e6
        else:
            env._model.joint(self.joint_name).damping[0] = 2.0

    def add_observation(self, env, ob, ob_info):
        pass

    def add_oracle_obs(self, env, ob, ob_info):
        pass

    def _set_site(self, env, val):
        env._model.site(self.target_site_name).pos[0] = val

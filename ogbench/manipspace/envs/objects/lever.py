import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class LeverObject(SceneObject):
    xml_file = "heca_lever.xml"
    name = "lever"
    joint_name = "lever_joint"
    site_name = "lever_handle_center"
    target_site_name = "lever_handle_center_target"
    pos_range = (0, 0.942)
    scaler = 1.0
    tolerance = 0.15

    def __init__(self, id=0, pos=(0, 0, 0), euler=(0, 0, 0), lock_rule=None):
        super().__init__(id, pos, euler)
        if id > 0:
            self.joint_name = f"{self.joint_name}_{id}"
            self.site_name = f"{self.site_name}_{id}"
            self.target_site_name = f"{self.target_site_name}_{id}"
        self._lock_rule = lock_rule or {}
        self._target_val = 0.0

    def is_closed(self, env):
        return bool(env._data.joint(self.joint_name).qpos[0] <= 0.785)

    def post_compilation(self, env):
        self._site_id = env._model.site(self.site_name).id
        self._target_site_id = env._model.site(self.target_site_name).id
        self._body_id = env._model.body(self._jname("lever_link")).id

    def randomize(self, env):
        lo, hi = self.pos_range
        env._data.joint(self.joint_name).qpos[0] = env.np_random.uniform(lo, hi)

    def init_to_goal(self, env, task_info):
        env._data.joint(self.joint_name).qpos[0] = task_info["goal"][f"{self.name}_pos"]

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
            f"heca_{self.name}_ste": np.array([1 if self.is_closed(env) else 0]),
            f"heca_{self.name}_ste_min": np.array([0]),
            f"heca_{self.name}_ste_max": np.array([1]),
            f"heca_{self.name}_ang": env._data.joint(self.joint_name).qpos.copy(),
            f"heca_{self.name}_ang_min": np.array([self.pos_range[0]]),
            f"heca_{self.name}_ang_max": np.array([self.pos_range[1]]),
        }

    def get_info_target(self, env):
        return {
            f"heca_target_{self.name}_ang": np.array([self._target_val]),
            f"heca_target_{self.name}_pos": env._data.site_xpos[
                self._target_site_id
            ].copy(),
        }

    def get_task_probability(self, env):
        if self._is_locked(env):
            return 0.25
        return 1.0

    def handle_target(self, env):
        lo, hi = self.pos_range
        self._target_val = hi if self.is_closed(env) else lo

    def set_state(self, env, value):
        env._data.joint(self.joint_name).qpos[0] = value
        self._target_val = value

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

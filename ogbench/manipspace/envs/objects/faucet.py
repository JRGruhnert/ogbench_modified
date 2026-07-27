import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class FaucetObject(SceneObject):
    xml_file = "faucet.xml"
    name = "faucet"
    joint_name = "faucet_knob"
    site_name = "faucet_handle_center"
    target_site_name = "faucet_handle_center_target"
    pos_range = (-1.57, 1.57)
    scaler = 4.0
    tolerance = 0.15

    def __init__(self, instance_id=0, pos=(0.3, -0.2, 0.084), euler=None, locked_by=None, button=None):
        super().__init__(instance_id, pos, euler)
        if instance_id > 0:
            self.name = f"{self.name}_{instance_id}"
            self.joint_name = f"{self.joint_name}_{instance_id}"
            self.site_name = f"{self.site_name}_{instance_id}"
            self.target_site_name = f"{self.target_site_name}_{instance_id}"
        self._locked_by = locked_by
        self._button = button
        self._target_val = 0.0

    def is_closed(self, env):
        return bool(env._data.joint(self.joint_name).qpos[0] <= -0.3)

    # ---- lifecycle ----
    def post_compilation(self, env):
        self._site_id = env._model.site(self.site_name).id
        self._target_site_id = env._model.site(self.target_site_name).id

    def randomize(self, env):
        lo, hi = self.pos_range
        env._data.joint(self.joint_name).qpos[0] = env.np_random.uniform(lo, hi)

    def init_to_goal(self, env, task_info):
        val = task_info["goal"][f"{self.name}_pos"]
        env._data.joint(self.joint_name).qpos[0] = val
        self._set_site(env, val)

    def init_to_init(self, env, task_info):
        lo, hi = self.pos_range
        val = task_info["init"][f"{self.name}_pos"]
        env._data.joint(self.joint_name).qpos[0] = float(
            np.clip(val + env.np_random.uniform(-0.05, 0.05), lo, hi)
        )

    # ---- queries ----
    def compute_success(self, env):
        cur = env._data.joint(self.joint_name).qpos[0]
        return (bool(np.abs(cur - self._target_val) <= self.tolerance), self.name)

    def get_info(self, env):
        from ogbench.manipspace import lie

        sid = self._site_id
        quat = np.array(lie.SO3.from_matrix(env._data.site_xmat[sid].reshape(3, 3)).wxyz.copy())
        return {
            f"privileged_{self.name}_pos": env._data.joint(self.joint_name).qpos.copy(),
            f"privileged_{self.name}_vel": env._data.joint(self.joint_name).qvel.copy(),
            f"privileged_{self.name}_handle_pos": env._data.site_xpos[sid].copy(),
            f"privileged_{self.name}_handle_state": 1 if self.is_closed(env) else 0,
            f"privileged_{self.name}_handle_yaw": np.array([
                lie.SO3.from_matrix(env._data.site_xmat[sid].reshape(3, 3)).compute_yaw_radians()
            ]),
            f"privileged_{self.name}_handle_quat": quat,
            f"heca_{self.name}_{self.instance_id}_pos": env._data.site_xpos[sid].copy(),
            f"heca_{self.name}_{self.instance_id}_rot": quat,
            f"heca_{self.name}_{self.instance_id}_ste": 1 if self.is_closed(env) else 0,
        }

    def get_info_target(self, env):
        return {
            f"privileged_target_{self.name}_pos": np.array([self._target_val]),
            f"privileged_target_{self.name}_handle_pos": env._data.site_xpos[self._target_site_id].copy(),
        }

    def get_task_probability(self, env):
        if self._locked_by is not None and self._button is not None and not self._button.is_pressed(self._locked_by):
            return 0.25
        return 1.0

    def handle_target(self, env):
        lo, hi = self.pos_range
        self._target_val = hi if self.is_closed(env) else lo
        self._set_site(env, self._target_val)

    def get_target_from_task(self, task_info):
        return task_info.get(f"{self.name}_pos")

    # ---- per-step ----
    def apply_lock(self, model):
        model.joint(self.joint_name).damping[0] = 2.0

    def add_observation(self, env, ob, ob_info):
        ob.extend([
            ob_info[f"privileged_{self.name}_pos"] * self.scaler,
            ob_info[f"privileged_{self.name}_vel"],
        ])

    def add_oracle_obs(self, env, ob, ob_info):
        ob.append(ob_info[f"privileged_{self.name}_pos"] * self.scaler)

    # ---- helpers ----
    def _set_site(self, env, val):
        env._model.site(self.target_site_name).pos[0] = 0.105 * np.sin(val)
        env._model.site(self.target_site_name).pos[1] = 0.105 * (1.0 - np.cos(val))

import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject, COLORS


class ButtonObject(SceneObject):
    """A single button loading button.xml.  Use multiple instances with different id."""

    xml_file = "heca_button.xml"
    name = "button"

    def __init__(self, id=0, pos=None, euler=None):
        super().__init__(id, pos, euler)
        self._target_button_states = np.array([0])
        self._num_states = 2
        self._cur_state = np.array([0])
        self._prev_states = np.array([0])

    def get_state(self):
        return self._cur_state[0]

    # -- lifecycle ---------------------------------------------------------
    def post_compilation(self, env):
        self._site_ids = [env._model.site(self._jname("btntop_0")).id]
        self._geom_ids = [[env._model.geom(self._jname("btngeom_0")).id]]

    def randomize(self, env):
        self._cur_state[0] = env.np_random.choice(self._num_states)

    def init_to_goal(self, env, task_info):
        self._cur_state[0] = task_info["goal"][self.name]

    def init_to_init(self, env, task_info):
        self._cur_state[0] = task_info["init"][self.name]
        self._target_button_states[0] = task_info["goal"][self.name]

    # -- queries -----------------------------------------------------------
    def compute_success(self, env):
        ok = self._cur_state[0] == self._target_button_states[0]
        return (ok, self.name)

    def get_info(self, env):
        idx = self.id
        return {
            f"privileged_button_{idx}_state": 0 if self._cur_state[0] == 0 else 1,
            f"privileged_button_{idx}_pos_full": env._data.site_xpos[
                self._site_ids[0]
            ].copy(),
            f"privileged_button_{idx}_pos": env._data.joint(
                self._jname("buttonbox_joint_0")
            ).qpos.copy(),
            f"privileged_button_{idx}_vel": env._data.joint(
                self._jname("buttonbox_joint_0")
            ).qvel.copy(),
            f"privileged_button_{idx}_quat": self.default_quaternion(),
            f"heca_button_{idx}_pos_base": env._data.joint(
                self._jname("buttonbox_joint_0")
            ).xanchor.copy(),
            f"heca_button_{idx}_pos_ee": env._data.site_xpos[self._site_ids[0]].copy(),
            f"heca_button_{idx}_rot": np.array([1.0, 0.0, 0.0, 0.0]),
            f"heca_button_{idx}_ste": 0 if self._cur_state[0] == 0 else 1,
            "prev_button_states": self._prev_states.copy(),
            "button_states": self._cur_state.copy(),
        }

    def get_info_target(self, env):
        if env._target_task != self.name:
            return {}
        return {
            "privileged_target_button": self.id,
            "privileged_target_button_state": self._target_button_states[0],
            "privileged_target_button_top_pos": env._data.site_xpos[
                self._site_ids[0]
            ].copy(),
            "privileged_target_button_quat": self.default_quaternion(),
            f"heca_target_button_{self.id}_pos_ee": env._data.site_xpos[
                self._site_ids[0]
            ].copy(),
        }

    def get_task_probability(self, env):
        return 1.0

    def handle_target(self, env):
        self._target_button_states[0] = (self._cur_state[0] + 1) % self._num_states

    def set_state(self, env, value):
        self._cur_state[0] = value
        self._target_button_states[0] = value

    def get_target_from_task(self, task_info):
        return task_info.get(self.name)

    # -- per-step ----------------------------------------------------------
    def apply_colors_and_locks(self, env):
        for gid in self._geom_ids[0]:
            env._model.geom(gid).rgba = COLORS[
                "red" if self._cur_state[0] == 0 else "white"
            ]

    def post_step(self, env):
        idx = self.id
        prev = env._prev_ob_info[f"privileged_button_{idx}_pos"][0]
        cur = env._data.joint(self._jname("buttonbox_joint_0")).qpos.copy()[0]
        if prev > -0.02 and cur <= -0.02:
            self._cur_state[0] = (self._cur_state[0] + 1) % self._num_states

    # -- observation -------------------------------------------------------
    def add_observation(self, env, ob, ob_info):
        button_scaler = 120.0
        idx = self.id
        state = np.eye(self._num_states)[self._cur_state[0]]
        ob.extend(
            [
                state,
                ob_info[f"privileged_button_{idx}_pos"] * button_scaler,
                ob_info[f"privileged_button_{idx}_vel"],
            ]
        )

    def add_oracle_obs(self, env, ob, ob_info):
        ob.append(self._cur_state.astype(np.float64))

import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject, COLORS


class ButtonObject(SceneObject):
    xml_file = "heca_button.xml"
    name = "button"

    def __init__(self, id=0, pos=None, euler=None, num_states=2, lock_rule=None):
        super().__init__(id, pos, euler)
        self._target_button_states = np.array([0])
        self._num_states = num_states
        self._cur_state = np.array([0])
        self._prev_joint_pos = 0.0
        self._lock_rule = lock_rule or {}

    def get_state(self):
        return self._cur_state[0]

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

    def compute_success(self, env):
        ok = self._cur_state[0] == self._target_button_states[0]
        return (ok, self.name)

    def get_info(self, env):
        idx = self.id
        return {
            f"heca_button_{idx}_pos": env._data.site_xpos[self._site_ids[0]].copy(),
            f"heca_button_{idx}_rot": self.default_quaternion(),
            f"heca_button_{idx}_ste": np.array([int(self._cur_state[0])]),
            f"heca_button_{idx}_yaw": np.array([0.0]),
            f"heca_button_{idx}_ste_min": np.array([0]),
            f"heca_button_{idx}_ste_max": np.array([self._num_states - 1]),
        }

    def get_info_target(self, env):
        if env._target_task != self.name:
            return {}
        return {
            f"heca_target_button_{self.id}_ste": np.array(
                [self._target_button_states[0]]
            ),
            f"heca_target_button_{self.id}_pos": env._data.site_xpos[
                self._site_ids[0]
            ].copy(),
            f"heca_target_button_{self.id}_rot": self.default_quaternion(),
        }

    def handle_target(self, env):
        self._target_button_states[0] = (self._cur_state[0] + 1) % self._num_states

    def set_state(self, env, value):
        self._cur_state[0] = value
        self._target_button_states[0] = value

    def get_target_from_task(self, task_info):
        return task_info.get(self.name)

    def get_task_probability(self, env):
        if self._is_locked(env):
            return 0.25
        return 1.0

    def apply_colors_and_locks(self, env):
        for gid in self._geom_ids[0]:
            env._model.geom(gid).rgba = COLORS[
                "red" if self._cur_state[0] == 0 else "white"
            ]
        if self._is_locked(env):
            env._model.joint(self._jname("buttonbox_joint_0")).damping[0] = 1e6
        else:
            env._model.joint(self._jname("buttonbox_joint_0")).damping[0] = 2.0

    def post_step(self, env):
        cur = env._data.joint(self._jname("buttonbox_joint_0")).qpos.copy()[0]
        if self._prev_joint_pos > -0.02 and cur <= -0.02:
            self._cur_state[0] = (self._cur_state[0] + 1) % self._num_states
        self._prev_joint_pos = cur

    def add_observation(self, env, ob, ob_info):
        pass

    def add_oracle_obs(self, env, ob, ob_info):
        pass

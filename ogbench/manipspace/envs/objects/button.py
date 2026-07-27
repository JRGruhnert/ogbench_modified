import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject, COLORS


class _ButtonBase(SceneObject):
    """Common button logic — subclass sets count / xml_file."""
    name = "button"

    def __init__(self, instance_id=0, pos=None, euler=None, locks=None):
        super().__init__(instance_id, pos, euler)
        self._locks = locks or {}
        self._target_button = 0
        self._target_button_states = np.array([0] * self.count)
        self._num_states = 2
        self._cur_states = np.array([0] * self.count)
        self._prev_states = np.array([0] * self.count)

    def is_pressed(self, btn_idx):
        """Public query: is button btn_idx currently pressed?"""
        return self._cur_states[btn_idx] != 0

    def post_compilation(self, env):
        self._site_ids = [env._model.site(f"btntop_{i}").id for i in range(self.count)]
        self._geom_ids = [[env._model.geom(f"btngeom_{i}").id] for i in range(self.count)]

    def randomize(self, env):
        for i in range(self.count):
            self._cur_states[i] = env.np_random.choice(self._num_states)

    def init_to_goal(self, env, task_info):
        self._cur_states = task_info["goal"]["button_states"].copy()

    def init_to_init(self, env, task_info):
        self._cur_states = task_info["init"]["button_states"].copy()
        self._target_button_states = task_info["goal"]["button_states"].copy()

    def compute_success(self, env):
        s = [self._cur_states[i] == self._target_button_states[i] for i in range(self.count)]
        return (all(s), "button")

    def get_info(self, env):
        info = {}
        for i in range(self.count):
            info[f"privileged_button_{i}_state"] = 0 if self._cur_states[i] == 0 else 1
            info[f"privileged_button_{i}_pos_full"] = env._data.site_xpos[self._site_ids[i]].copy()
            info[f"privileged_button_{i}_pos"] = env._data.joint(f"buttonbox_joint_{i}").qpos.copy()
            info[f"privileged_button_{i}_vel"] = env._data.joint(f"buttonbox_joint_{i}").qvel.copy()
            info[f"privileged_button_{i}_quat"] = self.default_quaternion()
            info[f"heca_button_{i}_pos"] = env._data.site_xpos[self._site_ids[i]].copy()
            info[f"heca_button_{i}_rot"] = np.array([1.0, 0.0, 0.0, 0.0])
            info[f"heca_button_{i}_ste"] = 0 if self._cur_states[i] == 0 else 1
        info["prev_button_states"] = self._prev_states.copy()
        info["button_states"] = self._cur_states.copy()
        return info

    def get_info_target(self, env):
        tb = self._target_button
        return {
            "privileged_target_button": tb,
            "privileged_target_button_state": self._target_button_states[tb],
            "privileged_target_button_top_pos": env._data.site_xpos[self._site_ids[tb]].copy(),
            "privileged_target_button_quat": self.default_quaternion(),
        }

    def handle_target(self, env):
        self._target_button = env.np_random.choice(self.count)
        self._target_button_states[self._target_button] = (
            self._cur_states[self._target_button] + 1
        ) % self._num_states

    def get_task_probability(self, env):
        return 1.0

    def get_target_from_task(self, task_info):
        return task_info.get("button_states")

    def add_observation(self, env, ob, ob_info):
        button_scaler = 120.0
        for i in range(self.count):
            state = np.eye(self._num_states)[self._cur_states[i]]
            ob.extend([
                state,
                ob_info[f"privileged_button_{i}_pos"] * button_scaler,
                ob_info[f"privileged_button_{i}_vel"],
            ])

    def add_oracle_obs(self, env, ob, ob_info):
        ob.append(self._cur_states.astype(np.float64))

    def apply_colors_and_locks(self, env):
        for i in range(self.count):
            for gid in self._geom_ids[i]:
                env._model.geom(gid).rgba = COLORS["red" if self._cur_states[i] == 0 else "white"]
        for btn_idx, joint_name in self._locks.items():
            if self._cur_states[btn_idx] == 0:
                env._model.joint(joint_name).damping[0] = 1e6
            else:
                env._model.joint(joint_name).damping[0] = 2.0

    def post_step(self, env):
        """Detect button presses and update states."""
        for i in range(self.count):
            prev = env._prev_ob_info[f"privileged_button_{i}_pos"][0]
            cur = env._data.joint(f"buttonbox_joint_{i}").qpos.copy()[0]
            if prev > -0.02 and cur <= -0.02:
                self._cur_states[i] = (self._cur_states[i] + 1) % self._num_states


class ButtonSingleObject(_ButtonBase):
    xml_file = "buttons_single.xml"
    count = 1


class ButtonDoubleObject(_ButtonBase):
    xml_file = "buttons.xml"
    count = 2


class ButtonTripleObject(_ButtonBase):
    xml_file = "buttons_triple.xml"
    count = 3

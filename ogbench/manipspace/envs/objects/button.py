import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class ButtonObject(SceneObject):
    xml_file = "button.xml"
    name = "button"

    def __init__(self, instance_id: int, pos: np.ndarray, euler: np.ndarray, joints: list[str], materials: list[str],):
        super().__init__(instance_id, pos, euler)
        self.current_state = 0
        self.target_state = 0
        self.prev_state = 0
        self.joints = joints
        self.materials = materials
        self.site_id = None


    def target_site_pos(self, env, name):
        return None

    def post_compilation(self, env):
        pass

    def randomize(self, env):
        for i in range(self.count):
            env._cur_button_states[i] = env.np_random.choice(2)

    def compute_success(self, env):
        successes = [
            (env._cur_button_states[i] == env._target_button_states[i])
            for i in range(self.count)
        ]
        return (all(successes), "button")

    def get_info(self, data):
        info = {}
        info[f"heca_{self.suffix}_ste"] = self.current_state
        info[f"heca_{self.suffix}_rot"] = self.default_quaternion()
        info[f"heca_{self.suffix}_pos"] = data.site_xpos[
            self.site_id
        ].copy()
        return info

    def get_target_info(self, data):
        info = {}
        info[f"heca_{self.suffix}_ste_target"] = self.target_state
        info[f"heca_{self.suffix}_rot_target"] = self.default_quaternion()
        info[f"heca_{self.suffix}_pos_target"] = data.site_xpos[
            self.site_id
        ].copy()
        return info

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

    def apply_lock(self, model):
        if self.ste == 0:
            for label in self.joints:
                model.joint(label).damping[0] = 1e6
            for label in self.materials:
                model.material(label).rgba = self._colors["white"]
        else:
            for label in self.joints:
                model.joint(label).damping[0] = 2.0
            for label in self.materials:
                model.material(label).rgba = self._colors["white"]

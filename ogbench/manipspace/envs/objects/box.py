import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class BoxObject(SceneObject):
    """Static box/bin — acts as a container for cube placement."""
    xml_file = "heca_box_base.xml"
    name = "box"

    def __init__(self, id=0, pos=None, euler=None):
        super().__init__(id, pos, euler)
        self._lid = None
        self._cube = None

    def set_lid(self, lid):
        self._lid = lid

    def set_cube(self, cube):
        self._cube = cube

    def post_compilation(self, env):
        self._body_id = env._model.body(self._jname("box")).id

    def get_task_probability(self, env):
        return None  # passive container, not a target

    def _surface_pos(self, env):
        """Top surface center of the box bin (for placement)."""
        p = env._data.xpos[self._body_id].copy()
        p[2] += 0.06  # bin rim height (50% deeper)
        return p

    def contains(self, env, obj_pos):
        """Check if a 3D point is inside the box bin."""
        p = self._surface_pos(env)
        return (
            abs(obj_pos[0] - p[0]) < 0.06 and
            abs(obj_pos[1] - p[1]) < 0.06 and
            obj_pos[2] > p[2] - 0.01
        )

    def is_open(self, env):
        """Box is accessible from the top unless the lid covers it."""
        return not self._lid_covers(env)

    def _lid_covers(self, env):
        if self._lid is None:
            return False
        lid_pos = env._data.joint(self._lid.joint_name).qpos[:3]
        p = self._surface_pos(env)
        return (abs(lid_pos[0] - p[0]) < 0.06 and
                abs(lid_pos[1] - p[1]) < 0.06 and
                abs(lid_pos[2] - p[2]) < 0.02)

    def get_placement_pos(self, env):
        """Target position for placing a block inside the box."""
        p = self._surface_pos(env)
        p[:2] += env.np_random.uniform(-0.005, 0.005, size=2)
        return p

    def get_info(self, env):
        return {
            f"heca_{self.name}_pos_base": env._data.xpos[self._body_id].copy(),
            f"heca_{self.name}_pos_ee": self._surface_pos(env),
            f"heca_{self.name}_rot": np.array([1.0, 0.0, 0.0, 0.0]),
            f"heca_{self.name}_ste": 0 if self.is_open(env) else 1,
        }

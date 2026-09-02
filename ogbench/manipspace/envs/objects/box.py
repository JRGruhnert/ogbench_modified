import numpy as np

from ogbench.manipspace.envs.objects.base import SceneObject


class BoxObject(SceneObject):
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
        self._goal_site_id = env._model.site(self._jname("box_goal")).id

    def get_task_probability(self, env):
        return None  # passive container, not a target

    def _surface_pos(self, env):
        """Top surface of the box walls — the rim the lid rests on.

        The box walls are geoms at local z 0.027 with half-height 0.027, so
        their top (and the resting height of a lid's base) is at +0.054.
        """
        p = env._data.xpos[self._body_id].copy()
        p[2] += 0.054  # wall height (geoms at z 0.027, half-height 0.027).
        return p

    def contains(self, env, obj_pos):
        """Check if a 3D point is inside the box bin.

        The check is expressed in the frame of the `box_goal` site, so it
        follows the box's position/orientation.
        """
        xpos = env._data.site_xpos[self._goal_site_id]
        xmat = env._data.site_xmat[self._goal_site_id].reshape(3, 3)
        local = xmat.T @ (obj_pos - xpos)
        return (
            abs(local[0]) < 0.06
            and abs(local[1]) < 0.06
            and -0.02 < local[2] < 0.04
        )

    def is_open(self, env):
        """Box is accessible from the top unless the lid covers it."""
        return not self._lid_covers(env)

    def _lid_covers(self, env):
        if self._lid is None:
            return False
        lid_pos = env._data.joint(self._lid.joint_name).qpos[:3]
        p = self._surface_pos(env)
        return (
            abs(lid_pos[0] - p[0]) < 0.06
            and abs(lid_pos[1] - p[1]) < 0.06
            and abs(lid_pos[2] - p[2]) < 0.02
        )

    def get_placement_pos(self, env):
        """Target position for placing a block inside the box."""
        p = env._data.site_xpos[self._goal_site_id].copy()
        p[:2] += env.np_random.uniform(-0.005, 0.005, size=2)
        return p

    def get_info(self, env):
        return {
            f"heca_{self.name}_pos": env._data.xpos[self._body_id].copy(),
            f"heca_{self.name}_rot": self.default_quaternion(),
            f"heca_{self.name}_ste": np.array([0 if self.is_open(env) else 1]),
            f"heca_{self.name}_yaw": np.array([0.0]),
            f"heca_{self.name}_ste_min": np.array([0]),
            f"heca_{self.name}_ste_max": np.array([1]),
        }

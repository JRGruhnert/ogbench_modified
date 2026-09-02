import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.objects.base import SceneObject


class ShelfObject(SceneObject):
    xml_file = "heca_shelf.xml"
    name = "shelf"

    def __init__(self, id=0, pos=None, euler=None):
        super().__init__(id, pos, euler)
        self._cube = None

    def set_cube(self, cube):
        """Set the CubeObject reference (called after cube is created)."""
        self._cube = cube

    def post_compilation(self, env):
        self._body_id = env._model.body(self._jname("shelf")).id
        self._goal_site_id = env._model.site(self._jname("shelf_goal")).id

    def compute_success(self, env):
        return (True, "shelf")

    def get_task_probability(self, env):
        return None  # passive container, not a target

    def contains(self, env, obj_pos):
        """Check if a 3D point is on the shelf."""
        shelf_pos = env._data.site_xpos[self._goal_site_id]
        return (
            abs(obj_pos[0] - shelf_pos[0]) < 0.08
            and abs(obj_pos[1] - shelf_pos[1]) < 0.08
            and obj_pos[2] > shelf_pos[2] - 0.01
        )

    def is_open(self, env):
        """Shelf is always accessible from the top."""
        return True

    def _cube_target_pos(self, env):
        """World position where a cube's *center* rests on the shelf board.

        `shelf_goal` marks the board surface; the cube rests with its center
        one half-height (0.02, cube geom is 0.04 tall) above that surface.
        """
        p = env._data.site_xpos[self._goal_site_id].copy()
        p[2] += 0.02  # cube half-height.
        return p

    def get_placement_pos(self, env):
        """Target position for placing a block on the shelf."""
        p = self._cube_target_pos(env)
        p[:2] += env.np_random.uniform(-0.005, 0.005, size=2)
        return p

    def get_info(self, env):
        return {
            f"heca_{self.name}_pos": env._data.xpos[self._body_id].copy(),
            f"heca_{self.name}_rot": self.default_quaternion(),
            f"heca_{self.name}_ste": np.array([0]),
            f"heca_{self.name}_yaw": np.array([0.0]),
            f"heca_{self.name}_ste_min": np.array([0]),
            f"heca_{self.name}_ste_max": np.array([0]),
        }

    def init_to_goal(self, env, task_info):
        """Override cube mocap when shelf_block=1 in task goal."""
        if task_info["goal"].get("shelf_block", 0) == 1 and self._cube is not None:
            identity = lie.SO3.identity().wxyz.tolist()
            self._cube.set_all_mocap(env, self._cube_target_pos(env), identity)

    def handle_target(self, env):
        """Set cube mocap target to the shelf goal position (data-collection mode)."""
        if self._cube is not None:
            identity = lie.SO3.identity().wxyz.tolist()
            self._cube.set_all_mocap(env, self._cube_target_pos(env), identity)

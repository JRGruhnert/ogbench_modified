import numpy as np

from ogbench.manipspace.envs.objects import (
    DrawerObject,
    ShelfObject,
    ButtonObject,
    CubeObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv5(SceneEnvBase):
    """Drawer + 3 single buttons + shelf + 1 cube.

    Button 0 locks/unlocks the drawer. Buttons 1 and 2 are toggle-only.
    """

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        cube_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        shelf = ShelfObject()
        btn0 = ButtonObject()
        btn1 = ButtonObject(id=1)
        btn2 = ButtonObject(id=2)
        drawer = DrawerObject(
            pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14), lock_rule={"button_0": 1}
        )
        cube = CubeObject(sampling_bounds=cube_bounds, containers=[drawer, shelf])
        shelf.set_cube(cube)
        self._objects = [cube, btn0, btn1, btn2, drawer, shelf]
        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

    def set_tasks(self):
        self.task_infos = [
            dict(task_name="task1_open",
                 init=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_0=1, button_1=1, button_2=1, drawer_pos=0.0),
                 goal=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_0=1, button_1=1, button_2=1, drawer_pos=-0.16)),
            dict(task_name="task2_unlock_and_lock",
                 init=dict(block_xyzs=np.array([[0.35, -0.05, 0.02]]), button_0=0, button_1=0, button_2=0, drawer_pos=-0.16),
                 goal=dict(block_xyzs=np.array([[0.35, -0.05, 0.02]]), button_0=0, button_1=0, button_2=0, drawer_pos=0.0)),
            dict(task_name="task3_rearrange_medium",
                 init=dict(block_xyzs=np.array([[0.4, -0.05, 0.02]]), button_0=1, button_1=0, button_2=0, drawer_pos=0.0),
                 goal=dict(block_xyzs=np.array([[0.4, 0.15, 0.02]]), button_0=1, button_1=1, button_2=1, drawer_pos=-0.16)),
            dict(task_name="task4_put_in_drawer",
                 init=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_0=0, button_1=0, button_2=0, drawer_pos=0.0),
                 goal=dict(block_xyzs=np.array([[0.33, -0.356, 0.065986]]), button_0=1, button_1=0, button_2=0, drawer_pos=0.0)),
            dict(task_name="task5_rearrange_hard",
                 init=dict(block_xyzs=np.array([[0.35, 0.15, 0.02]]), button_0=0, button_1=0, button_2=0, drawer_pos=0.0),
                 goal=dict(block_xyzs=np.array([[0.35, 0.15, 0.02]]), button_0=0, button_1=0, button_2=0, drawer_pos=0.0, shelf_block=1)),
            dict(task_name="task6_put_on_shelf",
                 init=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_0=0, button_1=0, button_2=0, drawer_pos=0.0),
                 goal=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_0=1, button_1=1, button_2=1, drawer_pos=-0.16, shelf_block=1)),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2

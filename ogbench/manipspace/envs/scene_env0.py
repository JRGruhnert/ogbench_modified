import numpy as np

from ogbench.manipspace.envs.objects import (
    DrawerObject,
    WindowObject,
    ButtonObject,
    CubeObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv0(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        cube_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])

        drawer = DrawerObject(
            pos=(0.33, -0.42, 0.084),
            euler=(0, 0, 3.14),
            lock_rule={"button_0": 1},
        )
        objects = [
            CubeObject(sampling_bounds=cube_bounds, containers=[drawer]),
            ButtonObject(id=0),
            ButtonObject(id=1),
            drawer,
            WindowObject(pos=(0.3, 0.3, 0.202), lock_rule={"button_1": 1}),
        ]
        super().__init__(env_type, objects, permute_blocks=permute_blocks, *args, **kwargs)

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_0=1,
                    button_1=1,
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_0=1,
                    button_1=1,
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
            ),
            dict(
                task_name="task2_unlock_and_lock",
                init=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_0=0,
                    button_1=0,
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_0=0,
                    button_1=0,
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task3_rearrange_medium",
                init=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button_0=1,
                    button_1=0,
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02]]),
                    button_0=1,
                    button_1=1,
                    drawer_pos=-0.16,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task4_put_in_drawer",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_0=0,
                    button_1=0,
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_0=1,
                    button_1=0,
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task5_rearrange_hard",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button_0=0,
                    button_1=0,
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_0=0,
                    button_1=0,
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
            ),
        ]
        if self._reward_task_id == 0:
            self._reward_task_id = 2

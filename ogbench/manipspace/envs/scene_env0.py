import numpy as np

from ogbench.manipspace.envs.objects import (
    FaucetObject,
    ButtonObject,
    PegObject,
    LidObject,
    WindowObject,
    CubeObject,
    DrawerObject,
    LeverObject,
    ShelfObject,
    BoxObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv0(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        btn0 = ButtonObject(id=0, pos=(0.58, -0.05, 0.048))
        btn1 = ButtonObject(id=1, pos=(0.58, 0.05, 0.048))
        drawer0 = DrawerObject(
            id=0, pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14), locks=[{"button0": 1}]
        )
        window0 = WindowObject(id=0, pos=(0.3, 0.3, 0.202), locks=[{"button1": 1}])
        cube0 = CubeObject(id=0, sampling_bounds=bounds, containers=[drawer0])
        objects = [drawer0, cube0, btn0, btn1, window0]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button0=1,
                    button1=1,
                    drawer0_pos=0.0,
                    window0_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button0=1,
                    button1=1,
                    drawer0_pos=-0.16,
                    window0_pos=0.2,
                ),
            ),
            dict(
                task_name="task2_unlock_and_lock",
                init=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button0=0,
                    button1=0,
                    drawer0_pos=-0.16,
                    window0_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button0=0,
                    button1=0,
                    drawer0_pos=0.0,
                    window0_pos=0.0,
                ),
            ),
            dict(
                task_name="task3_rearrange_medium",
                init=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button0=1,
                    button1=0,
                    drawer0_pos=0.0,
                    window0_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02]]),
                    button0=1,
                    button1=1,
                    drawer0_pos=-0.16,
                    window0_pos=0.0,
                ),
            ),
            dict(
                task_name="task4_put_in_drawer",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button0=0,
                    button1=0,
                    drawer0_pos=0.0,
                    window0_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button0=1,
                    button1=0,
                    drawer0_pos=0.0,
                    window0_pos=0.0,
                ),
            ),
            dict(
                task_name="task5_rearrange_hard",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button0=0,
                    button1=0,
                    drawer0_pos=0.0,
                    window0_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button0=0,
                    button1=0,
                    drawer0_pos=0.0,
                    window0_pos=0.2,
                ),
            ),
        ]
        if self._reward_task_id == 0:
            self._reward_task_id = 2

import numpy as np
from ogbench.manipspace.envs.objects import (
    DrawerObject,
    DoorlockObject,
    LeverObject,
    ButtonSingleObject,
    CubeObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv3(SceneEnvBase):
    """Drawer + 1 button + doorlock + lever + 1 cube."""

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        cube_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        btn = ButtonSingleObject(locks={0: "drawer_slide"})
        drawer = DrawerObject(pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14), locked_by=0, button=btn)
        self._objects = [
            CubeObject(sampling_bounds=cube_bounds, containers=[drawer]),
            btn,
            drawer,
            DoorlockObject(button=btn),
            LeverObject(button=btn),
        ]
        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open_drawer_doorlock_lever",
                init=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_states=np.array([1]), drawer_pos=0.0, doorlock_pos=0.0, lever_pos=0.0),
                goal=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_states=np.array([1]), drawer_pos=-0.16, doorlock_pos=-2.0, lever_pos=1.57),
            ),
            dict(
                task_name="task2_close_drawer_doorlock_lever",
                init=dict(block_xyzs=np.array([[0.35, -0.05, 0.02]]), button_states=np.array([1]), drawer_pos=-0.16, doorlock_pos=-2.0, lever_pos=1.57),
                goal=dict(block_xyzs=np.array([[0.35, -0.05, 0.02]]), button_states=np.array([1]), drawer_pos=0.0, doorlock_pos=0.0, lever_pos=0.0),
            ),
            dict(
                task_name="task3_rearrange_doorlock",
                init=dict(block_xyzs=np.array([[0.4, -0.05, 0.02]]), button_states=np.array([1]), drawer_pos=0.0, doorlock_pos=0.0, lever_pos=0.0),
                goal=dict(block_xyzs=np.array([[0.4, 0.15, 0.02]]), button_states=np.array([1]), drawer_pos=-0.16, doorlock_pos=-2.0, lever_pos=0.0),
            ),
            dict(
                task_name="task4_put_in_drawer_doorlock",
                init=dict(block_xyzs=np.array([[0.35, 0.05, 0.02]]), button_states=np.array([1]), drawer_pos=0.0, doorlock_pos=0.0, lever_pos=0.0),
                goal=dict(block_xyzs=np.array([[0.33, -0.356, 0.065986]]), button_states=np.array([1]), drawer_pos=0.0, doorlock_pos=-2.0, lever_pos=0.0),
            ),
            dict(
                task_name="task5_rearrange_hard_doorlock_lever",
                init=dict(block_xyzs=np.array([[0.35, 0.15, 0.02]]), button_states=np.array([1]), drawer_pos=0.0, doorlock_pos=-2.0, lever_pos=0.0),
                goal=dict(block_xyzs=np.array([[0.33, -0.356, 0.065986]]), button_states=np.array([1]), drawer_pos=0.0, doorlock_pos=0.0, lever_pos=1.57),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

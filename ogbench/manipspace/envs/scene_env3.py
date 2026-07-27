import numpy as np
from ogbench.manipspace.envs.objects import (
    DrawerObject,
    DoorlockObject,
    LeverObject,
    ButtonObject,
    CubeObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv3(SceneEnvBase):
    """Drawer + 1 button + doorlock + lever + 1 cube."""

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        self._objects = [
            CubeObject(count=1),
            ButtonObject(),
            DrawerObject(pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14)),
            DoorlockObject(),
            LeverObject(),
        ]
        self._button_locks = {0: "drawer_slide"}

        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

    def _configure_scene(self):
        self._object_sampling_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        self._target_sampling_bounds = self._object_sampling_bounds
        self._drawer_center = np.array([0.33, -0.24, 0.066])
        self._cube_colors = np.array([self._colors["red"], self._colors["blue"]])
        self._cube_success_colors = np.array(
            [self._colors["lightred"], self._colors["lightblue"]]
        )

    # ------------------------------------------------------------------
    # Task definitions
    # ------------------------------------------------------------------
    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open_drawer_doorlock_lever",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=0.0,
                    doorlock_pos=0.0,
                    lever_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=-0.16,
                    doorlock_pos=-2.0,
                    lever_pos=1.57,
                ),
            ),
            dict(
                task_name="task2_close_drawer_doorlock_lever",
                init=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=-0.16,
                    doorlock_pos=-2.0,
                    lever_pos=1.57,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=0.0,
                    doorlock_pos=0.0,
                    lever_pos=0.0,
                ),
            ),
            dict(
                task_name="task3_rearrange_doorlock",
                init=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=0.0,
                    doorlock_pos=0.0,
                    lever_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=-0.16,
                    doorlock_pos=-2.0,
                    lever_pos=0.0,
                ),
            ),
            dict(
                task_name="task4_put_in_drawer_doorlock",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=0.0,
                    doorlock_pos=0.0,
                    lever_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([1]),
                    drawer_pos=0.0,
                    doorlock_pos=-2.0,
                    lever_pos=0.0,
                ),
            ),
            dict(
                task_name="task5_rearrange_hard_doorlock_lever",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button_states=np.array([1]),
                    drawer_pos=0.0,
                    doorlock_pos=-2.0,
                    lever_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([1]),
                    drawer_pos=0.0,
                    doorlock_pos=0.0,
                    lever_pos=1.57,
                ),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

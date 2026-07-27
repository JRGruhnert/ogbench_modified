import numpy as np
from ogbench.manipspace.envs.objects import DrawerObject, WindowObject, ButtonDoubleObject, CubeObject
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv1(SceneEnvBase):
    """Drawer + 2 buttons + window + 1 cube."""

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        self._objects = [
            CubeObject(count=1),
            ButtonDoubleObject(),
            DrawerObject(pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14)),
            WindowObject(pos=(0.3, 0.3, 0.202)),
        ]

        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

        self._button_locks = {0: "drawer_slide", 1: "window_slide"}
        self._drawer_center = np.array([0.33, -0.24, 0.066])

    # ------------------------------------------------------------------
    # Task definitions
    # ------------------------------------------------------------------
    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
            ),
            dict(
                task_name="task2_unlock_and_lock",
                init=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task3_rearrange_medium",
                init=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button_states=np.array([1, 0]),
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=-0.16,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task4_put_in_drawer",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([1, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                task_name="task5_rearrange_hard",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2

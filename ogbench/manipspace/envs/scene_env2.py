import numpy as np

from ogbench.manipspace.envs.objects import (
    FaucetObject,
    ButtonSingleObject,
    PegObject,
    LidObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv2(SceneEnvBase):
    """Faucet + 1 button + peg + lid. No cubes."""

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        peg_bounds = np.asarray([[0.25, -0.05], [0.45, 0.15]])
        lid_bounds = np.asarray([[0.45, 0.15], [0.6, 0.35]])
        btn = ButtonSingleObject(locks={0: "faucet_knob"})
        self._objects = [
            btn,
            FaucetObject(locked_by=0, button=btn),
            PegObject(sampling_bounds=peg_bounds),
            LidObject(sampling_bounds=lid_bounds),
        ]
        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open_faucet_and_move_peg",
                init=dict(peg_xyzs=np.array([[0.30, -0.02, 0.02]]), lid_xyzs=np.array([[0.3, 0.3, 0.039]]), button_states=np.array([0]), faucet_pos=-1.57),
                goal=dict(peg_xyzs=np.array([[0.45, 0.1, 0.02]]), lid_xyzs=np.array([[0.3, 0.3, 0.039]]), button_states=np.array([0]), faucet_pos=1.57),
            ),
            dict(
                task_name="task2_close_faucet_and_move_lid",
                init=dict(peg_xyzs=np.array([[0.35, 0.05, 0.02]]), lid_xyzs=np.array([[0.45, 0.3, 0.00]]), button_states=np.array([0]), faucet_pos=1.57),
                goal=dict(peg_xyzs=np.array([[0.35, 0.05, 0.02]]), lid_xyzs=np.array([[0.3, 0.3, 0.039]]), button_states=np.array([0]), faucet_pos=-1.57),
            ),
            dict(
                task_name="task3_move_peg_and_press_button",
                init=dict(peg_xyzs=np.array([[0.35, 0.05, 0.02]]), lid_xyzs=np.array([[0.3, 0.3, 0.039]]), button_states=np.array([0]), faucet_pos=-1.57),
                goal=dict(peg_xyzs=np.array([[0.42, 0.12, 0.02]]), lid_xyzs=np.array([[0.3, 0.3, 0.039]]), button_states=np.array([1]), faucet_pos=-1.57),
            ),
            dict(
                task_name="task4_rearrange_open_faucet",
                init=dict(peg_xyzs=np.array([[0.39, 0.07, 0.02]]), lid_xyzs=np.array([[0.5, 0.35, 0.00]]), button_states=np.array([0]), faucet_pos=-1.57),
                goal=dict(peg_xyzs=np.array([[0.35, 0.11, 0.02]]), lid_xyzs=np.array([[0.3, 0.3, 0.039]]), button_states=np.array([0]), faucet_pos=1.57),
            ),
            dict(
                task_name="task5_rearrange_close_faucet",
                init=dict(peg_xyzs=np.array([[0.35, 0.0, 0.02]]), lid_xyzs=np.array([[0.32, 0.33, 0.000]]), button_states=np.array([0]), faucet_pos=1.57),
                goal=dict(peg_xyzs=np.array([[0.42, 0.1, 0.02]]), lid_xyzs=np.array([[0.3, 0.3, 0.039]]), button_states=np.array([0]), faucet_pos=-1.57),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2

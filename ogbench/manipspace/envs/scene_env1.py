import numpy as np

from ogbench.manipspace.envs.objects import (
    FaucetObject,
    ButtonObject,
    PegObject,
    LidObject,
    WindowObject,
    CubeObject,
    DrawerObject,
    ShelfObject,
    BoxObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv1(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        bounds = np.asarray([[0.3, -0.1], [0.4, 0.15]])
        btn0 = ButtonObject(
            id=0,
            num_states=3,
            pos=(0.56, -0.1, 0.048),
        )
        faucet0 = FaucetObject(
            id=0,
            pos=(0.32, -0.2, 0.00),
            pos_range=(-1.3, 1.3),
            handle_radius=0.120,
            locks=[{"button0": 1}],
        )
        faucet1 = FaucetObject(
            id=1,
            pos=(0.56, 0.1, 0.00),
            euler=(0, 0, -1.57),
            locks=[{"faucet0": 1.3, "button0": 2}],
        )
        shelf0 = ShelfObject(
            id=0,
            pos=(0.3, 0.35, -0.1),
        )
        cube0 = CubeObject(
            id=0,
            sampling_bounds=bounds,
            containers=[shelf0],
        )
        objects = [shelf0, cube0, btn0, faucet0, faucet1]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

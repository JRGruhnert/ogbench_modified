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


class SceneEnv9(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        btn0 = ButtonObject(
            id=0,
            pos=(0.3, 0.3, 0.048),
            locks=[{"faucet1": -1.57}],
        )
        btn1 = ButtonObject(
            id=1,
            pos=(0.41, 0.32, 0.048),
            num_states=3,
            locks=[{"faucet1": 1.57}],
        )
        btn2 = ButtonObject(
            id=2,
            pos=(0.53, 0.28, 0.048),
            locks=[{"faucet0": 1.4}],
        )
        btn3 = ButtonObject(
            id=3,
            pos=(0.3, 0.15, 0.048),
            locks=[{"faucet0": -1.3}],
            num_states=3,
        )
        faucet0 = FaucetObject(
            id=0,
            pos=(0.32, -0.2, 0.00),
            pos_range=(-1.3, 1.4),
            handle_radius=0.1,
        )
        faucet1 = FaucetObject(
            id=1,
            pos=(0.56, 0.0, 0.00),
            euler=(0, 0, -1.57),
            locks=[{"button1": 2}],
        )
        objects = [btn0, btn1, btn2, btn3, faucet0, faucet1]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

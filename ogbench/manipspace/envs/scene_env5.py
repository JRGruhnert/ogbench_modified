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
    SliderObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv5(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        slider0 = SliderObject(id=0, pos=(0.3, 0.1, 0.0))
        btn0 = ButtonObject(id=0, pos=(0.3, 0.3, 0.048), locks=[{"button1": 1}])
        btn1 = ButtonObject(id=1, pos=(0.41, 0.32, 0.048))
        btn2 = ButtonObject(id=2, pos=(0.53, 0.28, 0.048), locks=[{"button0": 0}])
        faucet0 = FaucetObject(id=0, pos=(0.56, -0.1, 0.00), euler=(0, 0, -1.57))
        drawer0 = DrawerObject(
            id=0, pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14), locks=[{"button0": 1}]
        )
        objects = [slider0, faucet0, btn0, btn1, btn2, drawer0]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

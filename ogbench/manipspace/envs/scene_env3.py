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


class SceneEnv3(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        bounds = np.asarray([[0.3, -0.18], [0.42, 0.05]])
        btn0 = ButtonObject(id=0, pos=(0.56, -0.1, 0.048), locks=[{"button1": 1}])
        btn1 = ButtonObject(id=1, pos=(0.56, -0.0, 0.048))
        btn2 = ButtonObject(id=2, pos=(0.56, 0.1, 0.048), locks=[{"button0": 0}])
        shelf0 = ShelfObject(
            id=0, pos=(0.33, -0.34, -0.1), euler=(0, 0, 3.14)
        )  # pos=(0.3, 0.35, -0.04))
        cube0 = CubeObject(id=0, sampling_bounds=bounds, containers=[shelf0])
        drawer0 = DrawerObject(id=0, pos=(0.3, 0.4, 0.084), locks=[{"button0": 1}])
        objects = [shelf0, cube0, btn0, btn1, btn2, drawer0]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

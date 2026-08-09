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


class SceneEnv1(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        btn0 = ButtonObject(id=0, pos=(0.56, -0.1, 0.048), euler=(-1.57, 0, 0))
        faucet0 = FaucetObject(id=0, pos=(0.32, -0.2, 0.00))
        lever0 = LeverObject(id=0, pos=(0.56, 0.2, 0.00), euler=(0, 0, -1.57))
        shelf0 = ShelfObject(id=0, pos=(0.3, 0.35, -0.04))
        cube0 = CubeObject(id=0, sampling_bounds=bounds, containers=[shelf0])
        objects = [shelf0, cube0, btn0, faucet0, lever0]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

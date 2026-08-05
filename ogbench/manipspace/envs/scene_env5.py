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
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv5(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        cube_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])

        drawer0 = DrawerObject(
                id=0,
                pos=(0.33, -0.42, 0.084),
                euler=(0, 0, 3.14),
                locks={"button_0": 1}
            )
        cube0 = CubeObject(id= 0, sampling_bounds=cube_bounds, containers=[drawer0])
        btn0 = ButtonObject(id=0, pos=(0.58, -0.05, 0.048),euler=(-1.57, 0, 0))
        btn1 = ButtonObject(id=1, pos=(0.58, 0.05, 0.048),euler=(-1.57, 0, 0))
        window0 = WindowObject(id= 0, pos=(0.3, 0.3, 0.202), locks={"button_1": 1})
        objects = [drawer0, cube0, btn0, btn1, window0]
        super().__init__(env_type, objects, permute_blocks=permute_blocks, *args, **kwargs)

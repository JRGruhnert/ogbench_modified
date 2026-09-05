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


class SceneEnv7(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        cube0_bounds = np.asarray([[0.3, 0.1], [0.39, 0.3]])
        lid_bounds = np.asarray([[0.48, -0.1], [0.55, 0.15]])
        cube1_bounds = np.asarray([[0.30, -0.17], [0.38, 0.05]])
        shelf0 = ShelfObject(
            id=0,
            pos=(0.32, -0.35, -0.1),
            euler=(0, 0, 3.14),
        )
        box0 = BoxObject(
            id=0,
            pos=(0.5, 0.35, 0.0),
        )
        lid0 = LidObject(
            id=0,
            sampling_bounds=lid_bounds,
            containers=[box0],
        )

        cube0 = CubeObject(
            id=0,
            sampling_bounds=cube0_bounds,
            containers=[box0],
        )
        cube1 = CubeObject(
            id=1,
            sampling_bounds=cube1_bounds,
            containers=[shelf0],
        )
        box0.set_lid(lid0)
        box0.set_cube(cube0)
        objects = [shelf0, cube0, box0, lid0, cube1]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

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


class SceneEnv10(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        cube1_bounds = np.asarray([[0.3, 0.15], [0.42, 0.35]])
        cube0_bounds = np.asarray([[0.3, -0.35], [0.42, -0.15]])
        lid_bounds = np.asarray([[0.48, -0.2], [0.55, 0.2]])
        box0 = BoxObject(
            id=0,
            pos=(0.54, -0.35, 0.0),
            euler=(0, 0, 3.14),
        )
        box1 = BoxObject(
            id=1,
            pos=(0.54, 0.35, 0.0),
        )
        lid0 = LidObject(
            id=0,
            sampling_bounds=lid_bounds,
            containers=[box0, box1],
        )

        cube0 = CubeObject(
            id=0,
            sampling_bounds=cube0_bounds,
            containers=[box0],
        )
        cube1 = CubeObject(
            id=1,
            sampling_bounds=cube1_bounds,
            containers=[box1],
        )
        faucet0 = FaucetObject(
            id=0,
            pos=(0.28, 0.0, 0.00),
            # pos_range=(-1.45, 1.0),
            euler=(0, 0, 1.57),
        )
        box0.set_lid(lid0)
        box1.set_lid(lid0)
        box0.set_cube(cube0)
        box1.set_cube(cube1)
        objects = [box0, cube0, box1, lid0, cube1, faucet0]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

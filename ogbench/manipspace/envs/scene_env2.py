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


class SceneEnv2(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        peg_bounds = np.asarray([[0.36, -0.08], [0.42, 0.15]])
        lid_bounds = np.asarray([[0.44, 0.25], [0.54, 0.35]])
        btn0 = ButtonObject(id=0, pos=(0.24, -0.24, 0.048))
        btn1 = ButtonObject(id=1, pos=(0.36, -0.26, 0.048))
        faucet0 = FaucetObject(
            id=0,
            pos=(0.56, -0.1, 0.00),
            pos_range=(-1.45, 1.0),
            euler=(0, 0, -1.3),
            locks=[{"button0": 0, "button1": 1}],
        )
        box0 = BoxObject(id=0, pos=(0.3, 0.32, 0.0))
        lid0 = LidObject(id=0, sampling_bounds=lid_bounds, containers=[box0])
        peg0 = PegObject(id=0, sampling_bounds=peg_bounds)
        box0.set_lid(lid0)
        objects = [faucet0, peg0, btn0, btn1, box0, lid0]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

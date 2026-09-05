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
from ogbench.manipspace.envs.objects.slider import SliderObject
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv8(SceneEnvBase):
    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        cube0_bounds = np.asarray([[0.30, -0.17], [0.38, 0.05]])
        slider0 = SliderObject(
            id=0,
            pos=(0.30, -0.35, 0.0),
            locks=[{"button0": 0, "button0": 1}],
            pos_range=(0, 0.24),
        )
        slider1 = SliderObject(
            id=1,
            pos=(0.54, 0.18, 0.0),
            euler=(0, 0, -1.57),
            locks=[{"button0": 2, "button0": 0}],
            pos_range=(0, 0.22),
        )
        drawer0 = DrawerObject(
            id=0,
            pos=(0.32, 0.4, 0.084),
            locks=[{"slider0": 0, "slider1": 0.22}],
        )

        cube0 = CubeObject(
            id=0,
            sampling_bounds=cube0_bounds,
            containers=[drawer0],
        )
        btn0 = ButtonObject(
            id=0,
            num_states=3,
            pos=(0.56, -0.2, 0.048),
        )
        objects = [drawer0, cube0, slider0, slider1, btn0]
        super().__init__(
            env_type, objects, permute_blocks=permute_blocks, *args, **kwargs
        )

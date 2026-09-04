import time

from ogbench.manipspace.envs.scene_env0 import SceneEnv0
from ogbench.manipspace.envs.scene_env1 import SceneEnv1
from ogbench.manipspace.envs.scene_env10 import SceneEnv10
from ogbench.manipspace.envs.scene_env2 import SceneEnv2
from ogbench.manipspace.envs.scene_env3 import SceneEnv3
from ogbench.manipspace.envs.scene_env4 import SceneEnv4
from ogbench.manipspace.envs.scene_env5 import SceneEnv5
from ogbench.manipspace.envs.scene_env6 import SceneEnv6
from ogbench.manipspace.envs.scene_env7 import SceneEnv7
from ogbench.manipspace.envs.scene_env8 import SceneEnv8
from ogbench.manipspace.envs.scene_env9 import SceneEnv9
from ogbench.manipspace.oracles.plan.heca_faucet_plan import FaucetPlanOracle
from ogbench.manipspace.oracles.plan.heca_lid_plan import LidPlanOracle
from ogbench.manipspace.oracles.plan.heca_peg_plan import PegPlanOracle
from ogbench.manipspace.oracles.plan.heca_cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.heca_window_plan import WindowPlanOracle
from ogbench.manipspace.oracles.plan.heca_button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.heca_slider_plan import SliderPlanOracle
from ogbench.manipspace.oracles.plan.heca_drawer_plan import DrawerPlanOracle

env = SceneEnv0(env_type="scene", mode="randomized")
# env = SceneEnv1(env_type="scene", mode="randomized")
# env = SceneEnv2(env_type="scene", mode="randomized")
# env = SceneEnv3(env_type="scene", mode="randomized")
# env = SceneEnv4(env_type="scene", mode="randomized")
# env = SceneEnv5(env_type="scene", mode="randomized")
# env = SceneEnv6(env_type="scene", mode="randomized")
# env = SceneEnv7(env_type="scene", mode="randomized")
# env = SceneEnv8(env_type="scene", mode="randomized")
# env = SceneEnv9(env_type="scene", mode="randomized")
# env = SceneEnv10(env_type="scene", mode="randomized")
# oracle = FaucetPlanOracle(env)
# oracle = LidPlanOracle(env)
oracle = CubePlanOracle(0, env)
# oracle = WindowPlanOracle(0, env)
obs, info = env.reset()
oracle.reset(obs, info)
print(info)
env.launch_passive_viewer()

for step in range(2000):
    time.sleep(0.05)
    if oracle.done:
        obs, info = env.reset()
        oracle.reset(obs, info)

    action = oracle.select_action(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    env.sync_passive_viewer()

env.close_passive_viewer()
env.close()

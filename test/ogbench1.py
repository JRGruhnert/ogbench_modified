import time

from ogbench.manipspace.envs.scene_env1 import SceneEnv1
from ogbench.manipspace.oracles.plan.heca_faucet_plan import FaucetPlanOracle
from ogbench.manipspace.oracles.plan.heca_lid_plan import LidPlanOracle
from ogbench.manipspace.oracles.plan.heca_peg_plan import PegPlanOracle
from ogbench.manipspace.oracles.plan.heca_cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.heca_window_plan import WindowPlanOracle
env = SceneEnv1(env_type="scene", mode="task")
#oracle = FaucetPlanOracle(env)
#oracle = LidPlanOracle(env)
oracle = CubePlanOracle(0, env)
#oracle = WindowPlanOracle(0, env)
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

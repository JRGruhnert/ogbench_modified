import time

from ogbench.manipspace.envs.scene_env3 import SceneEnv3
from ogbench.manipspace.oracles.plan.faucet_plan import FaucetPlanOracle
from ogbench.manipspace.oracles.plan.lid_plan import LidPlanOracle
from ogbench.manipspace.oracles.plan.peg_plan import PegPlanOracle
env = SceneEnv3(env_type="scene", mode="task")
#oracle = FaucetPlanOracle(env)
#oracle = LidPlanOracle(env)
oracle = PegPlanOracle(env)

obs, info = env.reset()
oracle.reset(obs, info)

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

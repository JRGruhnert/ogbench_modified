import time

from ogbench.manipspace.envs.scene_env2 import SceneEnv2
from ogbench.manipspace.oracles.plan.faucet_plan import FaucetPlanOracle
from ogbench.manipspace.oracles.plan.lid_plan import LidPlanOracle
from ogbench.manipspace.oracles.plan.peg_plan import PegPlanOracle
env = SceneEnv2(env_type="scene", mode="task")
obs, info = env.reset()

# ── Set up oracle ───────────────────────────────────
#oracle = FaucetPlanOracle(env)
#oracle = LidPlanOracle(env)
oracle = PegPlanOracle(env)
oracle.reset(obs, info)

d = env._data
m = env._model

# ── Terminal debug ──────────────────────────────────
print("=" * 50)
print("JOINTS")
print("=" * 50)
for i in range(m.njnt):
    name = m.joint(i).name
    if name and name != "world":
        print(f"  {name:30s}  qpos={d.joint(name).qpos.round(4)}")

print("\n" + "=" * 50)
print("TASK TARGETS")
print("=" * 50)
print(f"  target_faucet_pos = {env._target_faucet_yaw:.3f}")
print(f"  task_name          = {env.cur_task_info['task_name']}")
print("\nFollowing oracle plan...\n")

# ── Interactive 3D viewer ───────────────────────────
env.launch_passive_viewer()

for step in range(2000):
    time.sleep(0.05)
    if oracle.done:
        print(f"\n✅ Oracle finished at step {step}")
        obs, info = env.reset()
        print(info)
        oracle.reset(obs, info)

    action = oracle.select_action(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    env.sync_passive_viewer()

    goal_reached = (
        abs(env._data.joint("faucet_knob").qpos[0] - env._target_faucet_yaw) <= 0.15
    )
    if step % 100 == 0:
        knob_angle = env._data.joint("faucet_knob").qpos[0]
        print(f"Step {step}: reward={reward:.3f}, knob={knob_angle:.2f}, goal_reached={goal_reached}")

env.close_passive_viewer()
env.close()

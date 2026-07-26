import time

from ogbench.manipspace.envs.scene_env2 import SceneEnv2
from ogbench.manipspace.oracles.plan.faucet_plan import FaucetPlanOracle

env = SceneEnv2(env_type="scene", mode="task")
obs, info = env.reset()

# ── Set up oracle ───────────────────────────────────
faucet_oracle = FaucetPlanOracle(env)
faucet_oracle.reset(obs, info)

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
print(f"  target_faucet_pos = {env._target_faucet_pos:.3f}")
print(f"  task_name          = {env.cur_task_info['task_name']}")
print("\nFollowing oracle plan...\n")

# ── Interactive 3D viewer ───────────────────────────
env.launch_passive_viewer()

for step in range(2000):
    time.sleep(0.02)
    if faucet_oracle.done:
        print(f"\n✅ Oracle finished at step {step}")
        obs, info = env.reset()
        faucet_oracle.reset(obs, info)

    action = faucet_oracle.select_action(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    env.sync_passive_viewer()

    if step % 100 == 0:
        knob_angle = env._data.joint("faucet_knob").qpos[0]
        print(f"Step {step}: reward={reward:.3f}, knob={knob_angle:.2f}")

env.close_passive_viewer()
env.close()

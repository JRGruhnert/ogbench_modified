from ogbench.manipspace.envs.scene_env1 import SceneEnv1

env = SceneEnv1(env_type="scene", mode="task")
obs, info = env.reset()
print(f"✅ Reset OK, obs shape: {obs.shape}")
print(f"   Task: {env.cur_task_info['task_name']}")
print(f"   Objects: {[o.name for o in env.objects]}")

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"✅ Step OK, reward: {reward:.3f}")

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Step {step}: reward={reward:.3f}, success={env._success}")

env.close()
print("Done.")

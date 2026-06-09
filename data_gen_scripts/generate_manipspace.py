import pathlib
from collections import defaultdict

import h5py
import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "cube-single-v0", "Environment name.")
flags.DEFINE_string("dataset_type", "play", "Dataset type.")
flags.DEFINE_string("save_path", None, "Save path.")
flags.DEFINE_float("noise", 0.1, "Action noise level.")
flags.DEFINE_float(
    "noise_smoothing", 0.5, "Action noise smoothing level for PlanOracle."
)
flags.DEFINE_float("min_norm", 0.4, "Minimum action norm for MarkovOracle.")
flags.DEFINE_float("p_random_action", 0, "Probability of selecting a random action.")
flags.DEFINE_integer("num_episodes", 1000, "Number of episodes.")
flags.DEFINE_integer("max_episode_steps", 1001, "Number of episodes.")
flags.DEFINE_integer("image_size", 256, "Image size for observations.")


def main(_):
    assert FLAGS.dataset_type in ["play", "noisy"]
    # 'play': Use a non-Markovian oracle (PlanOracle) that follows a pre-computed plan.
    # 'noisy': Use a Markovian, closed-loop oracle (MarkovOracle) with Gaussian action noise.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        max_episode_steps=FLAGS.max_episode_steps,
        width=FLAGS.image_size,
        height=FLAGS.image_size,
    )

    # Initialize oracles.
    oracle_type = "plan" if FLAGS.dataset_type == "play" else "markov"
    has_button_states = hasattr(env.unwrapped, "_cur_button_states")
    if "cube" in FLAGS.env_name:
        if oracle_type == "markov":
            agents = {
                "cube": CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                "cube": CubePlanOracle(
                    env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing
                ),
            }
    elif "scene" in FLAGS.env_name:
        if oracle_type == "markov":
            agents = {
                "cube": CubeMarkovOracle(
                    env=env, min_norm=FLAGS.min_norm, max_step=100
                ),
                "button": ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                "drawer": DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                "window": WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                "cube": CubePlanOracle(
                    env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing
                ),
                "button": ButtonPlanOracle(
                    env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing
                ),
                "drawer": DrawerPlanOracle(
                    env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing
                ),
                "window": WindowPlanOracle(
                    env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing
                ),
            }
    elif "puzzle" in FLAGS.env_name:
        if oracle_type == "markov":
            agents = {
                "button": ButtonMarkovOracle(
                    env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True
                ),
            }
        else:
            agents = {
                "button": ButtonPlanOracle(
                    env=env,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                    gripper_always_closed=True,
                ),
            }

    # Collect data.
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10

    print("Total steps:", total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace(".h5", "-val.h5")
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_file = h5py.File(train_path, "w")
    val_file = h5py.File(val_path, "w")
    train_datasets: dict = {}
    val_datasets: dict = {}
    written_train_steps = 0  # steps flushed to HDF5 so far
    written_val_steps = 0  # steps flushed to HDF5 so far

    def _flush_episode(ep_buf: dict, is_val: bool = False):
        """Append one episode's data to the HDF5 file."""
        nonlocal written_train_steps, written_val_steps
        file = val_file if is_val else train_file
        datasets = val_datasets if is_val else train_datasets
        written = written_val_steps if is_val else written_train_steps
        n = len(ep_buf["terminals"])
        for k, rows in ep_buf.items():
            arr = np.array(rows)
            if k not in datasets:
                maxshape = (None,) + arr.shape[1:]
                # print(k, arr.shape)
                datasets[k] = file.create_dataset(
                    k, data=arr, maxshape=maxshape, chunks=(1,) + arr.shape[1:]
                )
            else:
                ds = datasets[k]
                ds.resize(written + n, axis=0)
                ds[written : written + n] = arr
        if is_val:
            written_val_steps += n
        else:
            written_train_steps += n

    for ep_idx in trange(num_train_episodes + num_val_episodes):
        # Have an additional while loop to handle rare cases with undesirable states (for the Scene environment).
        episode_buffer: dict = defaultdict(list)
        while True:
            ob, info = env.reset()

            # Set the cube stacking probability for this episode.
            if "single" in FLAGS.env_name:
                p_stack = 0.0
            elif "double" in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif "triple" in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            elif "quadruple" in FLAGS.env_name:
                p_stack = np.random.uniform(0.1, 0.5)
            elif "octuple" in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.35)
            else:
                p_stack = 0.5

            if oracle_type == "markov":
                # Set the action noise level for this episode.
                xi = np.random.uniform(0, FLAGS.noise)

            agent = agents[info["privileged_target_task"]]
            agent.reset(ob, info)

            done = False
            step = 0
            ep_qpos = []

            while not done:
                if np.random.rand() < FLAGS.p_random_action:
                    # Sample a random action.
                    action = env.action_space.sample()
                else:
                    # Get an action from the oracle.
                    action = agent.select_action(ob, info)
                    action = np.array(action)
                    if oracle_type == "markov":
                        # Add Gaussian noise to the action.
                        action = action + np.random.normal(
                            0, [xi, xi, xi, xi * 3, xi * 10], action.shape
                        )
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if agent.done:
                    # Set a new task when the current task is done.
                    agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent = agents[agent_info["privileged_target_task"]]
                    agent.reset(agent_ob, agent_info)

                if isinstance(ob, dict):
                    for ob_key, ob_val in ob.items():
                        episode_buffer[ob_key].append(ob_val)
                else:
                    episode_buffer["observations"].append(ob)
                episode_buffer["actions"].append(action)
                episode_buffer["terminals"].append(done)

                for k, v in info.items():
                    if isinstance(v, np.ndarray):
                        episode_buffer[k].append(v)
                    elif np.isscalar(v) and not isinstance(v, (str, bytes)):
                        episode_buffer[k].append(np.array([v], dtype=np.float32))

                ep_qpos.append(info["prev_qpos"])

                ob = next_ob
                step += 1

            if "scene" in FLAGS.env_name:
                # Perform health check. We want to ensure that the cube is always visible unless it's in the drawer.
                # Otherwise, the test-time goal images may become ambiguous.
                is_healthy = True
                ep_qpos = np.array(ep_qpos)
                block_xyzs = ep_qpos[:, 14:17]
                if (block_xyzs[:, 1] >= 0.29).any():
                    is_healthy = False  # Block goes too far right.
                if (
                    (block_xyzs[:, 1] <= -0.3)
                    & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))
                ).any():
                    is_healthy = (
                        False  # Block goes too far left, without being in the drawer.
                    )

                if is_healthy:
                    _flush_episode(
                        episode_buffer, is_val=(ep_idx >= num_train_episodes)
                    )
                    break
                else:
                    # Remove the last episode and retry.
                    print("Unhealthy episode, retrying...", flush=True)
                    episode_buffer = defaultdict(list)
            else:
                _flush_episode(episode_buffer, is_val=(ep_idx >= num_train_episodes))
                break

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    train_file.close()
    val_file.close()
    print("Total steps:", total_steps)
    print(f"Train steps: {written_train_steps}  Val steps: {written_val_steps}")
    print("Done.")
    print(f"  train → {train_path}")
    print(f"  val   → {val_path}")


if __name__ == "__main__":
    app.run(main)

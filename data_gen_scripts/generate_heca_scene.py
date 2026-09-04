import os
import pathlib
import time
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
from ogbench.manipspace.oracles.plan.heca_button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.heca_cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.heca_drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.heca_window_plan import WindowPlanOracle
from ogbench.manipspace.oracles.plan.heca_faucet_plan import FaucetPlanOracle
from ogbench.manipspace.oracles.plan.heca_peg_plan import PegPlanOracle
from ogbench.manipspace.oracles.plan.heca_lid_plan import LidPlanOracle
from ogbench.manipspace.oracles.plan.heca_slider_plan import SliderPlanOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "cube-single-v0", "Environment name.")
flags.DEFINE_string("dataset_type", "play", "Dataset type.")
flags.DEFINE_string("save_path", None, "Save path.")
flags.DEFINE_float("noise", 0.01, "Action noise level.")
flags.DEFINE_float(
    "noise_smoothing", 0.5, "Action noise smoothing level for PlanOracle."
)
flags.DEFINE_float("min_norm", 0.4, "Minimum action norm for MarkovOracle.")
flags.DEFINE_float("p_random_action", 0, "Probability of selecting a random action.")
flags.DEFINE_integer("num_episodes", 250, "Number of episodes.")
flags.DEFINE_integer("max_episode_steps", 1201, "Number of episodes.")
flags.DEFINE_integer("image_size", 256, "Image size for observations.")
flags.DEFINE_bool("dry_run", False, "Run data collection without saving to file.")
flags.DEFINE_float(
    "viewer_delay", 0.0, "Delay between steps in dry_run mode (for visual inspection)."
)


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
    agents = {}
    for obj in env.unwrapped.objects:
        oid = obj.id
        name = obj.name  # e.g. "button_0", "drawer_0", "cube_0"
        if oracle_type == "markov":
            if name.startswith("cube"):
                agents[name] = CubeMarkovOracle(
                    env=env, min_norm=FLAGS.min_norm, max_step=100
                )
            elif name.startswith("button"):
                agents[name] = ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm)
            elif name.startswith("drawer"):
                agents[name] = DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm)
            elif name.startswith("window"):
                agents[name] = WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm)
        else:
            if name.startswith("cube"):
                agents[name] = CubePlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )
            elif name.startswith("button"):
                agents[name] = ButtonPlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )
            elif name.startswith("drawer"):
                agents[name] = DrawerPlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )
            elif name.startswith("window"):
                agents[name] = WindowPlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )
            elif name.startswith("faucet"):
                agents[name] = FaucetPlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )
            elif name.startswith("peg"):
                agents[name] = PegPlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )
            elif name.startswith("lid"):
                agents[name] = LidPlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )
            elif name.startswith("slider"):
                agents[name] = SliderPlanOracle(
                    env=env,
                    object_id=oid,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                )

    # Collect data.
    total_steps = 0
    num_episodes = FLAGS.num_episodes

    if not FLAGS.dry_run:
        save_path = f"data/{FLAGS.env_name}"
        if not save_path.endswith(".h5"):
            save_path += ".h5"
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_file = h5py.File(save_path, "w")
        datasets: dict = {}
        written_steps = 0

    def _flush_episode(ep_buf: dict):
        """Append one episode's data to the HDF5 file."""
        nonlocal written_steps
        n = len(ep_buf["terminals"])
        for k, rows in ep_buf.items():
            arr = np.array(rows)
            if arr.dtype.kind in ("U", "S"):
                # Store string values as variable-length UTF-8.
                str_arr = np.asarray(rows, dtype=h5py.string_dtype("utf-8"))
                if k not in datasets:
                    datasets[k] = save_file.create_dataset(
                        k,
                        data=str_arr,
                        maxshape=(None,),
                        chunks=True,
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    ds = datasets[k]
                    ds.resize(written_steps + n, axis=0)
                    ds[written_steps : written_steps + n] = str_arr
                continue
            if k not in datasets:
                maxshape = (None,) + arr.shape[1:]
                datasets[k] = save_file.create_dataset(
                    k,
                    data=arr,
                    maxshape=maxshape,
                    chunks=(1,) + arr.shape[1:],
                    compression="gzip",
                    compression_opts=4,
                )
            else:
                ds = datasets[k]
                ds.resize(written_steps + n, axis=0)
                ds[written_steps : written_steps + n] = arr
        written_steps += n

    for ep_idx in trange(num_episodes):
        episode_buffer: dict = defaultdict(list)
        while True:
            ob, info = env.reset()

            if FLAGS.dry_run:
                env.unwrapped.launch_passive_viewer()

            # Stacking only possible with multiple cubes — hardcoded for now.
            p_stack = (
                0.5
                if any(o.name.startswith("cube") for o in env.unwrapped.objects)
                else 0.0
            )

            if oracle_type == "markov":
                xi = np.random.uniform(0, FLAGS.noise)

            agent = agents[info["privileged_target_task"]]
            agent.reset(ob, info)

            done = False
            step = 0
            free_positions = []  # track free-body positions for health check
            oracle_start = True  # first step of the first oracle

            while not done:
                if np.random.rand() < FLAGS.p_random_action:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(ob, info)
                    action = np.array(action)
                    if oracle_type == "markov":
                        action = action + np.random.normal(
                            0, [xi, xi, xi, xi * 3, xi * 10], action.shape
                        )
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                # print(next_ob.keys())
                done = terminated or truncated

                current_is_start = oracle_start
                oracle_start = False

                if agent.done:
                    agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent = agents[agent_info["privileged_target_task"]]
                    agent.reset(agent_ob, agent_info)
                    # This step is the boundary where the previous oracle finished.
                    info["oracle_done"] = 1.0
                    # The next step will be the first step of the new oracle.
                    oracle_start = True
                    # Use the freshly computed (post-randomization) observation as
                    # the next step's `ob`, so the saved observation also reflects
                    # the new oracle's start state.
                    next_ob = agent_ob

                info["oracle_start"] = 1.0 if current_is_start else 0.0

                if isinstance(ob, dict):
                    for ob_key, ob_val in ob.items():
                        episode_buffer[ob_key].append(ob_val)
                else:
                    episode_buffer["observations"].append(ob)
                episode_buffer["actions"].append(action)
                episode_buffer["terminals"].append(done)

                for k, v in info.items():
                    if isinstance(v, np.ndarray):
                        episode_buffer[k].append(v.ravel())
                    elif isinstance(v, str):
                        episode_buffer[k].append(v)
                    elif np.isscalar(v) and not isinstance(v, (str, bytes)):
                        episode_buffer[k].append(np.array([v], dtype=np.float32))

                # Track free-body positions for health check.
                for obj in env.unwrapped.objects:
                    if hasattr(obj, "joint_name") and obj.name.startswith(
                        ("cube", "peg", "lid")
                    ):
                        free_positions.append(
                            env.unwrapped._data.joint(obj.joint_name).qpos[:3].copy()
                        )

                ob = next_ob
                step += 1
                if FLAGS.dry_run:
                    env.unwrapped.sync_passive_viewer()
                    if FLAGS.viewer_delay > 0:
                        time.sleep(FLAGS.viewer_delay)

            # Health check: discard episode if any free-body went out of workspace bounds.
            bounds = env.unwrapped._workspace_bounds
            is_healthy = True
            for pos in free_positions:
                if np.any(pos <= bounds[0] - 0.2) or np.any(pos >= bounds[1] + 0.2):
                    is_healthy = False
                    break

            if is_healthy:
                if FLAGS.dry_run:
                    env.unwrapped.close_passive_viewer()
                if not FLAGS.dry_run:
                    _flush_episode(episode_buffer)
                break
            else:
                if FLAGS.dry_run:
                    env.unwrapped.close_passive_viewer()
                print("Unhealthy episode, retrying...", flush=True)
                episode_buffer = defaultdict(list)

        total_steps += step

    if FLAGS.dry_run:
        env.unwrapped.close_passive_viewer()
    env.close()

    if not FLAGS.dry_run:
        save_file.close()
    print("Total steps:", total_steps)
    print("Done.")
    if not FLAGS.dry_run:
        print(f"  saved → {save_path}")

    # Force the process to exit cleanly. MuJoCo's GLFW/viewer can leave a
    # non-daemon thread running, which would otherwise keep the terminal alive
    # after `main` returns.
    os._exit(0)


if __name__ == "__main__":
    app.run(main)

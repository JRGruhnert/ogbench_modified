import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.objects.base import SceneObject, COLORS


class CubeObject(SceneObject):
    """Free-body cube."""
    xml_file = "cube.xml"
    name = "cube"
    count = 1

    def __init__(self, instance_id=0, pos=None, euler=None, sampling_bounds=None, target_bounds=None, containers=None):
        super().__init__(instance_id, pos, euler)
        self._sampling_bounds = sampling_bounds
        self._target_bounds = target_bounds or sampling_bounds
        self._colors = np.array([COLORS["red"], COLORS["blue"]])
        self._success_colors = np.array([COLORS["lightred"], COLORS["lightblue"]])
        self._containers = containers or []  # objects with contains/is_open/get_placement_pos
        self._target_block = 0

    def post_compilation(self, env):
        self._target_mocap_ids = [
            env._model.body(f"object_target_{i}").mocapid[0] for i in range(self.count)
        ]
        self._geom_ids = [
            [env._model.geom(g.full_identifier).id for g in gl]
            for gl in self._geom_lists
        ]
        self._target_geom_ids = [
            [env._model.geom(g.full_identifier).id for g in gl]
            for gl in self._target_geom_lists
        ]

    def randomize(self, env):
        bounds = self._sampling_bounds if self._sampling_bounds is not None else env._object_sampling_bounds
        for i in range(self.count):
            xy = env.np_random.uniform(*bounds)
            env._data.joint(f"object_joint_{i}").qpos[:3] = (*xy, 0.02)
            env._data.joint(f"object_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(
                env.np_random.uniform(0, 2 * np.pi)
            ).wxyz.tolist()

    def init_to_goal(self, env, task_info):
        xyzs = task_info["goal"]["block_xyzs"]
        identity = lie.SO3.identity().wxyz.tolist()
        for i in range(self.count):
            env._data.joint(f"object_joint_{i}").qpos[:3] = xyzs[i]
            env._data.joint(f"object_joint_{i}").qpos[3:] = identity
            env._data.mocap_pos[self._target_mocap_ids[i]] = xyzs[i]
            env._data.mocap_quat[self._target_mocap_ids[i]] = identity

    def init_to_init(self, env, task_info):
        xyzs = task_info["init"]["block_xyzs"].copy()
        goal_xyzs = task_info["goal"]["block_xyzs"].copy()
        identity = lie.SO3.identity().wxyz.tolist()
        for i in range(self.count):
            p = xyzs[i].copy()
            p[:2] += env.np_random.uniform(-0.01, 0.01, size=2)
            env._data.joint(f"object_joint_{i}").qpos[:3] = p
            env._data.joint(f"object_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(
                env.np_random.uniform(0, 2 * np.pi)
            ).wxyz.tolist()
            env._data.mocap_pos[self._target_mocap_ids[i]] = goal_xyzs[i]
            env._data.mocap_quat[self._target_mocap_ids[i]] = identity

    def _is_inside_any_container(self, env, i):
        """Check if cube i is inside any container."""
        pos = env._data.joint(f"object_joint_{i}").qpos[:3]
        for c in self._containers:
            if c.contains(env, pos):
                return True
        return False

    def compute_success(self, env):
        for i in range(self.count):
            obj_pos = env._data.joint(f"object_joint_{i}").qpos[:3]
            tar_pos = env._data.mocap_pos[self._target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) > 0.04:
                return (False, "cube")
        return (True, "cube")

    def get_info(self, env):
        info = {}
        for i in range(self.count):
            q = env._data.joint(f"object_joint_{i}")
            quat = q.qpos[3:].copy()
            info[f"privileged_block_{i}_pos"] = q.qpos[:3].copy()
            info[f"privileged_block_{i}_quat"] = quat
            info[f"privileged_block_{i}_yaw"] = np.array([
                lie.SO3(wxyz=quat).compute_yaw_radians()
            ])
            info[f"privileged_block_{i}_state"] = 1
            info[f"heca_cube_{i}_pos"] = q.qpos[:3].copy()
            info[f"heca_cube_{i}_rot"] = quat
            info[f"heca_cube_{i}_ste"] = 1
        return info

    def get_info_target(self, env):
        mid = self._target_mocap_ids[self._target_block]
        return {
            "privileged_target_block": self._target_block,
            "privileged_target_block_pos": env._data.mocap_pos[mid].copy(),
            "privileged_target_block_yaw": np.array([
                lie.SO3(wxyz=env._data.mocap_quat[mid]).compute_yaw_radians()
            ]),
            "privileged_target_block_quat": env._data.mocap_quat[mid].copy(),
        }

    def get_task_probability(self, env):
        """Cube is selectable if at least one cube is not trapped inside a container."""
        if not self._containers:
            return 1.0
        available = sum(
            1 for i in range(self.count)
            if not self._is_inside_any_container(env, i)
        )
        return 1.0 if available > 0 else 0.0

    def handle_target(self, env, p_stack=0.5):
        """Set a new random target for the cube."""
        available = [
            i for i in range(self.count)
            if not self._is_inside_any_container(env, i)
        ] if self._containers else list(range(self.count))

        block_xyzs = np.array([env._data.joint(f"object_joint_{i}").qpos[:3] for i in range(self.count)])
        top_blocks = []
        for i in range(self.count):
            if i not in available:
                continue
            for j in range(self.count):
                if i != j and block_xyzs[j][2] > block_xyzs[i][2] and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02:
                    break
            else:
                top_blocks.append(i)

        target_block = env.np_random.choice(top_blocks)
        self._target_block = target_block

        # Pick a random open container to put the block in.
        open_containers = [c for c in self._containers if c.is_open(env)]
        use_container = open_containers and env.np_random.uniform() < 0.3
        stack = len(top_blocks) >= 2 and env.np_random.uniform() < p_stack

        if use_container:
            container = open_containers[env.np_random.choice(len(open_containers))]
            tar_pos = container.get_placement_pos(env)
        elif stack:
            other = env.np_random.choice(list(set(top_blocks) - {target_block}))
            bp = env._data.joint(f"object_joint_{other}").qpos[:3]
            tar_pos = np.array([bp[0], bp[1], bp[2] + 0.04])
        else:
            bounds = self._target_bounds if self._target_bounds is not None else env._target_sampling_bounds
            xy = env.np_random.uniform(*bounds)
            tar_pos = (*xy, 0.02)

        yaw = env.np_random.uniform(0, 2 * np.pi)
        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        identity = lie.SO3.identity().wxyz.tolist()

        for i in range(self.count):
            if i == target_block:
                env._data.mocap_pos[self._target_mocap_ids[i]] = tar_pos
                env._data.mocap_quat[self._target_mocap_ids[i]] = tar_ori
            else:
                env._data.mocap_pos[self._target_mocap_ids[i]] = (0, 0, -0.3)
                env._data.mocap_quat[self._target_mocap_ids[i]] = identity

        for i in range(self.count):
            alpha = 0.2 if (env._visualize_info and i == target_block) else 0.0
            for gid in self._target_geom_ids[i]:
                env._model.geom(gid).rgba[3] = alpha

    def load(self, arena_mjcf, desc_dir):
        super().load(arena_mjcf, desc_dir)
        self._geom_lists = []
        self._target_geom_lists = []
        for i in range(self.count):
            self._geom_lists.append(self._mjcf.find("body", f"object_{i}").find_all("geom"))
            self._target_geom_lists.append(self._mjcf.find("body", f"object_target_{i}").find_all("geom"))

    def get_target_from_task(self, task_info):
        return task_info.get("block_xyzs")

    def set_all_mocap(self, env, pos, quat):
        """Set mocap for all cubes to the same position/quaternion."""
        for i in range(self.count):
            env._data.mocap_pos[self._target_mocap_ids[i]] = pos
            env._data.mocap_quat[self._target_mocap_ids[i]] = quat

    def add_observation(self, env, ob, ob_info):
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        for i in range(self.count):
            ob.extend([
                (ob_info[f"privileged_block_{i}_pos"] - c) * s,
                ob_info[f"privileged_block_{i}_quat"],
                np.cos(ob_info[f"privileged_block_{i}_yaw"]),
                np.sin(ob_info[f"privileged_block_{i}_yaw"]),
            ])

    def add_oracle_obs(self, env, ob, ob_info):
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        for i in range(self.count):
            ob.append((ob_info[f"privileged_block_{i}_pos"] - c) * s)

    def post_step(self, env):
        pass  # health check + colors handled below

    def health_check_and_colors(self, env, successes):
        """Called by scene_env.post_step — handles stability check and color updates."""
        if env._mode == "task":
            is_healthy = True
            for i in range(self.count):
                p = env._data.joint(f"object_joint_{i}").qpos[:3]
                if np.any(p <= env._workspace_bounds[0] - 0.2) or np.any(p >= env._workspace_bounds[1] + 0.2):
                    is_healthy = False
                    break
            if not is_healthy:
                bounds = self._sampling_bounds if self._sampling_bounds is not None else env._object_sampling_bounds
                for i in range(self.count):
                    xy = env.np_random.uniform(*bounds)
                    yaw = env.np_random.uniform(0, 2 * np.pi)
                    env._data.joint(f"object_joint_{i}").qpos[:3] = (*xy, 0.02)
                    env._data.joint(f"object_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                if self.count > 0:
                    env._data.joint("object_joint_0").qvel[:] = 0.0

        for i in range(self.count):
            if env._visualize_info and (env._mode == "task" or i == self._target_block):
                for gid in self._target_geom_ids[i]:
                    env._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._target_geom_ids[i]:
                    env._model.geom(gid).rgba[3] = 0.0

            cube_ok = False
            for val, name in successes:
                if name == "cube":
                    cube_ok = val
                    break
            if env._visualize_info and cube_ok:
                for gid in self._geom_ids[i]:
                    env._model.geom(gid).rgba[:3] = self._success_colors[i, :3]
            else:
                for gid in self._geom_ids[i]:
                    env._model.geom(gid).rgba[:3] = self._colors[i, :3]

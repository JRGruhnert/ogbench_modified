import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.objects.base import SceneObject, COLORS


class CubeObject(SceneObject):
    """Free-body cube."""
    xml_file = "heca_cube.xml"
    name = "cube"

    def __init__(self, id=0, pos=None, euler=None, sampling_bounds=None, target_bounds=None, containers=None):
        super().__init__(id, pos, euler)
        self._sampling_bounds = sampling_bounds
        self._target_bounds = target_bounds or sampling_bounds
        self._colors = np.array([COLORS["red"], COLORS["blue"]])
        self._success_colors = np.array([COLORS["lightred"], COLORS["lightblue"]])
        self._containers = containers or []
        self._target_block = 0

    def _j(self, base):
        """Shortcut for _jname with cube-specific base names."""
        return self._jname(base)

    @property
    def joint_name(self):
        return self._j("object_joint_0")

    def post_compilation(self, env):
        self._target_mocap_id = env._model.body(self._j("object_target_0")).mocapid[0]
        self._geom_ids = [env._model.geom(g.full_identifier).id for g in self._geom_list]
        self._target_geom_ids = [env._model.geom(g.full_identifier).id for g in self._target_geom_list]

    def load(self, arena_mjcf, desc_dir):
        super().load(arena_mjcf, desc_dir)
        self._geom_list = self._mjcf.find("body", self._jname("object_0")).find_all("geom")
        self._target_geom_list = self._mjcf.find("body", self._jname("object_target_0")).find_all("geom")

    def randomize(self, env):
        bounds = self._sampling_bounds if self._sampling_bounds is not None else env._object_sampling_bounds
        xy = env.np_random.uniform(*bounds)
        env._data.joint(self.joint_name).qpos[:3] = (*xy, 0.02)
        env._data.joint(self.joint_name).qpos[3:] = lie.SO3.from_z_radians(env.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
        # Initialize mocap target to current position.
        env._data.mocap_pos[self._target_mocap_id] = env._data.joint(self.joint_name).qpos[:3].copy()
        env._data.mocap_quat[self._target_mocap_id] = env._data.joint(self.joint_name).qpos[3:].copy()

    def init_to_goal(self, env, task_info):
        xyz = task_info["goal"]["block_xyzs"][0]
        identity = lie.SO3.identity().wxyz.tolist()
        env._data.joint(self.joint_name).qpos[:3] = xyz
        env._data.joint(self.joint_name).qpos[3:] = identity
        env._data.mocap_pos[self._target_mocap_id] = xyz
        env._data.mocap_quat[self._target_mocap_id] = identity

    def init_to_init(self, env, task_info):
        xyz = task_info["init"]["block_xyzs"][0].copy()
        goal_xyz = task_info["goal"]["block_xyzs"][0].copy()
        identity = lie.SO3.identity().wxyz.tolist()
        p = xyz.copy()
        p[:2] += env.np_random.uniform(-0.01, 0.01, size=2)
        env._data.joint(self.joint_name).qpos[:3] = p
        env._data.joint(self.joint_name).qpos[3:] = lie.SO3.from_z_radians(env.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
        env._data.mocap_pos[self._target_mocap_id] = goal_xyz
        env._data.mocap_quat[self._target_mocap_id] = identity

    def _is_inside_any_container(self, env):
        pos = env._data.joint(self.joint_name).qpos[:3]
        return any(c.contains(env, pos) for c in self._containers)

    def compute_success(self, env):
        obj_pos = env._data.joint(self.joint_name).qpos[:3]
        tar_pos = env._data.mocap_pos[self._target_mocap_id]
        return (bool(np.linalg.norm(obj_pos - tar_pos) <= 0.04), self.name)

    def get_info(self, env):
        q = env._data.joint(self.joint_name)
        quat = q.qpos[3:].copy()
        i = self.id
        return {
            f"privileged_block_{i}_pos": q.qpos[:3].copy(),
            f"privileged_block_{i}_quat": quat,
            f"privileged_block_{i}_yaw": np.array([lie.SO3(wxyz=quat).compute_yaw_radians()]),
            f"privileged_block_{i}_state": 1,
            f"heca_cube_{i}_pos_base": q.qpos[:3].copy(),
            f"heca_cube_{i}_pos_ee": q.qpos[:3].copy(),
            f"heca_cube_{i}_rot": quat,
            f"heca_cube_{i}_yaw": np.array([lie.SO3(wxyz=quat).compute_yaw_radians()]),
            f"heca_cube_{i}_ste": 0,
        }

    def get_info_target(self, env):
        mid = self._target_mocap_id
        i = self.id
        return {
            f"heca_target_cube_{i}": self._target_block,
            f"heca_target_cube_{i}_pos": env._data.mocap_pos[mid].copy(),
            f"heca_target_cube_{i}_yaw": np.array([lie.SO3(wxyz=env._data.mocap_quat[mid]).compute_yaw_radians()]),
        }

    def get_task_probability(self, env):
        if self._containers and self._is_inside_any_container(env):
            return 0.0
        return 1.0

    def handle_target(self, env, p_stack=0.5):
        available = not self._is_inside_any_container(env) if self._containers else True
        if not available:
            return

        open_containers = [c for c in self._containers if c.is_open(env)]
        use_container = open_containers and env.np_random.uniform() < 0.3

        if use_container:
            container = open_containers[env.np_random.choice(len(open_containers))]
            tar_pos = container.get_placement_pos(env)
        else:
            bounds = self._target_bounds if self._target_bounds is not None else env._target_sampling_bounds
            xy = env.np_random.uniform(*bounds)
            tar_pos = (*xy, 0.02)

        yaw = env.np_random.uniform(0, 2 * np.pi)
        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()

        env._data.mocap_pos[self._target_mocap_id] = tar_pos
        env._data.mocap_quat[self._target_mocap_id] = tar_ori

        alpha = 0.2 if env._visualize_info else 0.0
        for gid in self._target_geom_ids:
            env._model.geom(gid).rgba[3] = alpha

    def set_all_mocap(self, env, pos, quat):
        env._data.mocap_pos[self._target_mocap_id] = pos
        env._data.mocap_quat[self._target_mocap_id] = quat

    def set_state(self, env, value):
        """value is a (pos, quat) tuple."""
        pos, quat = value
        env._data.joint(self.joint_name).qpos[:3] = pos
        env._data.joint(self.joint_name).qpos[3:] = quat
        self.set_all_mocap(env, pos, quat)

    def get_target_from_task(self, task_info):
        return task_info.get("block_xyzs")

    def add_observation(self, env, ob, ob_info):
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        i = self.id
        ob.extend([
            (ob_info[f"privileged_block_{i}_pos"] - c) * s,
            ob_info[f"privileged_block_{i}_quat"],
            np.cos(ob_info[f"privileged_block_{i}_yaw"]),
            np.sin(ob_info[f"privileged_block_{i}_yaw"]),
        ])

    def add_oracle_obs(self, env, ob, ob_info):
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        i = self.id
        ob.append((ob_info[f"privileged_block_{i}_pos"] - c) * s)

    def health_check_and_colors(self, env, successes):
        if env._mode == "task":
            p = env._data.joint(self.joint_name).qpos[:3]
            if np.any(p <= env._workspace_bounds[0] - 0.2) or np.any(p >= env._workspace_bounds[1] + 0.2):
                bounds = self._sampling_bounds if self._sampling_bounds is not None else env._object_sampling_bounds
                xy = env.np_random.uniform(*bounds)
                env._data.joint(self.joint_name).qpos[:3] = (*xy, 0.02)
                env._data.joint(self.joint_name).qpos[3:] = lie.SO3.from_z_radians(env.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
                env._data.joint(self.joint_name).qvel[:] = 0.0

        if env._visualize_info:
            for gid in self._target_geom_ids:
                env._model.geom(gid).rgba[3] = 0.2

            cube_ok = any(val for val, name in successes if name == self.name)
            color = self._success_colors[0, :3] if cube_ok else self._colors[0, :3]
            for gid in self._geom_ids:
                env._model.geom(gid).rgba[:3] = color

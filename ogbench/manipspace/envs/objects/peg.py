import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.objects.base import SceneObject


class PegObject(SceneObject):
    xml_file = "heca_peg.xml"
    name = "peg"

    # Local offset from the free body origin (base) to the handle site, used to
    # recover the base position from `heca_peg0_pos` (which is the handle site).
    # The offset is lateral, so it rotates with the peg's orientation.
    handle_offset = np.array([0.0, -0.078, 0.0])

    def __init__(self, id=0, pos=None, euler=None, sampling_bounds=None):
        super().__init__(id, pos, euler)
        self._sampling_bounds = sampling_bounds

    @property
    def joint_name(self):
        return self._jname("peg_joint_0")

    def post_compilation(self, env):
        self._target_mocap_id = env._model.body(self._jname("peg_target_0")).mocapid[0]
        self._handle_site_id = env._model.site(self._jname("peg_handle_site_0")).id

    def randomize(self, env):
        bounds = (
            self._sampling_bounds
            if self._sampling_bounds is not None
            else env._object_sampling_bounds
        )
        xy = env.np_random.uniform(*bounds)
        env._data.joint(self.joint_name).qpos[:3] = (*xy, 0.02)
        env._data.joint(self.joint_name).qpos[3:] = lie.SO3.from_z_radians(env.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()
        # Init mocap to a random goal — handle_target overwrites when selected.
        bounds = self._sampling_bounds if self._sampling_bounds is not None else [[0.3, -0.3], [0.55, 0.3]]
        xy = env.np_random.uniform(*bounds)
        env._data.mocap_pos[self._target_mocap_id] = (*xy, 0.02)
        env._data.mocap_quat[self._target_mocap_id] = lie.SO3.from_z_radians(env.np_random.uniform(0, 2 * np.pi)).wxyz.tolist()

    def init_to_goal(self, env, task_info):
        xyz = task_info["goal"]["peg_xyzs"][0]
        identity = lie.SO3.identity().wxyz.tolist()
        env._data.joint(self.joint_name).qpos[:3] = xyz
        env._data.joint(self.joint_name).qpos[3:] = identity
        env._data.mocap_pos[self._target_mocap_id] = xyz
        env._data.mocap_quat[self._target_mocap_id] = identity

    def init_to_init(self, env, task_info):
        xyz = task_info["init"]["peg_xyzs"][0].copy()
        goal_xyz = task_info["goal"]["peg_xyzs"][0].copy()
        identity = lie.SO3.identity().wxyz.tolist()
        p = xyz.copy()
        p[:2] += env.np_random.uniform(-0.01, 0.01, size=2)
        env._data.joint(self.joint_name).qpos[:3] = p
        env._data.joint(self.joint_name).qpos[3:] = lie.SO3.from_z_radians(
            env.np_random.uniform(0, 2 * np.pi)
        ).wxyz.tolist()
        env._data.mocap_pos[self._target_mocap_id] = goal_xyz
        env._data.mocap_quat[self._target_mocap_id] = identity

    def compute_success(self, env):
        obj_pos = env._data.joint(self.joint_name).qpos[:3]
        tar_pos = env._data.mocap_pos[self._target_mocap_id]
        return (bool(np.linalg.norm(obj_pos - tar_pos) <= 0.04), self.name)

    def get_info(self, env):
        q = env._data.joint(self.joint_name)
        quat = q.qpos[3:].copy()
        i = self.id
        return {
            f"heca_{self.name}_pos_base": q.qpos[:3].copy(),
            f"heca_{self.name}_pos": env._data.site_xpos[
                env._model.site(self._jname("peg_handle_site_0")).id
            ].copy(),
            f"heca_{self.name}_rot": quat,
            f"heca_{self.name}_yaw": np.array([lie.SO3(wxyz=quat).compute_yaw_radians()]),
            f"heca_{self.name}_ste": np.array([0]),
            f"heca_{self.name}_ste_min": np.array([0]),
            f"heca_{self.name}_ste_max": np.array([0]),
            f"heca_{self.name}_loc": "base",
        }

    def get_info_target(self, env):
        mid = self._target_mocap_id
        i = self.id
        return {
            f"heca_target_{self.name}_pos": env._data.mocap_pos[mid].copy(),
            f"heca_target_{self.name}_yaw": np.array(
                [lie.SO3(wxyz=env._data.mocap_quat[mid]).compute_yaw_radians()]
            ),
        }

    def get_task_probability(self, env):
        return 1.0

    def handle_target(self, env):
        bounds = (
            self._sampling_bounds
            if self._sampling_bounds is not None
            else [[0.3, -0.3], [0.55, 0.3]]
        )
        for _ in range(40):
            xy = env.np_random.uniform(*bounds)
            yaw = env.np_random.uniform(0, 2 * np.pi)
            tar_pos = np.array([*xy, 0.02])
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            handle_pos = env._data.site_xpos[self._handle_site_id][:2]
            if np.linalg.norm(handle_pos - tar_pos[:2]) > 0.08:
                break
        env._data.mocap_pos[self._target_mocap_id] = tar_pos
        env._data.mocap_quat[self._target_mocap_id] = tar_ori

    def set_state(self, env, value):
        """value is a (pos, quat) tuple."""
        pos, quat = value
        env._data.joint(self.joint_name).qpos[:3] = pos
        env._data.joint(self.joint_name).qpos[3:] = quat
        env._data.mocap_pos[self._target_mocap_id] = pos
        env._data.mocap_quat[self._target_mocap_id] = quat

    def get_target_from_task(self, task_info):
        return task_info.get("peg_xyzs")

    def add_observation(self, env, ob, ob_info):
        pass

    def add_oracle_obs(self, env, ob, ob_info):
        pass

    def health_check_and_colors(self, env, successes):
        if env._mode == "task":
            p = env._data.joint(self.joint_name).qpos[:3]
            if np.any(p <= env._workspace_bounds[0] - 0.2) or np.any(
                p >= env._workspace_bounds[1] + 0.2
            ):
                bounds = (
                    self._sampling_bounds
                    if self._sampling_bounds is not None
                    else env._object_sampling_bounds
                )
                xy = env.np_random.uniform(*bounds)
                env._data.joint(self.joint_name).qpos[:3] = (*xy, 0.02)
                env._data.joint(self.joint_name).qpos[3:] = lie.SO3.from_z_radians(
                    env.np_random.uniform(0, 2 * np.pi)
                ).wxyz.tolist()
                env._data.joint(self.joint_name).qvel[:] = 0.0

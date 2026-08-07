import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.objects.base import SceneObject


class PegObject(SceneObject):
    """Assembly peg — a free-body ring to be placed on a peg fixture."""

    xml_file = "heca_peg.xml"
    name = "peg"

    def __init__(self, id=0, pos=None, euler=None, sampling_bounds=None):
        super().__init__(id, pos, euler)
        self._sampling_bounds = sampling_bounds

    @property
    def joint_name(self):
        return self._jname("peg_joint_0")

    def post_compilation(self, env):
        self._target_mocap_id = env._model.body(self._jname("peg_target_0")).mocapid[0]

    def randomize(self, env):
        bounds = (
            self._sampling_bounds
            if self._sampling_bounds is not None
            else env._object_sampling_bounds
        )
        for _ in range(20):
            xy = env.np_random.uniform(*bounds)
            pos = np.array([*xy, 0.02])
            # Keep distance from mocap target
            target = env._data.mocap_pos[self._target_mocap_id][:2]
            if np.linalg.norm(pos[:2] - target) > 0.08:
                break
        env._data.joint(self.joint_name).qpos[:3] = pos
        env._data.joint(self.joint_name).qpos[3:] = lie.SO3.from_z_radians(
            env.np_random.uniform(0, 2 * np.pi)
        ).wxyz.tolist()
        env._data.mocap_pos[self._target_mocap_id] = pos.copy()
        env._data.mocap_quat[self._target_mocap_id] = (
            env._data.joint(self.joint_name).qpos[3:].copy()
        )

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
            f"privileged_peg_{i}_pos": q.qpos[:3].copy(),
            f"privileged_peg_{i}_quat": quat,
            f"privileged_peg_{i}_yaw": np.array(
                [lie.SO3(wxyz=quat).compute_yaw_radians()]
            ),
            f"privileged_peg_{i}_state": 1,
            f"privileged_peg_{i}_handle_pos": env._data.site_xpos[
                env._model.site(self._jname("peg_handle_site_0")).id
            ].copy(),
            f"heca_peg_{i}_pos_base": q.qpos[:3].copy(),
            f"heca_peg_{i}_pos_ee": env._data.site_xpos[
                env._model.site(self._jname("peg_handle_site_0")).id
            ].copy(),
            f"heca_peg_{i}_rot": quat,
            f"heca_peg_{i}_yaw": np.array([lie.SO3(wxyz=quat).compute_yaw_radians()]),
            f"heca_peg_{i}_ste": 0,
        }

    def get_info_target(self, env):
        mid = self._target_mocap_id
        i = self.id
        return {
            f"heca_target_peg_{i}": 0,
            f"heca_target_peg_{i}_pos": env._data.mocap_pos[mid].copy(),
            f"heca_target_peg_{i}_yaw": np.array(
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
        xy = env.np_random.uniform(*bounds)
        yaw = env.np_random.uniform(0, 2 * np.pi)
        env._data.mocap_pos[self._target_mocap_id] = (*xy, 0.02)
        env._data.mocap_quat[self._target_mocap_id] = lie.SO3.from_z_radians(
            yaw
        ).wxyz.tolist()

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
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        i = self.id
        ob.extend(
            [
                (ob_info[f"privileged_peg_{i}_pos"] - c) * s,
                ob_info[f"privileged_peg_{i}_quat"],
                np.cos(ob_info[f"privileged_peg_{i}_yaw"]),
                np.sin(ob_info[f"privileged_peg_{i}_yaw"]),
            ]
        )

    def add_oracle_obs(self, env, ob, ob_info):
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        i = self.id
        ob.append((ob_info[f"privileged_peg_{i}_pos"] - c) * s)

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

import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.objects.base import SceneObject


class LidObject(SceneObject):
    """Box lid — a free-body lid for the box/bin."""
    xml_file = "box.xml"
    name = "lid"
    count = 1

    def __init__(self, instance_id=0, pos=None, euler=None, sampling_bounds=None):
        super().__init__(instance_id, pos, euler)
        self._sampling_bounds = sampling_bounds

    def post_compilation(self, env):
        self._target_mocap_ids = [
            env._model.body(f"box_lid_target_{i}").mocapid[0] for i in range(self.count)
        ]

    def randomize(self, env):
        bounds = self._sampling_bounds if self._sampling_bounds is not None else env._object_sampling_bounds
        for i in range(self.count):
            xy = env.np_random.uniform(*bounds)
            env._data.joint(f"box_lid_joint_{i}").qpos[:3] = (*xy, 0.02)
            env._data.joint(f"box_lid_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(
                env.np_random.uniform(0, 2 * np.pi)
            ).wxyz.tolist()

    def init_to_goal(self, env, task_info):
        xyzs = task_info["goal"]["lid_xyzs"]
        identity = lie.SO3.identity().wxyz.tolist()
        for i in range(self.count):
            env._data.joint(f"box_lid_joint_{i}").qpos[:3] = xyzs[i]
            env._data.joint(f"box_lid_joint_{i}").qpos[3:] = identity
            env._data.mocap_pos[self._target_mocap_ids[i]] = xyzs[i]
            env._data.mocap_quat[self._target_mocap_ids[i]] = identity

    def init_to_init(self, env, task_info):
        xyzs = task_info["init"]["lid_xyzs"].copy()
        goal_xyzs = task_info["goal"]["lid_xyzs"].copy()
        identity = lie.SO3.identity().wxyz.tolist()
        for i in range(self.count):
            p = xyzs[i].copy()
            p[:2] += env.np_random.uniform(-0.01, 0.01, size=2)
            env._data.joint(f"box_lid_joint_{i}").qpos[:3] = p
            env._data.joint(f"box_lid_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(
                env.np_random.uniform(0, 2 * np.pi)
            ).wxyz.tolist()
            env._data.mocap_pos[self._target_mocap_ids[i]] = goal_xyzs[i]
            env._data.mocap_quat[self._target_mocap_ids[i]] = identity

    def compute_success(self, env):
        for i in range(self.count):
            obj_pos = env._data.joint(f"box_lid_joint_{i}").qpos[:3]
            tar_pos = env._data.mocap_pos[self._target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) > 0.04:
                return (False, "lid")
        return (True, "lid")

    def get_info(self, env):
        info = {}
        for i in range(self.count):
            q = env._data.joint(f"box_lid_joint_{i}")
            quat = q.qpos[3:].copy()
            info[f"privileged_lid_{i}_pos"] = q.qpos[:3].copy()
            info[f"privileged_lid_{i}_quat"] = quat
            info[f"privileged_lid_{i}_yaw"] = np.array([
                lie.SO3(wxyz=quat).compute_yaw_radians()
            ])
            info[f"privileged_lid_{i}_state"] = 1
            info[f"privileged_lid_{i}_handle_pos"] = env._data.site_xpos[
                env._model.site(f"box_lid_handle_center_{i}").id
            ].copy()
            info[f"heca_lid_{i}_pos"] = q.qpos[:3].copy()
            info[f"heca_lid_{i}_rot"] = quat
            info[f"heca_lid_{i}_ste"] = 1
        return info

    def get_info_target(self, env):
        mid = self._target_mocap_ids[0]
        return {
            "privileged_target_lid": 0,
            "privileged_target_lid_pos": env._data.mocap_pos[mid].copy(),
            "privileged_target_lid_yaw": np.array([
                lie.SO3(wxyz=env._data.mocap_quat[mid]).compute_yaw_radians()
            ]),
            "privileged_target_lid_quat": env._data.mocap_quat[mid].copy(),
        }

    def get_task_probability(self, env):
        return 1.0

    def handle_target(self, env):
        xy = env.np_random.uniform(*self._sampling_bounds if self._sampling_bounds is not None else [[0.3, -0.3], [0.55, 0.3]])
        yaw = env.np_random.uniform(0, 2 * np.pi)
        env._data.mocap_pos[self._target_mocap_ids[0]] = (*xy, 0.02)
        env._data.mocap_quat[self._target_mocap_ids[0]] = lie.SO3.from_z_radians(yaw).wxyz.tolist()

    def get_target_from_task(self, task_info):
        return task_info.get("lid_xyzs")

    def add_observation(self, env, ob, ob_info):
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        for i in range(self.count):
            ob.extend([
                (ob_info[f"privileged_lid_{i}_pos"] - c) * s,
                ob_info[f"privileged_lid_{i}_quat"],
                np.cos(ob_info[f"privileged_lid_{i}_yaw"]),
                np.sin(ob_info[f"privileged_lid_{i}_yaw"]),
            ])

    def add_oracle_obs(self, env, ob, ob_info):
        c = np.array([0.425, 0.0, 0.0])
        s = 10.0
        for i in range(self.count):
            ob.append((ob_info[f"privileged_lid_{i}_pos"] - c) * s)

    def health_check_and_colors(self, env, successes):
        if env._mode == "task":
            is_healthy = True
            for i in range(self.count):
                p = env._data.joint(f"box_lid_joint_{i}").qpos[:3]
                if np.any(p <= env._workspace_bounds[0] - 0.2) or np.any(p >= env._workspace_bounds[1] + 0.2):
                    is_healthy = False
                    break
            if not is_healthy:
                bounds = self._sampling_bounds if self._sampling_bounds is not None else env._object_sampling_bounds
                for i in range(self.count):
                    xy = env.np_random.uniform(*bounds)
                    env._data.joint(f"box_lid_joint_{i}").qpos[:3] = (*xy, 0.02)
                    env._data.joint(f"box_lid_joint_{i}").qpos[3:] = lie.SO3.from_z_radians(
                        env.np_random.uniform(0, 2 * np.pi)
                    ).wxyz.tolist()
                    env._data.joint(f"box_lid_joint_{i}").qvel[:] = 0.0

import mujoco
import numpy as np
from ogbench.manipspace import lie
from ogbench.manipspace.envs.objects import FaucetObject, ButtonSingleObject
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv2(SceneEnvBase):
    """Scene with faucet + 1 button + peg + lid.  No cubes."""

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        self._objects = [
            ButtonSingleObject(),
            FaucetObject(),
        ]
        self._button_locks = {0: "faucet_knob"}

        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

        # Scene-specific extents.
        self._num_pegs = 1
        self._num_lids = 1

        # Separate bounds so peg and lid don't overlap on spawn.
        self._peg_sampling_bounds = np.asarray([[0.25, -0.05], [0.45, 0.15]])
        self._lid_sampling_bounds = np.asarray([[0.45, 0.15], [0.6, 0.35]])

        # Target indices (always 0 — one peg / one lid).
        self._target_peg = 0
        self._target_lid = 0

    # ==================================================================
    # Task definitions
    # ==================================================================
    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open_faucet_and_move_peg",
                init=dict(
                    peg_xyzs=np.array([[0.30, -0.02, 0.02]]),
                    lid_xyzs=np.array([[0.3, 0.3, 0.039]]),
                    button_states=np.array([0]),
                    faucet_pos=-1.57,
                ),
                goal=dict(
                    peg_xyzs=np.array([[0.45, 0.1, 0.02]]),
                    lid_xyzs=np.array([[0.3, 0.3, 0.039]]),
                    button_states=np.array([0]),
                    faucet_pos=1.57,
                ),
            ),
            dict(
                task_name="task2_close_faucet_and_move_lid",
                init=dict(
                    peg_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    lid_xyzs=np.array([[0.45, 0.3, 0.00]]),
                    button_states=np.array([0]),
                    faucet_pos=1.57,
                ),
                goal=dict(
                    peg_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    lid_xyzs=np.array([[0.3, 0.3, 0.039]]),
                    button_states=np.array([0]),
                    faucet_pos=-1.57,
                ),
            ),
            dict(
                task_name="task3_move_peg_and_press_button",
                init=dict(
                    peg_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    lid_xyzs=np.array([[0.3, 0.3, 0.039]]),
                    button_states=np.array([0]),
                    faucet_pos=-1.57,
                ),
                goal=dict(
                    peg_xyzs=np.array([[0.42, 0.12, 0.02]]),
                    lid_xyzs=np.array([[0.3, 0.3, 0.039]]),
                    button_states=np.array([1]),
                    faucet_pos=-1.57,
                ),
            ),
            dict(
                task_name="task4_rearrange_open_faucet",
                init=dict(
                    peg_xyzs=np.array([[0.39, 0.07, 0.02]]),
                    lid_xyzs=np.array([[0.5, 0.35, 0.00]]),
                    button_states=np.array([0]),
                    faucet_pos=-1.57,
                ),
                goal=dict(
                    peg_xyzs=np.array([[0.35, 0.11, 0.02]]),
                    lid_xyzs=np.array([[0.3, 0.3, 0.039]]),
                    button_states=np.array([0]),
                    faucet_pos=1.57,
                ),
            ),
            dict(
                task_name="task5_rearrange_close_faucet",
                init=dict(
                    peg_xyzs=np.array([[0.35, 0.0, 0.02]]),
                    lid_xyzs=np.array([[0.32, 0.33, 0.000]]),
                    button_states=np.array([0]),
                    faucet_pos=1.57,
                ),
                goal=dict(
                    peg_xyzs=np.array([[0.42, 0.1, 0.02]]),
                    lid_xyzs=np.array([[0.3, 0.3, 0.039]]),
                    button_states=np.array([0]),
                    faucet_pos=-1.57,
                ),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

    # ==================================================================
    # Hooks: add peg & lid XML objects
    # ==================================================================
    def _add_specific_objects(self, arena_mjcf):
        from dm_control import mjcf

        peg_mjcf = mjcf.from_path(
            (self._desc_dir / "assembly_peg.xml").as_posix()
        )
        arena_mjcf.include_copy(peg_mjcf)
        box_mjcf = mjcf.from_path((self._desc_dir / "box.xml").as_posix())
        arena_mjcf.include_copy(box_mjcf)

        # Peg geoms.
        self._peg_geoms_list = []
        for i in range(self._num_pegs):
            self._peg_geoms_list.append(
                peg_mjcf.find("body", f"peg_{i}").find_all("geom")
            )
        self._peg_target_geoms_list = []
        for i in range(self._num_pegs):
            self._peg_target_geoms_list.append(
                peg_mjcf.find("body", f"peg_target_{i}").find_all("geom")
            )

        # Lid geoms (from box.xml).
        self._lid_geoms_list = []
        for i in range(self._num_lids):
            self._lid_geoms_list.append(
                box_mjcf.find("body", f"box_lid_{i}").find_all("geom")
            )
        self._lid_target_geoms_list = []
        for i in range(self._num_lids):
            self._lid_target_geoms_list.append(
                box_mjcf.find("body", f"box_lid_target_{i}").find_all("geom")
            )

    # ==================================================================
    # Hooks: resolve peg & lid IDs after model compilation
    # ==================================================================
    def _post_compilation_specific(self):
        # Peg geom IDs — use body lookup because peg geoms have auto-generated names.
        peg_body_id = self._model.body("peg_0").id
        self._peg_geom_ids_list = [
            [
                i
                for i in range(self._model.ngeom)
                if self._model.geom_bodyid[i] == peg_body_id
            ]
        ]
        self._peg_target_mocap_ids = [
            self._model.body(f"peg_target_{i}").mocapid[0]
            for i in range(self._num_pegs)
        ]
        peg_target_body_id = self._model.body("peg_target_0").id
        self._peg_target_geom_ids_list = [
            [
                i
                for i in range(self._model.ngeom)
                if self._model.geom_bodyid[i] == peg_target_body_id
            ]
        ]

        # Lid geom IDs — use body lookup because lid geoms have auto-generated names.
        lid_body_id = self._model.body("box_lid_0").id
        self._lid_geom_ids_list = [
            [
                i
                for i in range(self._model.ngeom)
                if self._model.geom_bodyid[i] == lid_body_id
            ]
        ]
        self._lid_target_mocap_ids = [
            self._model.body(f"box_lid_target_{i}").mocapid[0]
            for i in range(self._num_lids)
        ]
        lid_target_body_id = self._model.body("box_lid_target_0").id
        self._lid_target_geom_ids_list = [
            [
                i
                for i in range(self._model.ngeom)
                if self._model.geom_bodyid[i] == lid_target_body_id
            ]
        ]

        # Peg / lid site IDs.
        self._peg_center_site_ids = [
            self._model.site(f"peg_center_{i}").id
            for i in range(self._num_pegs)
        ]
        self._peg_handle_site_ids = [
            self._model.site(f"peg_handle_site_{i}").id
            for i in range(self._num_pegs)
        ]
        self._lid_center_site_ids = [
            self._model.site(f"box_lid_center_{i}").id
            for i in range(self._num_lids)
        ]
        self._lid_handle_site_ids = [
            self._model.site(f"box_lid_handle_center_{i}").id
            for i in range(self._num_lids)
        ]

    # ==================================================================
    # Hooks: specific task fields
    # ==================================================================
    def _get_specific_task_fields(self):
        return {
            "peg_xyzs": np.zeros((self._num_pegs, 3)),
            "lid_xyzs": np.zeros((self._num_lids, 3)),
        }

    # ==================================================================
    # Hooks: randomize pegs & lids (data-collection mode)
    # ==================================================================
    def _randomize_specific_objects(self):
        for i in range(self._num_pegs):
            xy = self.np_random.uniform(*self._peg_sampling_bounds)
            obj_pos = (*xy, 0.02)
            yaw = self.np_random.uniform(0, 2 * np.pi)
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.joint(f"peg_joint_{i}").qpos[:3] = obj_pos
            self._data.joint(f"peg_joint_{i}").qpos[3:] = obj_ori

        for i in range(self._num_lids):
            xy = self.np_random.uniform(*self._lid_sampling_bounds)
            obj_pos = (*xy, 0.02)
            yaw = self.np_random.uniform(0, 2 * np.pi)
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.joint(f"box_lid_joint_{i}").qpos[:3] = obj_pos
            self._data.joint(f"box_lid_joint_{i}").qpos[3:] = obj_ori

    # ==================================================================
    # Hooks: set pegs & lids to goal (task-mode goal observation)
    # ==================================================================
    def _init_specific_objects_to_goal(self, task_info):
        goal_peg_xyzs = task_info["goal"]["peg_xyzs"]
        goal_lid_xyzs = task_info["goal"]["lid_xyzs"]

        for i in range(self._num_pegs):
            self._data.joint(f"peg_joint_{i}").qpos[:3] = goal_peg_xyzs[i]
            self._data.joint(f"peg_joint_{i}").qpos[
                3:
            ] = lie.SO3.identity().wxyz.tolist()
            self._data.mocap_pos[self._peg_target_mocap_ids[i]] = goal_peg_xyzs[i]
            self._data.mocap_quat[self._peg_target_mocap_ids[i]] = (
                lie.SO3.identity().wxyz.tolist()
            )

        for i in range(self._num_lids):
            self._data.joint(f"box_lid_joint_{i}").qpos[:3] = goal_lid_xyzs[i]
            self._data.joint(f"box_lid_joint_{i}").qpos[
                3:
            ] = lie.SO3.identity().wxyz.tolist()
            self._data.mocap_pos[self._lid_target_mocap_ids[i]] = goal_lid_xyzs[i]
            self._data.mocap_quat[self._lid_target_mocap_ids[i]] = (
                lie.SO3.identity().wxyz.tolist()
            )

    # ==================================================================
    # Hooks: set pegs & lids to init (task-mode actual reset)
    # ==================================================================
    def _init_specific_objects_to_init(self, task_info):
        init_peg_xyzs = task_info["init"]["peg_xyzs"]
        init_lid_xyzs = task_info["init"]["lid_xyzs"]
        goal_peg_xyzs = task_info["goal"]["peg_xyzs"]
        goal_lid_xyzs = task_info["goal"]["lid_xyzs"]

        for i in range(self._num_pegs):
            obj_pos = init_peg_xyzs[i].copy()
            obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
            self._data.joint(f"peg_joint_{i}").qpos[:3] = obj_pos
            yaw = self.np_random.uniform(0, 2 * np.pi)
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.joint(f"peg_joint_{i}").qpos[3:] = obj_ori
            self._data.mocap_pos[self._peg_target_mocap_ids[i]] = goal_peg_xyzs[i]
            self._data.mocap_quat[self._peg_target_mocap_ids[i]] = (
                lie.SO3.identity().wxyz.tolist()
            )

        for i in range(self._num_lids):
            # Randomize lid spawn position within lid bounds.
            xy = self.np_random.uniform(*self._lid_sampling_bounds)
            obj_pos = np.array([xy[0], xy[1], init_lid_xyzs[i][2]])
            self._data.joint(f"box_lid_joint_{i}").qpos[:3] = obj_pos
            yaw = self.np_random.uniform(0, 2 * np.pi)
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.joint(f"box_lid_joint_{i}").qpos[3:] = obj_ori
            self._data.mocap_pos[self._lid_target_mocap_ids[i]] = goal_lid_xyzs[i]
            self._data.mocap_quat[self._lid_target_mocap_ids[i]] = (
                lie.SO3.identity().wxyz.tolist()
            )

    # ==================================================================
    # Hooks: task probabilities for set_new_target (data-collection)
    # ==================================================================
    def _get_task_probabilities(self):
        probs = super()._get_task_probabilities()
        probs["peg"] = 1.0
        probs["lid"] = 1.0
        return probs

    # ==================================================================
    # Hooks: handle peg / lid targets in set_new_target
    # ==================================================================
    def _handle_specific_target(self, task_name):
        if task_name == "peg":
            xy = self.np_random.uniform(*self._peg_sampling_bounds)
            tar_pos = (*xy, 0.02)
            yaw = self.np_random.uniform(0, 2 * np.pi)
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.mocap_pos[self._peg_target_mocap_ids[0]] = tar_pos
            self._data.mocap_quat[self._peg_target_mocap_ids[0]] = tar_ori
            if self._visualize_info:
                for gid in self._peg_target_geom_ids_list[0]:
                    self._model.geom(gid).rgba[3] = 0.2
            return True

        if task_name == "lid":
            xy = self.np_random.uniform(*self._lid_sampling_bounds)
            tar_pos = (*xy, 0.02)
            yaw = self.np_random.uniform(0, 2 * np.pi)
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.mocap_pos[self._lid_target_mocap_ids[0]] = tar_pos
            self._data.mocap_quat[self._lid_target_mocap_ids[0]] = tar_ori
            if self._visualize_info:
                for gid in self._lid_target_geom_ids_list[0]:
                    self._model.geom(gid).rgba[3] = 0.2
            return True

        return False

    # ==================================================================
    # Hooks: peg / lid successes
    # ==================================================================
    def _get_specific_successes(self):
        peg_success = bool(
            np.linalg.norm(
                self._data.joint("peg_joint_0").qpos[:3]
                - self._data.mocap_pos[self._peg_target_mocap_ids[0]]
            )
            <= 0.04
        )
        lid_success = bool(
            np.linalg.norm(
                self._data.joint("box_lid_joint_0").qpos[:3]
                - self._data.mocap_pos[self._lid_target_mocap_ids[0]]
            )
            <= 0.04
        )
        return [(peg_success, "peg"), (lid_success, "lid")]

    # ==================================================================
    # Hooks: peg / lid privileged object info
    # ==================================================================
    def _get_specific_object_info(self, ob_info):
        for i in range(self._num_pegs):
            ob_info[f"privileged_peg_{i}_pos"] = (
                self._data.joint(f"peg_joint_{i}").qpos[:3].copy()
            )
            ob_info[f"privileged_peg_{i}_quat"] = (
                self._data.joint(f"peg_joint_{i}").qpos[3:].copy()
            )
            ob_info[f"privileged_peg_{i}_yaw"] = np.array(
                [
                    lie.SO3(
                        wxyz=self._data.joint(f"peg_joint_{i}").qpos[3:]
                    ).compute_yaw_radians()
                ]
            )
            ob_info[f"privileged_peg_{i}_state"] = 1
            ob_info[f"privileged_peg_{i}_handle_pos"] = self._data.site_xpos[
                self._peg_handle_site_ids[i]
            ].copy()

        for i in range(self._num_lids):
            ob_info[f"privileged_lid_{i}_pos"] = (
                self._data.joint(f"box_lid_joint_{i}").qpos[:3].copy()
            )
            ob_info[f"privileged_lid_{i}_quat"] = (
                self._data.joint(f"box_lid_joint_{i}").qpos[3:].copy()
            )
            ob_info[f"privileged_lid_{i}_yaw"] = np.array(
                [
                    lie.SO3(
                        wxyz=self._data.joint(f"box_lid_joint_{i}").qpos[3:]
                    ).compute_yaw_radians()
                ]
            )
            ob_info[f"privileged_lid_{i}_state"] = 1
            ob_info[f"privileged_lid_{i}_handle_pos"] = self._data.site_xpos[
                self._lid_handle_site_ids[i]
            ].copy()

    # ==================================================================
    # Hooks: peg / lid privileged target info (data-collection mode)
    # ==================================================================
    def _get_specific_target_info(self, ob_info):
        target_peg_mocap_id = self._peg_target_mocap_ids[self._target_peg]
        ob_info["privileged_target_peg"] = self._target_peg
        ob_info["privileged_target_peg_pos"] = self._data.mocap_pos[
            target_peg_mocap_id
        ].copy()
        ob_info["privileged_target_peg_yaw"] = np.array(
            [
                lie.SO3(
                    wxyz=self._data.mocap_quat[target_peg_mocap_id]
                ).compute_yaw_radians()
            ]
        )
        ob_info["privileged_target_peg_quat"] = self._data.mocap_quat[
            target_peg_mocap_id
        ].copy()

        target_lid_mocap_id = self._lid_target_mocap_ids[self._target_lid]
        ob_info["privileged_target_lid"] = self._target_lid
        ob_info["privileged_target_lid_pos"] = self._data.mocap_pos[
            target_lid_mocap_id
        ].copy()
        ob_info["privileged_target_lid_yaw"] = np.array(
            [
                lie.SO3(
                    wxyz=self._data.mocap_quat[target_lid_mocap_id]
                ).compute_yaw_radians()
            ]
        )
        ob_info["privileged_target_lid_quat"] = self._data.mocap_quat[
            target_lid_mocap_id
        ].copy()

    # ==================================================================
    # Hooks: peg / lid observation vectors
    # ==================================================================
    def _add_specific_observations(self, ob, ob_info, xyz_center, xyz_scaler):
        for i in range(self._num_pegs):
            ob.extend(
                [
                    (ob_info[f"privileged_peg_{i}_pos"] - xyz_center) * xyz_scaler,
                    ob_info[f"privileged_peg_{i}_quat"],
                    np.cos(ob_info[f"privileged_peg_{i}_yaw"]),
                    np.sin(ob_info[f"privileged_peg_{i}_yaw"]),
                ]
            )
        for i in range(self._num_lids):
            ob.extend(
                [
                    (ob_info[f"privileged_lid_{i}_pos"] - xyz_center) * xyz_scaler,
                    ob_info[f"privileged_lid_{i}_quat"],
                    np.cos(ob_info[f"privileged_lid_{i}_yaw"]),
                    np.sin(ob_info[f"privileged_lid_{i}_yaw"]),
                ]
            )

    # ==================================================================
    # Hooks: peg / lid oracle observation vectors
    # ==================================================================
    def _add_specific_oracle_obs(self, ob, ob_info, xyz_center, xyz_scaler):
        for i in range(self._num_pegs):
            ob.append(
                (ob_info[f"privileged_peg_{i}_pos"] - xyz_center) * xyz_scaler
            )
        for i in range(self._num_lids):
            ob.append(
                (ob_info[f"privileged_lid_{i}_pos"] - xyz_center) * xyz_scaler
            )

    # ==================================================================
    # post_step — add peg/lid stability check & colour adjustment
    # ==================================================================
    def post_step(self):
        # -- numerical stability check for pegs & lids (task mode) -------------
        if self._mode == "task":
            is_healthy = True
            for i in range(self._num_pegs):
                obj_pos = self._data.joint(f"peg_joint_{i}").qpos[:3]
                if np.any(obj_pos <= self._workspace_bounds[0] - 0.2) or np.any(
                    obj_pos >= self._workspace_bounds[1] + 0.2
                ):
                    is_healthy = False
                    break
            if is_healthy:
                for i in range(self._num_lids):
                    obj_pos = self._data.joint(f"box_lid_joint_{i}").qpos[:3]
                    if np.any(obj_pos <= self._workspace_bounds[0] - 0.2) or np.any(
                        obj_pos >= self._workspace_bounds[1] + 0.2
                    ):
                        is_healthy = False
                        break
            if not is_healthy:
                print(
                    "Numerical instability detected. Resetting peg/lid positions.",
                    flush=True,
                )
                for i in range(self._num_pegs):
                    xy = self.np_random.uniform(*self._object_sampling_bounds)
                    obj_pos = (*xy, 0.02)
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    self._data.joint(f"peg_joint_{i}").qpos[:3] = obj_pos
                    self._data.joint(f"peg_joint_{i}").qpos[3:] = obj_ori
                    self._data.joint(f"peg_joint_{i}").qvel[:] = 0.0
                for i in range(self._num_lids):
                    xy = self.np_random.uniform(*self._object_sampling_bounds)
                    obj_pos = (*xy, 0.02)
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    self._data.joint(f"box_lid_joint_{i}").qpos[:3] = obj_pos
                    self._data.joint(f"box_lid_joint_{i}").qpos[3:] = obj_ori
                    self._data.joint(f"box_lid_joint_{i}").qvel[:] = 0.0
                mujoco.mj_forward(self._model, self._data)

        super().post_step()

        # -- adjust peg / lid colours based on success (visualize_info) --------
        if self._visualize_info:
            peg_success = (
                np.linalg.norm(
                    self._data.joint("peg_joint_0").qpos[:3]
                    - self._data.mocap_pos[self._peg_target_mocap_ids[0]]
                )
                <= 0.04
            )
            lid_success = (
                np.linalg.norm(
                    self._data.joint("box_lid_joint_0").qpos[:3]
                    - self._data.mocap_pos[self._lid_target_mocap_ids[0]]
                )
                <= 0.04
            )
            for gid in self._peg_geom_ids_list[0]:
                self._model.geom(gid).rgba[3] = 0.5 if peg_success else 1.0
            for gid in self._lid_geom_ids_list[0]:
                self._model.geom(gid).rgba[3] = 0.5 if lid_success else 1.0

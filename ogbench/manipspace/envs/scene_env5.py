import mujoco
import numpy as np
from ogbench.manipspace.envs.objects import (
    DrawerObject,
    ShelfObject,
    ButtonTripleObject,
    CubeObject,
)
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase


class SceneEnv5(SceneEnvBase):
    """Scene environment: drawer + 3 buttons + shelf + 1 cube.

    Button 0 locks/unlocks the drawer. Buttons 1 and 2 are toggle-only.
    The shelf is a static fixture (not a joint object).
    """

    def __init__(self, env_type, permute_blocks=True, *args, **kwargs):
        self._objects = [
            CubeObject(count=1),
            ButtonTripleObject(),
            DrawerObject(pos=(0.33, -0.42, 0.084), euler=(0, 0, 3.14)),
            ShelfObject(),
        ]
        self._button_locks = {0: "drawer_slide"}

        super().__init__(env_type, permute_blocks=permute_blocks, *args, **kwargs)

        self._drawer_center = np.array([0.33, -0.24, 0.066])

        # Shelf goal position (static fixture, cached after compilation).
        self._shelf_goal_pos = None

    # ------------------------------------------------------------------
    # Task definitions
    # ------------------------------------------------------------------
    def set_tasks(self):
        self.task_infos = [
            dict(
                task_name="task1_open",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1, 1, 1]),
                    drawer_pos=0.0,
                    shelf_block=0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([1, 1, 1]),
                    drawer_pos=-0.16,
                    shelf_block=0,
                ),
            ),
            dict(
                task_name="task2_unlock_and_lock",
                init=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([0, 0, 0]),
                    drawer_pos=-0.16,
                    shelf_block=0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02]]),
                    button_states=np.array([0, 0, 0]),
                    drawer_pos=0.0,
                    shelf_block=0,
                ),
            ),
            dict(
                task_name="task3_rearrange_medium",
                init=dict(
                    block_xyzs=np.array([[0.4, -0.05, 0.02]]),
                    button_states=np.array([1, 0, 0]),
                    drawer_pos=0.0,
                    shelf_block=0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02]]),
                    button_states=np.array([1, 1, 1]),
                    drawer_pos=-0.16,
                    shelf_block=0,
                ),
            ),
            dict(
                task_name="task4_put_in_drawer",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0, 0]),
                    drawer_pos=0.0,
                    shelf_block=0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([1, 0, 0]),
                    drawer_pos=0.0,
                    shelf_block=0,
                ),
            ),
            dict(
                task_name="task5_rearrange_hard",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button_states=np.array([0, 0, 0]),
                    drawer_pos=0.0,
                    shelf_block=0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.356, 0.065986]]),
                    button_states=np.array([0, 0, 0]),
                    drawer_pos=0.0,
                    shelf_block=1,
                ),
            ),
            dict(
                task_name="task6_put_on_shelf",
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0, 0]),
                    drawer_pos=0.0,
                    shelf_block=0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02]]),
                    button_states=np.array([1, 1, 1]),
                    drawer_pos=-0.16,
                    shelf_block=1,
                ),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

    # ------------------------------------------------------------------
    # Hooks: load shelf XML (drawer is auto-loaded via _joint_objects)
    # ------------------------------------------------------------------
    def _get_objects_to_load(self):
        objects = super()._get_objects_to_load()
        objects.append(("shelf.xml", "shelf"))
        return objects

    # ------------------------------------------------------------------
    # Hooks: post-compilation — resolve shelf site and cache position
    # ------------------------------------------------------------------
    def _post_compilation_specific(self):
        self._shelf_goal_site_id = self._model.site("shelf_goal").id
        mujoco.mj_kinematics(self._model, self._data)
        self._shelf_goal_pos = self._data.site_xpos[self._shelf_goal_site_id].copy()

    # ------------------------------------------------------------------
    # Hooks: task probabilities — add shelf
    # ------------------------------------------------------------------
    def _get_task_probabilities(self):
        probs = super()._get_task_probabilities()
        available = sum(
            1
            for i in range(self._num_cubes)
            if not self._is_in_drawer(self._data.joint(f"object_joint_{i}").qpos[:3])
        )
        probs["shelf"] = 1.0 if available > 0 else 0.0
        return probs

    # ------------------------------------------------------------------
    # Hooks: specific task fields used in set_tasks
    # ------------------------------------------------------------------
    def _get_specific_task_fields(self):
        return {"shelf_block": 0}

    # ------------------------------------------------------------------
    # Hooks: handle shelf target in data-collection set_new_target
    # ------------------------------------------------------------------
    def _handle_specific_target(self, task_name):
        if task_name != "shelf":
            return False

        available_blocks = [
            i
            for i in range(self._num_cubes)
            if not self._is_in_drawer(self._data.joint(f"object_joint_{i}").qpos[:3])
        ]
        self._target_block = self.np_random.choice(available_blocks)

        shelf_pos = self._data.site_xpos[self._shelf_goal_site_id].copy()
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz

        self._data.mocap_pos[self._cube_target_mocap_ids[self._target_block]] = shelf_pos
        self._data.mocap_quat[self._cube_target_mocap_ids[self._target_block]] = identity_quat

        # Hide non-target blocks.
        for i in range(self._num_cubes):
            if i != self._target_block:
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = identity_quat

        # Adjust target colours.
        for i in range(self._num_cubes):
            alpha = 0.2 if (self._visualize_info and i == self._target_block) else 0.0
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[3] = alpha
        return True

    # ------------------------------------------------------------------
    # Hooks: init specific objects to goal / init (task mode)
    # ------------------------------------------------------------------
    def _init_specific_objects_to_goal(self, task_info):
        goal_shelf_block = task_info["goal"].get("shelf_block", 0)
        if goal_shelf_block == 1 and self._shelf_goal_pos is not None:
            for i in range(self._num_cubes):
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (
                    self._shelf_goal_pos.copy()
                )

    def _init_specific_objects_to_init(self, task_info):
        goal_shelf_block = task_info["goal"].get("shelf_block", 0)
        if goal_shelf_block == 1 and self._shelf_goal_pos is not None:
            for i in range(self._num_cubes):
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (
                    self._shelf_goal_pos.copy()
                )

    # ------------------------------------------------------------------
    # Hooks: specific object info
    # ------------------------------------------------------------------
    def _get_specific_object_info(self, ob_info):
        if self._shelf_goal_pos is not None:
            ob_info["privileged_shelf_goal_pos"] = self._shelf_goal_pos.copy()

    # ------------------------------------------------------------------
    # Hooks: specific observations (add shelf goal position)
    # ------------------------------------------------------------------
    def _add_specific_observations(self, ob, ob_info, xyz_center, xyz_scaler):
        if "privileged_shelf_goal_pos" in ob_info:
            ob.append((ob_info["privileged_shelf_goal_pos"] - xyz_center) * xyz_scaler)

    # ------------------------------------------------------------------
    # Hooks: specific successes (shelf = cube at shelf_goal, data-col only)
    # ------------------------------------------------------------------
    def _get_specific_successes(self):
        # In task mode, shelf is already covered by cube_successes (mocap at
        # shelf_goal).  Only add an explicit shelf entry for data-collection.
        if self._mode == "data_collection" and self._shelf_goal_pos is not None and self._num_cubes > 0:
            obj_pos = self._data.joint(f"object_joint_{self._target_block}").qpos[:3]
            shelf_success = bool(np.linalg.norm(obj_pos - self._shelf_goal_pos) <= 0.04)
            return [(shelf_success, "shelf")]
        return []

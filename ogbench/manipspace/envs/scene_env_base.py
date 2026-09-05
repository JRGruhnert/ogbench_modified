import mujoco
import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ogbench.manipspace.envs.objects.base import SceneObject


class SceneEnvBase(ManipSpaceEnv):
    def __init__(self, env_type, objects=None, permute_blocks=True, *args, **kwargs):
        self._env_type = env_type
        self._objects = objects or []
        self._permute_blocks = permute_blocks
        super().__init__(*args, **kwargs)
        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.6, 0.2, 0.35]])
        self._oracle_just_done = False
        self._task_selection_counts = {}
        self._cur_goal_ob = None
        self._cur_goal_rendered = None
        self._render_goal = False

    def set_tasks(self):
        self.task_infos = []

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        is_collection = self._mode in ("data_collection", "collection")
        is_randomized = self._mode == "randomized"

        if is_collection:
            self.initialize_arm()
            for obj in self.objects:
                obj.randomize(self)
            self._apply_button_states()
            self.set_new_target(return_info=False)
        elif is_randomized:
            # Like task mode, but the goals are randomized instead of coming from a task spec
            self.initialize_arm()
            for obj in self.objects:
                obj.randomize(self)
                obj.handle_target(self)
            self._apply_button_states()

            saved_qpos, saved_qvel = self._data.qpos.copy(), self._data.qvel.copy()
            saved_cur_states = {}
            for obj in self.objects:
                if hasattr(obj, "_cur_state"):
                    saved_cur_states[obj.name] = obj._cur_state.copy()

            for obj in self.objects:
                self._set_object_to_target(obj)
            self._apply_button_states()
            self._cur_goal_ob = (
                self.compute_oracle_observation()
                if self._use_oracle_rep
                else self.compute_ob_info()
            )
            self._cur_goal_rendered = (
                self.get_pixel_observation() if self._render_goal else None
            )

            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            for obj in self.objects:
                if obj.name in saved_cur_states:
                    obj._cur_state[:] = saved_cur_states[obj.name]
            self._apply_button_states()
        else:
            if self.cur_task_info is None:
                self.cur_task_id = 1
                self.cur_task_info = self.task_infos[0]

            saved_qpos, saved_qvel = self._data.qpos.copy(), self._data.qvel.copy()
            self.initialize_arm()
            for obj in self._objects:
                obj.init_to_goal(self, self.cur_task_info)
            self._apply_button_states()
            for _ in range(2):
                self.step(self.action_space.sample())
            self._cur_goal_ob = (
                self.compute_oracle_observation()
                if self._use_oracle_rep
                else self.compute_ob_info()
            )
            self._cur_goal_rendered = (
                self.get_pixel_observation() if self._render_goal else None
            )

            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for obj in self._objects:
                obj.init_to_init(self, self.cur_task_info)
            for obj in self._objects:
                value = obj.get_target_from_task(self.cur_task_info["goal"])
                if value is None:
                    continue
                if hasattr(obj, "_target_mocap_id"):
                    arr = np.asarray(value)
                    pos = arr[obj.id] if arr.ndim == 2 else np.asarray(value).ravel()
                    self._set_object_target(obj, (pos, lie.SO3.identity().wxyz))
                else:
                    self._set_object_target(obj, value)
            self._apply_button_states()

        self.pre_step()
        self.post_step()
        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        assert self._mode in ("data_collection", "collection")
        self._oracle_just_done = True

        probs = self._get_task_probabilities()
        task_list, prob_list = [], []
        for obj in self.objects:
            if obj.name in probs:
                task_list.append(obj.name)
                prob_list.append(probs[obj.name])

        # Keep only tasks that are currently available (positive probability).
        available = [(n, w) for n, w in zip(task_list, prob_list) if w > 0]
        if not available:
            if return_info:
                return self.compute_observation(), self.get_reset_info()
            return

        names = [n for n, _ in available]
        raw = np.array([w for _, w in available], dtype=float)

        # Inverse-frequency balancing: make under-selected tasks more likely while
        # still respecting each object's availability weight.
        counts = np.array(
            [self._task_selection_counts.get(n, 0) for n in names], dtype=float
        )
        weights = raw / (counts + 1.0)
        weights /= weights.sum()

        self._target_task = self.np_random.choice(names, p=weights)
        self._task_selection_counts[self._target_task] = (
            self._task_selection_counts.get(self._target_task, 0) + 1
        )

        for obj in self.objects:
            if obj.name == self._target_task:
                obj.randomize(self)
                obj.handle_target(self)
                break

        mujoco.mj_kinematics(self._model, self._data)

        # Compute the goal observation (target state) for goal-conditioned data.
        self._cur_goal_ob = self._compute_goal_observation()
        self._cur_goal_rendered = (
            self.get_pixel_observation() if self._render_goal else None
        )

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def _set_object_to_target(self, obj):
        """Move one object's current state to its target state."""
        if hasattr(obj, "_target_mocap_id"):
            pos = self._data.mocap_pos[obj._target_mocap_id].copy()
            quat = self._data.mocap_quat[obj._target_mocap_id].copy()
            self._data.joint(obj.joint_name).qpos[:3] = pos
            self._data.joint(obj.joint_name).qpos[3:] = quat
        elif hasattr(obj, "_target_val"):
            self._data.joint(obj.joint_name).qpos[0] = obj._target_val
        elif hasattr(obj, "_target_button_states"):
            obj._cur_state[0] = obj._target_button_states[0]

    def _set_target_object_to_target(self):
        """Move the target object to its current target state."""
        for obj in self.objects:
            if obj.name != self._target_task:
                continue
            self._set_object_to_target(obj)
            break

    def _compute_goal_observation(self):
        """Compute the observation of the target state (used as the goal)."""
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        saved_oracle_just_done = self._oracle_just_done

        saved_cur_states = {}
        for obj in self.objects:
            if hasattr(obj, "_cur_state"):
                saved_cur_states[obj.name] = obj._cur_state.copy()

        self._set_target_object_to_target()
        self._apply_button_states()

        goal_ob = (
            self.compute_oracle_observation()
            if self._use_oracle_rep
            else self.compute_ob_info()
        )

        self._data.qpos[:] = saved_qpos
        self._data.qvel[:] = saved_qvel
        for obj in self.objects:
            if obj.name in saved_cur_states:
                obj._cur_state[:] = saved_cur_states[obj.name]
        self._oracle_just_done = saved_oracle_just_done

        # Restore colors/locks to match the restored current state.
        self._apply_button_states()
        return goal_ob

    def _get_task_probabilities(self):
        probs = {}
        for obj in self.objects:
            prob = obj.get_task_probability(self)
            if prob is not None:
                probs[obj.name] = prob
        return probs

    def _apply_button_states(self):
        for obj in self.objects:
            obj.apply_colors_and_locks(self)
        mujoco.mj_forward(self._model, self._data)

    def get_object_boundaries(self) -> dict:
        shapes = {}
        for obj in self.objects:
            if hasattr(obj, "_target_mocap_id"):
                shape = self._free_body_boundary(obj)
            elif (
                hasattr(obj, "_target_val")
                and getattr(obj, "_site_id", None) is not None
            ):
                shape = self._joint_reach_boundary(obj)
            elif getattr(obj, "_body_id", None) is not None:
                # Passive container (box/shelf): static footprint.
                body_ids = self._subtree_body_ids(obj._body_id)
                lo, hi = self._body_xy_aabb(body_ids)
                shape = dict(
                    type="rect",
                    kind="static",
                    xy_min=[float(lo[0]), float(lo[1])],
                    xy_max=[float(hi[0]), float(hi[1])],
                )
            else:
                # Remaining static objects (e.g. buttons): derive the assembly
                # body from their geoms (walk up to the top-level body) and use
                # its footprint.
                gids = getattr(obj, "_geom_ids", None)
                if not gids:
                    continue
                flat = (
                    [g for group in gids for g in group]
                    if any(isinstance(g, (list, tuple)) for g in gids)
                    else list(gids)
                )
                root = int(self._model.geom_bodyid[flat[0]])
                while self._model.body_parentid[root] != 0:
                    root = int(self._model.body_parentid[root])
                lo, hi = self._body_xy_aabb(self._subtree_body_ids(root))
                shape = dict(
                    type="rect",
                    kind="static",
                    xy_min=[float(lo[0]), float(lo[1])],
                    xy_max=[float(hi[0]), float(hi[1])],
                )
            if shape is not None:
                shapes[obj.name] = shape
        return shapes

    def _subtree_body_ids(self, root: int):
        """All body ids in the kinematic subtree below `root` (root included)."""
        out = [root]
        for b in range(self._model.nbody):
            p = b
            while p != 0 and p != root:
                p = self._model.body_parentid[p]
            if p == root and b != root:
                out.append(b)
        return out

    def _body_planar_radius(self, body_ids):
        """Max x-y distance from a body frame origin to any of its geoms.

        Used to pad spawn/reach rectangles by the object's own size. For
        rotated parts this is an upper bound of the planar extent, which is the
        right thing when the object's orientation is free/random.
        """
        m = self._model
        if not isinstance(body_ids, (list, tuple)):
            body_ids = [body_ids]
        r = 0.0
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] not in body_ids:
                continue
            px, py = float(m.geom_pos[gid][0]), float(m.geom_pos[gid][1])
            s = m.geom_size[gid]
            gt = int(m.geom_type[gid])
            if gt == mujoco.mjtGeom.mjGEOM_BOX:
                # Farthest corner distance from the body origin (rotating the
                # whole object about the origin preserves corner norms, so this
                # stays valid for arbitrary yaw).
                shape = float(np.hypot(abs(px) + s[0], abs(py) + s[1]))
            elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
                shape = float(np.hypot(px, py) + s[0])
            elif gt in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                shape = float(np.hypot(px, py) + s[0] + s[1])  # radius + half len
            elif gt == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[gid])
                adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                if num:
                    shape = float(
                        np.hypot(px, py)
                        + np.max(np.linalg.norm(m.mesh_vert[adr : adr + num], axis=1))
                    )
                else:
                    shape = float(np.hypot(px, py))
            else:
                shape = float(np.hypot(px, py) + np.max(s))
            r = max(r, shape)
        return r

    def _free_body_boundary(self, obj):
        """Padded spawn rectangle for a free-floating object."""
        bounds = getattr(obj, "_sampling_bounds", None)
        if bounds is None:
            bounds = self._object_sampling_bounds
        jid = self._model.joint(obj.joint_name).id
        pad = self._body_planar_radius(
            self._subtree_body_ids(self._model.jnt_bodyid[jid])
        )
        return dict(
            type="rect",
            kind="free",
            xy_min=[float(bounds[0][0] - pad), float(bounds[0][1] - pad)],
            xy_max=[float(bounds[1][0] + pad), float(bounds[1][1] + pad)],
            spawn_xy_min=[float(bounds[0][0]), float(bounds[0][1])],
            spawn_xy_max=[float(bounds[1][0]), float(bounds[1][1])],
            pad=float(pad),
        )

    def _site_xy_at(self, obj, joint_val, site_id):
        """Set a 1-D joint to `joint_val` and return the site world x-y."""
        self._data.joint(obj.joint_name).qpos[0] = float(joint_val)
        mujoco.mj_forward(self._model, self._data)
        return self._data.site_xpos[site_id][:2].copy()

    def _body_xy_aabb(self, body_ids):
        """World x-y bounding box (geom surfaces included) of the given bodies.

        Includes mesh geometry (vertices transformed into the world frame), so
        the box reflects the full occupied footprint of the part, not just the
        collision primitives. Evaluated at the *current* state, so call it
        after moving the joint to the pose you want to measure.
        """
        m, d = self._model, self._data
        if not isinstance(body_ids, (list, tuple)):
            body_ids = [body_ids]
        pts = []
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] not in body_ids:
                continue
            p = d.geom_xpos[gid]
            R = d.geom_xmat[gid].reshape(3, 3)
            gt = int(m.geom_type[gid])
            s = m.geom_size[gid]
            if gt == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[gid])
                adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                if num:
                    verts = m.mesh_vert[adr : adr + num]
                    world = verts @ R.T + p  # (n,3)
                    pts.append(world[:, :2])
                continue
            # Convex-primitive half-extent along each world axis (rotation aware
            # via the geom's world rotation), enough for a footprint box.
            h = np.abs(R) @ np.asarray(s[:3], dtype=float)
            pts.append(
                np.array([[p[0] + h[0], p[1] + h[1]], [p[0] - h[0], p[1] - h[1]]])
            )
        if not pts:
            return np.zeros(2), np.zeros(2)
        allp = np.vstack(pts)
        return allp.min(axis=0), allp.max(axis=0)

    def _joint_reach_boundary(self, obj):
        """Prismatic strip (rect) or revolute arc for a jointed object."""
        jid = self._model.joint(obj.joint_name).id
        jt = int(self._model.jnt_type[jid])
        site = obj._site_id
        lo, hi = obj.pos_range
        saved = float(self._data.joint(obj.joint_name).qpos[0])
        try:
            body_ids = self._subtree_body_ids(self._model.jnt_bodyid[jid])
            if jt == mujoco.mjtJoint.mjJNT_SLIDE:
                # Swept area of the moving body over the whole joint range.
                mins, maxs = [], []
                endpoints = []
                for v in (lo, hi):
                    A = self._site_xy_at(obj, v, site)
                    endpoints.append(A)
                    lo2, hi2 = self._body_xy_aabb(body_ids)
                    mins.append(lo2)
                    maxs.append(hi2)
                lo_all = np.min(np.array(mins), axis=0)
                hi_all = np.max(np.array(maxs), axis=0)
                return dict(
                    type="rect",
                    kind="prismatic",
                    xy_min=[float(lo_all[0]), float(lo_all[1])],
                    xy_max=[float(hi_all[0]), float(hi_all[1])],
                    end0=[float(endpoints[0][0]), float(endpoints[0][1])],
                    end1=[float(endpoints[1][0]), float(endpoints[1][1])],
                    joint_range=[float(lo), float(hi)],
                )
            A = self._site_xy_at(obj, lo, site)
            B = self._site_xy_at(obj, hi, site)
            # Revolute: pivot + handle-tip arc over the joint range.
            pivot = self._joint_world_pivot(jid)
            r = float(np.linalg.norm(A - pivot))
            a0 = float(np.arctan2(A[1] - pivot[1], A[0] - pivot[0]))
            a1 = float(np.arctan2(B[1] - pivot[1], B[0] - pivot[0]))
            # Direction: sample the midpoint of the range and see which way the
            # handle moves from the lo-angle to the hi-angle.
            am = float(
                np.arctan2(
                    self._site_xy_at(obj, 0.5 * (lo + hi), site)[1] - pivot[1],
                    self._site_xy_at(obj, 0.5 * (lo + hi), site)[0] - pivot[0],
                )
            )
            dmid = (am - a0 + np.pi) % (2 * np.pi) - np.pi
            direction = 1 if dmid >= 0 else -1
            if direction == 1:
                while a1 < a0:
                    a1 += 2 * np.pi
            else:
                while a1 > a0:
                    a1 -= 2 * np.pi
            return dict(
                type="arc",
                kind="revolute",
                center=[float(pivot[0]), float(pivot[1])],
                radius=r,
                theta0=a0,
                theta1=a1,
                direction=direction,
                joint_range=[float(lo), float(hi)],
            )
        finally:
            self._data.joint(obj.joint_name).qpos[0] = saved
            mujoco.mj_forward(self._model, self._data)

    def _joint_world_pivot(self, jid):
        """World x-y of a joint's pivot (anchor in the parent-body frame)."""
        m = self._model
        anchor = m.jnt_pos[jid][:2]
        parent = m.body_parentid[m.jnt_bodyid[jid]]
        if parent == 0:
            return anchor.copy()
        p = self._data.xpos[parent][:2]
        quat = self._data.xquat[parent]
        # Rotate the anchor's x-y by the parent orientation.
        x, y = float(anchor[0]), float(anchor[1])
        w, qx, qy, qz = quat
        # (x, y, 0) rotated by quat, take x-y.
        vx = (1 - 2 * (qy * qy + qz * qz)) * x + 2 * (qx * qy - w * qz) * y
        vy = 2 * (qx * qy + w * qz) * x + (1 - 2 * (qx * qx + qz * qz)) * y
        return np.array([p[0] + vx, p[1] + vy])

    def add_objects(self, arena_mjcf):
        for obj in self.objects:
            obj.load(arena_mjcf, self._desc_dir)
        self.add_cameras(arena_mjcf)

    def add_cameras(self, arena_mjcf):
        # Add cameras.
        cameras = {
            "front": {
                "pos": (1.139, 0.000, 0.821),
                "xyaxes": (0.000, 1.000, 0.000, -0.627, 0.000, 0.779),
            },
            "front_pixels": {
                "pos": (0.905, 0.000, 0.762),
                "xyaxes": (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add("camera", name=camera_name, **camera_kwargs)

    @property
    def objects(self) -> list[SceneObject]:
        return self._objects

    def get_object(self, name):
        """Find a SceneObject by name."""
        for obj in self._objects:
            if obj.name == name:
                return obj
        return None

    def set_state(self, qpos, qvel):
        for obj in self.objects:
            obj.apply_lock(self._model)

        mujoco.mj_forward(self._model, self._data)  # type: ignore
        super().set_state(qpos, qvel)

    def post_compilation_objects(self):
        for obj in self.objects:
            obj.post_compilation(self)

    def default_quaternion(self) -> np.ndarray:
        return np.array(lie.SO3.identity().wxyz.tolist())

    def pre_step(self):
        for obj in self.objects:
            obj.pre_step()
        super().pre_step()

    def _compute_successes(self):
        successes = []
        for obj in self.objects:
            result = obj.compute_success(self)
            if result is not None:
                successes.append(result)
        return successes

    def _evaluate_success(self, successes):
        if self._mode in ("data_collection", "collection"):
            return any(val for val, name in successes if name == self._target_task)
        return all(val for val, _ in successes)

    def post_step(self):
        successes = self._compute_successes()
        self._success = self._evaluate_success(successes)

        for obj in self.objects:
            obj.post_step(self)
            obj.health_check_and_colors(self, successes)

        self._apply_button_states()

    def add_object_info(self, ob_info: dict):
        for obj in self.objects:
            ob_info.update(obj.get_info(self))

        if self._mode in ("data_collection", "collection"):
            ob_info["privileged_target_task"] = self._target_task
            ob_info["oracle_done"] = float(self._oracle_just_done)
            self._oracle_just_done = False
            for obj in self.objects:
                ob_info.update(obj.get_info_target(self))
            # Oracle success: is the current target object at its goal?
            ob_info["oracle_success"] = float(
                any(
                    val
                    for val, name in self._compute_successes()
                    if name == self._target_task
                )
            )

        ob_info["meta_xyz_center"] = np.array([0.425, 0.0, 0.0])
        ob_info["meta_xyz_scaler"] = np.array([10.0])
        ob_info["meta_gripper_scaler"] = np.array([3.0])
        ob_info["meta_prismatic_max"] = np.array([3.0])

    def get_reset_info(self):
        reset_info = super().get_reset_info()
        if self._mode == "randomized":
            reset_info["goal"] = self._cur_goal_ob
            if self._render_goal and self._cur_goal_rendered is not None:
                reset_info["goal_rendered"] = self._cur_goal_rendered
        return reset_info

    def get_step_info(self):
        ob_info = super().get_step_info()
        if self._mode == "randomized":
            ob_info["goal"] = self._cur_goal_ob
        return ob_info

    def _append_object_state(self, ob: list):
        """Append each object's goal-relevant state to the observation list."""
        for obj in self.objects:
            if hasattr(obj, "_target_mocap_id"):
                # Free body: position (3D).
                ob.append(self._data.joint(obj.joint_name).qpos[:3].copy())
            elif hasattr(obj, "_target_val"):
                # Articulated joint: joint value (1D).
                ob.append(np.array([self._data.joint(obj.joint_name).qpos[0]]))
            elif hasattr(obj, "_target_button_states"):
                # Button: discrete state (1D).
                ob.append(np.array([obj._cur_state[0]], dtype=np.float64))
            # Passive containers (shelf/box) contribute no state.

    def compute_observation(self):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()

        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        ob_info = self.compute_ob_info()
        ob = [
            ob_info["proprio_joint_pos"],
            ob_info["proprio_joint_vel"],
            (ob_info["proprio_effector_pos"] - xyz_center) * xyz_scaler,
            np.cos(ob_info["proprio_effector_yaw"]),
            np.sin(ob_info["proprio_effector_yaw"]),
            ob_info["proprio_gripper_opening"] * gripper_scaler,
            ob_info["proprio_gripper_contact"],
        ]
        self._append_object_state(ob)
        return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        ob = []
        self._append_object_state(ob)
        return np.concatenate(ob)

    def compute_reward(self):
        successes = self._compute_successes()
        return float(self._evaluate_success(successes))

    def set_scene_state(self, state_dict: dict):
        for name, value in state_dict.items():
            obj = self.get_object(name)
            if obj is None:
                continue
            if not obj.can_set_state(self, value):
                return False

        for name, value in state_dict.items():
            obj = self.get_object(name)
            if obj is None:
                continue
            obj.set_state(self, value)

        self._apply_button_states()
        return True

    def _object_state_from_info(self, obj, info_dict, use_target_keys=False):
        """Parse a settable value for `obj` from a step/reset info dict.

        With `use_target_keys=True` the `heca_target_*` goal keys are read
        (e.g. `heca_target_peg0_pos`), otherwise the current-state `heca_*`
        keys are used (e.g. `heca_peg0_pos`).
        """
        prefix = "target_" if use_target_keys else ""

        if hasattr(obj, "_target_button_states"):
            key = f"heca_{prefix}{obj.name}_ste"
            if key in info_dict:
                return int(round(float(np.asarray(info_dict[key]).ravel()[0])))
            return None

        if hasattr(obj, "_target_val"):
            pos_key = f"heca_{prefix}{obj.name}_pos"

            if pos_key in info_dict and getattr(obj, "_site_id", None) is not None:
                jt = self._model.joint(obj.joint_name).type
                if jt == mujoco.mjtJoint.mjJNT_SLIDE:
                    return self._world_handle_to_joint_val(obj, info_dict[pos_key])
            for suffix in ("_ang", "_sca"):
                key = f"heca_{prefix}{obj.name}{suffix}"
                if key in info_dict:
                    return float(np.asarray(info_dict[key]).ravel()[0])
            return None

        if hasattr(obj, "_target_mocap_id"):
            base_key = f"heca_{prefix}{obj.name}_pos_base"
            pos_key = f"heca_{prefix}{obj.name}_pos"
            rot_key = f"heca_{prefix}{obj.name}_rot"
            yaw_key = f"heca_{prefix}{obj.name}_yaw"

            # Parse the requested orientation first (needed to recover the base
            # from a handle-site position below).
            if rot_key in info_dict:
                quat = np.asarray(info_dict[rot_key], dtype=float).ravel()
            elif yaw_key in info_dict:
                quat = lie.SO3.from_z_radians(
                    float(np.asarray(info_dict[yaw_key]).ravel()[0])
                ).wxyz
            else:
                quat = lie.SO3.identity().wxyz

            if base_key in info_dict:
                pos = np.asarray(info_dict[base_key], dtype=float).ravel()
            elif pos_key in info_dict:
                pos = np.asarray(info_dict[pos_key], dtype=float).ravel()
                if not use_target_keys:
                    handle_offset = getattr(obj, "handle_offset", None)
                    if handle_offset is not None:
                        pos = pos - lie.SO3(wxyz=quat).apply(
                            np.asarray(handle_offset, dtype=float)
                        )
            else:
                return None
            return (pos, quat)

        return None

    def _world_handle_to_joint_val(self, obj, handle_pos):

        joint = self._data.joint(obj.joint_name)
        q0 = float(joint.qpos[0])
        p0 = self._data.site_xpos[obj._site_id].copy()

        # Measure the world displacement of the handle for +1 joint unit.
        joint.qpos[0] = q0 + 1.0
        mujoco.mj_kinematics(self._model, self._data)
        axis = self._data.site_xpos[obj._site_id].copy() - p0
        joint.qpos[0] = q0
        mujoco.mj_kinematics(self._model, self._data)

        denom = float(np.dot(axis, axis))
        if denom <= 1e-12:
            return q0
        target = np.asarray(handle_pos, dtype=float).ravel()[:3]
        q = float(q0 + np.dot(target - p0, axis) / denom)

        pos_range = getattr(obj, "pos_range", None)
        if pos_range is not None:
            q = float(np.clip(q, pos_range[0], pos_range[1]))
        return q

    def step_scene(self, info_dict):
        self.pre_step()

        # Parse requested updates.
        targets = {}
        for obj in self.objects:
            value = self._object_state_from_info(obj, info_dict)
            if value is not None:
                targets[obj.name] = (obj, value)

        for name, (obj, value) in targets.items():
            self._set_current(obj, value)
            joint_name = getattr(obj, "joint_name", None)
            if joint_name is not None:
                self._data.joint(joint_name).qvel[:] = 0.0

        self._apply_button_states()
        self._success = self._evaluate_success(self._compute_successes())

        ob = self.compute_observation()
        info = self.get_step_info()
        info["success"] = bool(self._success)
        reward = self.compute_reward()
        terminated = self.terminate_episode()
        truncated = self.truncate_episode()
        return ob, reward, terminated, truncated, info

    def set_start(self, info_dict, return_info=True):
        """Teleport the scene to a start configuration (ignores locks).

        Entities in `info_dict` are set exactly to their requested values.
        All other entities are randomized, and their goal is pinned to their
        (random) current state so start == goal for them and they never block
        success. Pair with `set_goal` for evaluation.
        """
        # Parse requested updates.
        targets = {}
        for obj in self.objects:
            value = self._object_state_from_info(obj, info_dict)
            if value is not None:
                targets[obj.name] = (obj, value)

        # Apply every requested state exactly (locks are ignored: this is a
        # reset-like teleport to a desired start configuration).
        for name, (obj, value) in targets.items():
            obj.set_state(self, value)

            joint_name = getattr(obj, "joint_name", None)
            if joint_name is not None:
                self._data.joint(joint_name).qvel[:] = 0.0

        # Entities not part of the start configuration: randomize and pin the
        # goal to the randomized current state (start == goal for them).
        for obj in self.objects:
            if obj.name in targets:
                continue
            if not (
                hasattr(obj, "_target_mocap_id")
                or hasattr(obj, "_target_val")
                or hasattr(obj, "_target_button_states")
            ):
                continue  # Passive containers (shelf/box).
            obj.randomize(self)
            joint_name = getattr(obj, "joint_name", None)
            if joint_name is not None:
                self._data.joint(joint_name).qvel[:] = 0.0
            self._pin_target_to_current(obj)

        self._apply_button_states()
        self._success = self._evaluate_success(self._compute_successes())

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def _set_current(self, obj, value):
        """Set only the *current* state of an object, leaving its goal untouched.

        Mirror of `_set_object_target` (goal only): free bodies -> joint qpos,
        articulated objects -> joint value, buttons -> `_cur_state`.
        """
        if hasattr(obj, "_target_mocap_id"):
            pos, quat = value
            self._data.joint(obj.joint_name).qpos[:3] = pos
            self._data.joint(obj.joint_name).qpos[3:] = quat
        elif hasattr(obj, "_target_val"):
            self._data.joint(obj.joint_name).qpos[0] = float(
                np.asarray(value).ravel()[0]
            )
        elif hasattr(obj, "_target_button_states"):
            obj._cur_state[0] = int(round(float(np.asarray(value).ravel()[0])))

    def _set_object_target(self, obj, value):
        """Set only the *goal* (target) of an object, leaving its current state.

        Free bodies: target mocap pose. Articulated objects: `_target_val`
        (plus the visual goal site where the object exposes `_set_site`).
        Buttons: `_target_button_states`.
        """
        if hasattr(obj, "_target_mocap_id"):
            pos, quat = value
            self._data.mocap_pos[obj._target_mocap_id] = pos
            self._data.mocap_quat[obj._target_mocap_id] = quat
        elif hasattr(obj, "_target_val"):
            val = float(np.asarray(value).ravel()[0])
            obj._target_val = val
            set_site = getattr(obj, "_set_site", None)
            if set_site is not None:
                set_site(self, val)
        elif hasattr(obj, "_target_button_states"):
            obj._target_button_states[0] = int(
                round(float(np.asarray(value).ravel()[0]))
            )

    def _pin_target_to_current(self, obj):
        """Set the object's goal to its current state (start == goal)."""
        if hasattr(obj, "_target_mocap_id"):
            pos = self._data.joint(obj.joint_name).qpos[:3].copy()
            quat = self._data.joint(obj.joint_name).qpos[3:].copy()
            self._data.mocap_pos[obj._target_mocap_id] = pos
            self._data.mocap_quat[obj._target_mocap_id] = quat
        elif hasattr(obj, "_target_val"):
            self._set_object_target(obj, self._data.joint(obj.joint_name).qpos[0])
        elif hasattr(obj, "_target_button_states"):
            obj._target_button_states[0] = int(obj._cur_state[0])

    def set_goal(self, info_dict, return_info=True):
        """Set only the goal (target) state of the entities in `info_dict`.

        The current scene is untouched — only each object's goal changes
        (target mocap pose, `_target_val`, or `_target_button_states`).
        Combine with `set_start`: after the model reaches the goals,
        `info["success"]` / `compute_reward()` return True. The goal
        observation (`reset_info["goal"]`) is recomputed accordingly.
        """
        targets = {}
        for obj in self.objects:
            value = self._object_state_from_info(obj, info_dict, use_target_keys=True)
            if value is None:
                value = self._object_state_from_info(obj, info_dict)
            if value is not None:
                targets[obj.name] = (obj, value)

        # Set only the goals (targets), never the current state.
        for name, (obj, value) in targets.items():
            self._set_object_target(obj, value)

        # Recompute the goal observation with all goal objects at their goals.
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        saved_cur_states = {}
        for obj in self.objects:
            if hasattr(obj, "_cur_state"):
                saved_cur_states[obj.name] = obj._cur_state.copy()

        for name, (obj, value) in targets.items():
            self._set_current(obj, value)
        self._apply_button_states()
        self._cur_goal_ob = (
            self.compute_oracle_observation()
            if self._use_oracle_rep
            else self.compute_ob_info()
        )
        self._cur_goal_rendered = (
            self.get_pixel_observation() if self._render_goal else None
        )

        self._data.qpos[:] = saved_qpos
        self._data.qvel[:] = saved_qvel
        for obj in self.objects:
            if obj.name in saved_cur_states:
                obj._cur_state[:] = saved_cur_states[obj.name]
        self._apply_button_states()

        self._success = self._evaluate_success(self._compute_successes())

        if return_info:
            return self.compute_observation(), self.get_reset_info()

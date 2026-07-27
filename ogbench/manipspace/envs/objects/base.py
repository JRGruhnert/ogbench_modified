"""Base class for all scene objects."""


class SceneObject:
    """Every scene object must implement this interface.

    Each object is fully self-contained: it knows its XML, its joints, how to
    compute success, what observations to contribute, etc.
    """

    # -- Class-level config (overridden by subclasses) --
    xml_file: str = ""           # XML filename in descriptions/
    var_prefix: str = ""         # Used for privileged_* keys
    is_free_body: bool = False   # Has a freejoint?
    has_target: bool = True      # Shows a mocap target?

    def __init__(self, instance_id=0, pos=None, euler=None):
        self.instance_id = instance_id
        self.pos = pos
        self.euler = euler

    # -- Naming helpers --------------------------------------------------

    def _suffix(self, name):
        return f"{name}_{self.instance_id}" if self.instance_id > 0 else name

    # -- XML loading ------------------------------------------------------

    def load(self, arena_mjcf, desc_dir):
        """Load this object's XML into the arena. Return geom refs dict (or None)."""
        from dm_control import mjcf
        self._mjcf = mjcf.from_path((desc_dir / self.xml_file).as_posix())
        arena_mjcf.include_copy(self._mjcf)
        return None  # subclasses return {name: [geom_list], ...}

    def post_compilation(self, env):
        """Resolve model IDs after compilation."""
        pass

    # -- State -------------------------------------------------------------

    def randomize(self, env):
        """Randomize this object's state (data_collection mode)."""
        pass

    def init_to_goal(self, env, task_info):
        """Set this object to the goal state from task_info."""
        pass

    def init_to_init(self, env, task_info):
        """Set this object to the init state from task_info (with jitter)."""
        pass

    # -- Success -----------------------------------------------------------

    def compute_success(self, env):
        """Return (success_bool, task_type_string) or None if not applicable."""
        return None

    # -- Observations -------------------------------------------------------

    def get_info(self, env):
        """Return dict of privileged_{name}_* key-value pairs."""
        return {}

    def get_target_info(self, env):
        """Return dict of privileged_target_{name}_* key-value pairs (data_collection)."""
        return {}

    def add_observation(self, env, ob, ob_info):
        """Append observation entries to the ob list."""
        pass

    def add_oracle_obs(self, env, ob, ob_info):
        """Append oracle observation entries to the ob list."""
        pass

    # -- Task generation -----------------------------------------------------

    def get_task_probability(self, env):
        """Return probability for this object's task type (data_collection)."""
        return 1.0

    def handle_target(self, env):
        """Set a new random target for this object (data_collection)."""
        pass

    def get_target_from_task(self, task_info):
        """Extract this object's target value from a task dict."""
        return None

    # -- Button locking ------------------------------------------------------

    def apply_lock(self, env, button_states, button_locks):
        """Apply button-based locking. Called by _apply_button_states."""
        pass

    # -- Free-body support ---------------------------------------------------

    def get_body_name(self, i=0):
        """Return the body name for free-body instance i."""
        return None

    def get_joint_name(self, i=0):
        """Return the free joint name for instance i."""
        return None

    def get_target_body_name(self, i=0):
        """Return the mocap target body name for instance i."""
        return None

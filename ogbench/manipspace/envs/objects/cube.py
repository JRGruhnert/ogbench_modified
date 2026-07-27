class CubeObject:
    """Encapsulates the Cube (free-body) object for use in any scene environment.

    Cubes are free bodies that can be picked up and moved.  Each cube has
    a corresponding target mocap body for goal specification.
    """

    # -- XML and file info --
    xml_file = "cube.xml"
    var_prefix = "cube"

    # -- Not a joint object --
    is_joint_object = False

    # -- Free-body config --
    is_free_body = True
    free_joint_name = "object_joint_{i}"
    body_name = "object_{i}"
    target_body_name = "object_target_{i}"

    # -- Position override (settable by scene) --
    pos = None
    euler = None

    def __init__(self, count=1):
        self.count = count

    # -- State methods --
    @staticmethod
    def is_closed(env):
        """Cubes don't have a closed/open concept."""
        return True

    @staticmethod
    def get_state(env):
        """Return 0 (no binary state for cubes)."""
        return 0

    @staticmethod
    def get_target_value(env):
        """No joint target value for cubes."""
        return None

    @staticmethod
    def set_target_in_model(env, val):
        """No model update needed for cubes."""
        pass

    @staticmethod
    def target_site_pos(env, name):
        """Where to place the cube in the scene."""
        return None

    @staticmethod
    def get_task_probability(env, button_locks):
        """Cubes are always available as a task (availability checked at
        runtime in ``set_new_target``)."""
        return "cube", 1.0

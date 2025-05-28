import cupy as cp

class SceneObject:
    """Interface for objects in the wave simulation scene."""

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        """Modify the wave speed or dampening fields in-place."""
        pass

    def update_field(self, field: cp.ndarray, t: float):
        """Inject or modify the wave field in-place."""
        pass

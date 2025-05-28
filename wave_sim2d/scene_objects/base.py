import cupy as cp


class SceneObject:
    """Base interface for objects in the simulation scene."""

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        """Modify wave speed or dampening fields in-place."""
        pass

    def update_field(self, field: cp.ndarray, t: float):
        """Inject or modify the wave field."""
        pass

from enum import Enum

class BoundaryCondition(Enum):
    REFLECTIVE = "reflective"
    PERIODIC = "periodic"
    ABSORBING = "absorbing"

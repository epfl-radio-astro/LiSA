from enum import Enum

class InputMode(Enum):
    THREEDEE = 1
    TWODEE = 2
    SPREAD = 3

class AugmentMode(Enum):
    OFF = 0
    ALL = 1
    FAST = 2
    TRAIN = 3
    TEST = 4
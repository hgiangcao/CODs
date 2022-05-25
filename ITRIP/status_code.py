from enum import Enum


class StatusCode():
    CODsE_SUCC = 0
    CODsE_FAIL = 1
    IK_FAILURE = 2
    CONTROL_FAILURE = 3
    COLLISION = 4
    PICK_ON_SURFACE = 5
    CODsE_EXCEEDED_MAX_ACTION = 6
    CODsE_FINISH = 7
    CODsE_NONE = 8
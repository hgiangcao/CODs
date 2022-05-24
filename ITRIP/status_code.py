from enum import Enum


class StatusCode():
    DONE_SUCC = 0
    DONE_FAIL = 1
    IK_FAILURE = 2
    CONTROL_FAILURE = 3
    COLLISION = 4
    PICK_ON_SURFACE = 5
    DONE_EXCEEDED_MAX_ACTION = 6
    DONE_FINISH = 7
    DONE_NONE = 8
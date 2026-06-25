from typing import Dict

import tensorflow as tf

from openx.data.utils import RobotType, StateEncoding

# observation.state layout  (32 dims total):
#   [0:18]   joint positions (18 joints, both arms)
#   [18:25]  right wrist pose (xyz + qxqyqzqw)
#   [25:32]  left  wrist pose (xyz + qxqyqzqw)

# action layout  (36 dims total):
#   [0:16]   arm_cmd  (joint-space commands, both arms)
#   [16]     gripper_left
#   [17]     gripper_right
#   [18:34]  hand_angle (16 finger joints)
#   [34:36]  head_angle (2 DOF head)


def manav_dataset_transform(ep: Dict) -> Dict:
    """
    Standardise a raw LeRobot episode from the manav_dual_arm robot.

    Input keys (produced by openx.data.lerobot._read_episode):
        "observation.state"              float32[T, 32]
        "action"                         float32[T, 36]
        "observation.images.cam_head"    string[T]   (JPEG bytes, 256×256)
        "is_first"                       bool[T]
        "is_last"                        bool[T]

    Output follows the standard openx episode structure consumed by
    transforms.concatenate / transforms.chunk / transforms.decode_and_augment.
    """
    state = ep["observation.state"]   # [T, 32]
    action = ep["action"]             # [T, 36]

    observation = {
        "state": {
            StateEncoding.JOINT_POS: state[:, :18],          # 18 joint positions
            StateEncoding.MISC:      state[:, 18:],           # 14-dim wrist poses
        },
        "image": {
            "agent": ep["observation.images.cam_head"],       # JPEG bytes [T]
        },
    }

    structured_action = {
        "desired_absolute": {
            StateEncoding.JOINT_POS: action[:, :16],          # arm commands
            StateEncoding.GRIPPER:   action[:, 16:18],        # left + right gripper
            StateEncoding.MISC:      action[:, 18:],          # hand + head angles
        },
    }

    return {
        "observation": observation,
        "action":      structured_action,
        "is_first":    ep["is_first"],
        "is_last":     ep["is_last"],
        "robot":       RobotType.UNKNOWN,
    }

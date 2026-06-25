from typing import Dict

import tensorflow as tf

from openx.data.utils import RobotType, StateEncoding


def manav_relative_dataset_transform(ep: Dict) -> Dict:
    """
    Same as manav_dataset_transform but with relative (delta) actions.
    Expects episode_data_relative/ where actions are already pre-converted
    to deltas (action[t] - action[t-1], zeros at first step).
    """
    state = ep["observation.state"]   # [T, 32]
    action = ep["action"]             # [T, 36]

    observation = {
        "state": {
            StateEncoding.JOINT_POS: state[:, :18],
            StateEncoding.MISC:      state[:, 18:],
        },
        "image": {
            "agent": ep["observation.images.cam_head"],
        },
    }

    structured_action = {
        "desired_delta": {
            StateEncoding.JOINT_POS: action[:, :16],
            StateEncoding.GRIPPER:   action[:, 16:18],
            StateEncoding.MISC:      action[:, 18:],
        },
    }

    return {
        "observation": observation,
        "action":      structured_action,
        "is_first":    ep["is_first"],
        "is_last":     ep["is_last"],
        "robot":       RobotType.UNKNOWN,
    }

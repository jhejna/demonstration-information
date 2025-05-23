import json
import os
from typing import Dict, Optional

import gymnasium as gym
import h5py
import numpy as np
import tensorflow as tf
from robomimic.utils import env_utils

from openx.data.utils import StateEncoding

OBJECT_STATE_SIZE = 44  # Set to the max size across robomimic envs. ToolHang is giant.


class RobomimicEnv(gym.Env):
    def __init__(
        self,
        path: str,
        terminate_early: bool = False,
        use_image_obs: Optional[bool] = None,
        horizon: Optional[int] = 500,
    ):
        super().__init__()
        # Copy env meta code to allow reading from gcp file systems
        path = os.path.expanduser(path)
        with h5py.File(tf.io.gfile.GFile(path, "rb"), "r") as f:
            env_meta = json.loads(f["data"].attrs["env_args"])
        self.use_image_obs = use_image_obs if use_image_obs is not None else env_meta["env_kwargs"]["use_camera_obs"]
        self.env = env_utils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=False,
            use_image_obs=self.use_image_obs,
        ).env
        self.env.ignore_done = False
        if horizon is not None:
            self.env.horizon = horizon
        self.env._max_episode_steps = self.env.horizon
        self.terminate_early = terminate_early

        observation_spaces = dict(
            state=gym.spaces.Dict(
                {
                    StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=-np.inf, high=np.inf, dtype=np.float32),
                    StateEncoding.EE_QUAT: gym.spaces.Box(shape=(4,), low=-np.inf, high=np.inf, dtype=np.float32),
                    StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32),
                    StateEncoding.JOINT_POS: gym.spaces.Box(shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32),
                    StateEncoding.JOINT_VEL: gym.spaces.Box(shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32),
                    StateEncoding.MISC: gym.spaces.Box(
                        shape=(OBJECT_STATE_SIZE,), low=-np.inf, high=np.inf, dtype=np.float32
                    ),
                }
            ),
        )

        if self.use_image_obs:
            observation_spaces["image"] = gym.spaces.Dict(
                dict(
                    agent=gym.spaces.Box(shape=(84, 84, 3), dtype=np.uint8, low=0, high=255),
                    wrist=gym.spaces.Box(shape=(84, 84, 3), dtype=np.uint8, low=0, high=255),
                )
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)
        low, high = self.env.action_spec
        self.action_space = gym.spaces.Dict(
            dict(
                desired_delta=gym.spaces.Dict(
                    {
                        StateEncoding.EE_POS: gym.spaces.Box(shape=(3,), low=low[:3], high=high[:3], dtype=np.float32),
                        StateEncoding.EE_EULER: gym.spaces.Box(
                            shape=(3,), low=low[3:6], high=high[3:6], dtype=np.float32
                        ),
                    }
                ),
                desired_absolute=gym.spaces.Dict(
                    {StateEncoding.GRIPPER: gym.spaces.Box(shape=(1,), low=low[-1:], high=high[-1:], dtype=np.float32)}
                ),
            )
        )

    def _format_obs(self, obs):
        obj_state = np.zeros(OBJECT_STATE_SIZE, dtype=np.float32)
        obj_state[: obs["object-state"].shape[0]] = obs["object-state"]
        new_obs = dict(
            state={
                StateEncoding.EE_POS: obs["robot0_eef_pos"],
                StateEncoding.EE_QUAT: obs["robot0_eef_quat"],
                StateEncoding.GRIPPER: obs["robot0_gripper_qpos"][..., :1],
                StateEncoding.JOINT_POS: obs["robot0_joint_pos"],
                StateEncoding.JOINT_VEL: obs["robot0_joint_vel"],
                StateEncoding.MISC: obj_state,
            },
        )
        if self.use_image_obs:
            new_obs["image"] = dict(
                agent=np.flip(obs["agentview_image"], 0), wrist=np.flip(obs["robot0_eye_in_hand_image"], 0)
            )
        return new_obs

    def step(self, action: Dict):
        # For now only allow control via the specific action space we care about.
        action = np.concatenate(
            (
                action["desired_delta"][StateEncoding.EE_POS],
                action["desired_delta"][StateEncoding.EE_EULER],
                action["desired_absolute"][StateEncoding.GRIPPER],
            ),
            axis=-1,
        )
        low, high = self.env.action_spec
        action = np.clip(action, a_min=low, a_max=high)
        obs, reward, done, info = self.env.step(action)
        success = self.env._check_success()
        info["success"] = success
        if self.terminate_early and success:
            done = True
        # Never terminate robot envs, but do truncate them.
        return self._format_obs(obs), reward, False, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        return self._format_obs(obs), dict()

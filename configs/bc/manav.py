"""
Training config for the manav_dual_arm LeRobot dataset (parquet + mp4).

Usage:
    python scripts/train.py \
        --config configs/bc/manav.py:default \
        --path save/manav_bc \
        --name manav_bc_run1
"""

import optax
import tensorflow as tf
from ml_collections import ConfigDict

from openx.algs.bc import BehaviorCloning
from openx.data.datasets.manav import manav_dataset_transform
from openx.data.utils import NormalizationType, StateEncoding
from openx.networks.action_heads.continuous import L2ActionHead
from openx.networks.components.mlp import MLP
from openx.networks.components.resnet import ResNet18
from openx.networks.core import Concatenate, MultiEncoder
from openx.utils.spec import ModuleSpec

# Dataset runs at 5 fps after 6x subsampling.
# State  : 32 dims  (18 joint pos  + 14 wrist poses)
# Action : 36 dims  (16 arm cmd + 2 gripper + 16 hand + 2 head)

DATASET_PATH = "episode_data"


def get_config(config_str: str = "default"):
    structure = {
        "observation": {
            "state": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,  # 18 joint pos
                StateEncoding.MISC:      NormalizationType.GAUSSIAN,  # 14 wrist poses
            },
            "image": {
                "agent": (256, 256),   # cam_head, centre-cropped + resized in lerobot.py
            },
        },
        "action": {
            "desired_absolute": {
                StateEncoding.JOINT_POS: NormalizationType.GAUSSIAN,  # 16 arm cmds
                StateEncoding.GRIPPER:   NormalizationType.BOUNDS,    # 2 gripper dims
                StateEncoding.MISC:      NormalizationType.GAUSSIAN,  # 18 hand+head
            },
        },
    }

    dataloader = dict(
        datasets={
            "manav": dict(
                path=DATASET_PATH,
                train_split="train",
                transform=ModuleSpec.create(manav_dataset_transform),
            ),
        },
        n_obs=1,
        n_action=5,           # 1-second action horizon @ 5 fps
        augment_kwargs=dict(
            scale_range=(0.85, 1.0),
            brightness=0.1,
            contrast_range=(0.9, 1.1),
        ),
        shuffle_size=5000,
        batch_size=64,
        recompute_statistics=False,
        prefetch=tf.data.AUTOTUNE,
    )

    alg = ModuleSpec.create(
        BehaviorCloning,
        observation_encoder=ModuleSpec.create(
            MultiEncoder,
            encoders={
                "observation->image->agent": ModuleSpec.create(ResNet18, num_kp=64),
                "observation->state":        None,
            },
            trunk=ModuleSpec.create(
                Concatenate,
                model=ModuleSpec.create(
                    MLP, [512, 512, 512], dropout_rate=0.1, activate_final=True
                ),
                flatten_time=True,
            ),
        ),
        action_head=ModuleSpec.create(
            L2ActionHead,
            model=None,
            action_dim=36,   # 16 + 2 + 18
            action_horizon=5,
        ),
    )

    lr_schedule = ModuleSpec.create(
        optax.warmup_cosine_decay_schedule,
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=500,
        decay_steps=50000,
        end_value=1e-6,
    )
    optimizer = ModuleSpec.create(optax.adamw)

    return ConfigDict(
        dict(
            structure=structure,
            alg=alg,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            steps=50001,
            log_freq=500,
            val_freq=5000,
            save_freq=1000,
            val_steps=16,
            exec_horizon=1,
            seed=42,
        )
    )

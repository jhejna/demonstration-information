import functools
import hashlib
import json
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .transforms import normalize
from .utils import DataType, NormalizationType, StateEncoding
from .lerobot import is_lerobot_dataset, is_mcap_dataset, load_lerobot_dataset, load_mcap_dataset

"""
All Datasets are expected to follow the TFDS Format.
Then, we add some specific metadata, specifically ep_idx and step_idx.

This file contains functions that operate directly on dataset objects.
For general functions that should be mapped on to a dataset with dataset.map()
see transforms.py

Afterwards, we will pad the dataset to match the same thing:
{
    "observation": {
        "state": {
            StateEncoding.EE_POS:
            StateEncoding.EE_ROT6D:
        },
        "image": {
            "agent": [Img_0, Img_1], # Will randomly select one of these two for each episode.
            "wrist: [Img_0],
        },
    }
    "action": {
        "achieved_delta": [],
        "achieved_absolute": [StateEncoding.EE_POS, StateEncoding.EE_ROT6D],
        "desired_delta": [],
        "desired_absolute": [StateEncoding.GRIPPER],
    }
    "language_instruction" : "Instruction",
    "is_first": np.ndarray,
    "is_last": np.ndarray,
    "dataset_id": np.ndarray,
    "robot_id":
    "controller_hz":
}
"""

STANDARD_STRUCTURE = {
    "observation": {
        "state": {StateEncoding.EE_POS: NormalizationType.NONE, StateEncoding.EE_EULER: NormalizationType.NONE},
        "image": {"agent": np.zeros((224, 224), dtype=np.uint8), "wrist": np.zeros((256, 256), dtype=np.uint8)},
    },
    "action": {
        "desired_delta": {
            StateEncoding.EE_POS: NormalizationType.GAUSSIAN,
            StateEncoding.EE_ROT6D: NormalizationType.GAUSSIAN,
        },
        "desired_absolute": {StateEncoding.GRIPPER: NormalizationType.NONE},
    },
    "is_last": DataType.BOOL,
}


def filter_by_structure(tree, structure):
    if isinstance(structure, dict):
        return {k: filter_by_structure(tree[k], v) for k, v in structure.items()}
    return tree  # otherwise return the item from the tree (episode)


def filter_dataset_statistics_by_structure(dataset_statistics, structure):
    # Compute the final sa_structure we will use to filter the dataset_statistics
    # (which are computed on everything for efficiency)
    sa_structure = dict(action=structure["action"])
    if "state" in structure["observation"]:
        sa_structure["state"] = structure["observation"]["state"]
    # Filter each of the stat nested structures.
    for k in ["mean", "std", "min", "max"]:
        dataset_statistics[k] = filter_by_structure(dataset_statistics[k], sa_structure)
    return dataset_statistics


def load_dataset(
    path: str | List[str],
    split: str,
    standardization_transform: Callable,
    structure: Optional[dict] = None,
    dataset_statistics: Optional[Union[str, Dict]] = None,
    recompute_statistics: bool = False,
    num_parallel_reads: Optional[int] = tf.data.AUTOTUNE,
    num_parallel_calls: Optional[int] = tf.data.AUTOTUNE,
    shuffle: bool = True,
    filter_fn: Optional[Callable] = None,  # filter functions operate on the RAW dataset pre standardization
    minimum_length: int = 3,
):
    """
    This function loads a dataset from a path and standardizes it.

    If you pass in a structure, it garuntees that the resulting dataset will follow that exact structure.

    Supports three on-disk formats, detected automatically from *path*:
      - LeRobot  (parquet + mp4)  -- path has meta/info.json
      - MCAP     (ROS 2 bags)     -- path contains *.mcap files
      - RLDS/TFDS (default)       -- everything else
    """
    # ------------------------------------------------------------------
    # LeRobot format  (parquet + mp4)
    # ------------------------------------------------------------------
    if isinstance(path, str) and is_lerobot_dataset(path):
        return load_lerobot_dataset(
            path=path,
            split=split,
            standardization_transform=standardization_transform,
            structure=structure,
            dataset_statistics=dataset_statistics,
            recompute_statistics=recompute_statistics,
            num_parallel_calls=num_parallel_calls,
            shuffle=shuffle,
            filter_fn=filter_fn,
            minimum_length=minimum_length,
        )

    # ------------------------------------------------------------------
    # MCAP format  (ROS 2 bag files)
    # ------------------------------------------------------------------
    if isinstance(path, str) and is_mcap_dataset(path):
        return load_mcap_dataset(
            path=path,
            split=split,
            standardization_transform=standardization_transform,
            structure=structure,
            dataset_statistics=dataset_statistics,
            num_parallel_calls=num_parallel_calls,
            shuffle=shuffle,
            filter_fn=filter_fn,
            minimum_length=minimum_length,
        )

    # ------------------------------------------------------------------
    # RLDS / TFDS format  (default)
    # ------------------------------------------------------------------
    if isinstance(path, list):
        builder = tfds.builder_from_directories(builder_dir=path)
    else:
        builder = tfds.builder_from_directory(builder_dir=path)
    dataset = builder.as_dataset(
        split=split,
        decoders=dict(steps=tfds.decode.SkipDecoding()),
        shuffle_files=shuffle,
        read_config=tfds.ReadConfig(
            skip_prefetch=True,
            num_parallel_calls_for_interleave_files=num_parallel_reads,
            interleave_cycle_length=num_parallel_reads,
            shuffle_reshuffle_each_iteration=shuffle,
        ),
    )
    options = tf.data.Options()
    options.autotune.enabled = True
    options.deterministic = not shuffle
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    options.experimental_optimization.inject_prefetch = False
    options.experimental_warm_start = True
    dataset = dataset.with_options(options)
    # filter the dataset according to the filter function BEFORE we do anything else.
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn())

    # Determine if we have dataset statistics
    if structure is not None:
        state_keys = tf.nest.flatten(structure["observation"].get("state", NormalizationType.NONE))
        action_keys = tf.nest.flatten(structure["action"])
        if any(norm_type != NormalizationType.NONE for norm_type in state_keys + action_keys):
            if dataset_statistics is None:
                dataset_statistics = compute_dataset_statistics(
                    path, standardization_transform, recompute_statistics=recompute_statistics
                )
            elif isinstance(dataset_statistics, str):
                dataset_statistics = load_dataset_statistics(dataset_statistics)
            dataset_statistics = filter_dataset_statistics_by_structure(dataset_statistics, structure)
        else:
            dataset_statistics = None

    def _standardize(ep):
        # Merge multiple tf operations to allow for graph optimization.
        # Calling metadata, standardization, and structure lets us eliminate calls note used by the final ds.

        steps = ep.pop("steps")
        assert "is_first" in steps
        assert "is_last" in steps
        ep_len = tf.shape(steps["is_first"])[0]

        # Broadcast metadata
        if "episode_metadata" in ep:
            metadata = tf.nest.map_structure(lambda x: tf.repeat(x, ep_len), ep["episode_metadata"])
            steps["episode_metadata"] = metadata

        steps = standardization_transform(steps)  # Standardize the episode

        if structure is not None:
            steps = filter_by_structure(steps, structure)  # Filter down to the keys in structure
            if dataset_statistics is not None:
                steps = normalize(steps, structure, dataset_statistics)

        # Reduce image keys to a single image if multiple are present.
        if "observation" in steps and "image" in steps["observation"]:
            multi_image_keys = []
            for k in steps["observation"]["image"]:
                if isinstance(steps["observation"]["image"][k], list):
                    multi_image_keys.append(k)
            for k in multi_image_keys:
                imgs = steps["observation"]["image"][k]
                steps["observation"]["image"][k] = imgs[tf.random.uniform((), minval=0, maxval=len(imgs))]

        return steps

    dataset = dataset.map(_standardize, num_parallel_calls=num_parallel_calls, deterministic=not shuffle)
    # Filter out episodes that are too short.
    dataset = dataset.filter(lambda ep: tf.shape(tf.nest.flatten(ep["action"])[0])[0] >= minimum_length)

    # TODO(jhejna): expand checks.
    element_spec = dataset.element_spec
    assert "observation" in element_spec
    assert "action" in element_spec

    return dataset, dataset_statistics


def load_dataset_statistics(path):
    if not path.endswith(".json"):
        path = tf.io.gfile.join(path, "dataset_statistics.json")
    with tf.io.gfile.GFile(path, "r") as f:
        dataset_statistics = json.load(f)

    # Convert everything to numpy
    def _convert_to_numpy(x):
        return {k: _convert_to_numpy(v) if isinstance(v, dict) else np.array(v, dtype=np.float32) for k, v in x.items()}

    return _convert_to_numpy(dataset_statistics)


def compute_dataset_statistics(
    path: str | List[str],
    standardization_transform: Callable,
    recompute_statistics: bool = False,
    save_statistics: bool = True,
):
    # Compute some hash of the path and other factors to determine the path.
    storage_path = sorted(path)[0] if isinstance(path, list) else path
    hash_deps = "".join(sorted(path) if isinstance(path, list) else [path])
    if isinstance(standardization_transform, functools.partial):
        hash_deps += standardization_transform.func.__name__
        hash_deps += ", ".join(standardization_transform.args)
        hash_deps += ", ".join(f"{k}={v}" for k, v in standardization_transform.keywords.items())
    else:
        hash_deps += standardization_transform.__name__
    unique_hash = hashlib.sha256(hash_deps.encode("utf-8"), usedforsecurity=False).hexdigest()

    # See if we need to compute the dataset statistics
    dataset_statistics_path = tf.io.gfile.join(storage_path, f"dataset_statistics_{unique_hash}.json")
    if not recompute_statistics and tf.io.gfile.exists(dataset_statistics_path):
        dataset_statistics = load_dataset_statistics(dataset_statistics_path)
    else:
        # Otherwise, load the dataset to compute the statistics, let tf data handle the parallelization
        dataset, _ = load_dataset(
            path,
            split="all",
            standardization_transform=standardization_transform,
            structure=None,
            dataset_statistics=None,
        )
        sa_elem_spec = dict(action=dataset.element_spec["action"], state=dataset.element_spec["observation"]["state"])
        initial_state = dict(
            num_steps=0,
            num_ep=0,
            mean=tf.nest.map_structure(lambda x: tf.zeros(x.shape[1:], dtype=np.float32), sa_elem_spec),
            var=tf.nest.map_structure(lambda x: 1e-5 * tf.ones(x.shape[1:], dtype=np.float32), sa_elem_spec),
            min=tf.nest.map_structure(lambda x: 1e10 * tf.ones(x.shape[1:], dtype=np.float32), sa_elem_spec),
            max=tf.nest.map_structure(lambda x: -1e10 * tf.ones(x.shape[1:], dtype=np.float32), sa_elem_spec),
        )

        def _reduce_fn(old_state, ep):
            # This uses a streaming algorithm to efficiently compute dataset statistics
            # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            ep = dict(action=ep["action"], state=ep["observation"]["state"])
            tf.nest.assert_same_structure(old_state["mean"], ep)  # Check we can flatten before doing

            batch_count = tf.shape(tf.nest.flatten(ep)[0])[0] - 1  # Reduce by 1 for last transition
            batch_mean = tf.nest.map_structure(lambda x: tf.reduce_mean(x[:-1], axis=0), ep)
            batch_var = tf.nest.map_structure(lambda x: tf.math.reduce_variance(x[:-1], axis=0), ep)

            count_f, batch_count_f = (
                tf.cast(old_state["num_steps"], dtype=np.float32),
                tf.cast(batch_count, dtype=np.float32),
            )
            total_count_f = count_f + batch_count_f

            delta = tf.nest.map_structure(lambda m, m_b: m_b - m, old_state["mean"], batch_mean)
            new_mean = tf.nest.map_structure(
                lambda mean, delta: mean + delta * batch_count_f / total_count_f, old_state["mean"], delta
            )
            new_m2 = tf.nest.map_structure(
                lambda var, var_b, d: var * count_f
                + var_b * batch_count_f
                + tf.square(d) * count_f * batch_count_f / total_count_f,
                old_state["var"],
                batch_var,
                delta,
            )
            new_var = tf.nest.map_structure(lambda m2: m2 / total_count_f, new_m2)

            # Return updated values
            return dict(
                num_steps=old_state["num_steps"] + batch_count,
                num_ep=old_state["num_ep"] + 1,
                mean=new_mean,
                var=new_var,
                min=tf.nest.map_structure(
                    lambda x, m: tf.minimum(tf.reduce_min(x[:-1], axis=0), m), ep, old_state["min"]
                ),
                max=tf.nest.map_structure(
                    lambda x, m: tf.maximum(tf.reduce_max(x[:-1], axis=0), m), ep, old_state["max"]
                ),
            )

        dataset_statistics = dataset.reduce(initial_state, _reduce_fn)
        dataset_statistics["std"] = tf.nest.map_structure(lambda x: tf.math.sqrt(x), dataset_statistics["var"])
        del dataset_statistics["var"]
        dataset_statistics = tf.nest.map_structure(lambda x: x.numpy(), dataset_statistics)

        # Now re-organize the dataset statistics to be the following:
        # dict(num_ep, num_steps, state=dict(), action=di)
        dataset_statistics["num_ep"] = int(dataset_statistics["num_ep"])
        dataset_statistics["num_steps"] = int(dataset_statistics["num_steps"])

        # Save the dataset statistics
        if save_statistics:
            list_dset_stats = tf.nest.map_structure(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x, dataset_statistics
            )
            with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
                json.dump(list_dset_stats, f, default=float, indent=4)

    return dataset_statistics

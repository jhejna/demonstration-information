import functools
from typing import Dict, Optional

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from openx.utils.spec import ModuleSpec

from . import transforms
from .core import STANDARD_STRUCTURE, load_dataset
from .lerobot import is_lerobot_dataset, is_mcap_dataset


def _split_for_jax_process(split: str) -> str:
    """Shard a split string across JAX processes without relying on tfds."""
    n = jax.process_count()
    i = jax.process_index()
    if n == 1:
        return split
    # Append a percent-slice that _parse_split in lerobot.py understands,
    # and that tfds also accepts for RLDS datasets.
    lo = i * 100 // n
    hi = (i + 1) * 100 // n
    return f"{split}[{lo}%:{hi}%]"

VAL_PARALLEL_CALLS = 1


def make_dataloader(
    datasets: Dict,
    structure: Dict = STANDARD_STRUCTURE,
    n_obs: int = 1,
    n_action: int = 1,
    goal_conditioning: str | None = None,
    n_step: int | None = None,
    add_initial_observation: bool = False,
    augment_kwargs: Optional[Dict] = None,
    batch_size: int = 256,
    shuffle_size: int = 10000,
    discard_fraction: float = 0.0,
    repeat: bool | int = True,
    cache: bool = False,
    repeat_early: bool = False,
    recompute_statistics: bool = False,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    use_parallel_flatten: bool = True,
    prefetch: int = 0,
    split_for_jax: bool = True,
    restrict_memory: bool = False,
):
    # Get all datasets
    train_datasets = dict()
    val_datasets = dict()
    weights = dict()
    dataset_statistics = dict()
    # Parse out the repeat count
    repeat_count = None if isinstance(repeat, bool) else repeat
    repeat = repeat if isinstance(repeat, bool) else True

    # If Cache option is set, ensure valid parameters
    if cache:
        repeat_early = False
        discard_fraction = 0.0

    # Loop through all datasets to construct dataloader
    for ds_name, ds_config in datasets.items():
        assert "path" in ds_config and "transform" in ds_config
        assert "train_split" in ds_config or "val_split" in ds_config
        path = ds_config["path"]
        transform_fn = ModuleSpec.instantiate(ds_config["transform"])
        dataset_statistics_path = ds_config.get("dataset_statistics")

        # Add train split
        if ds_config.get("train_split"):
            split = ds_config["train_split"]
            if split_for_jax:
                if is_lerobot_dataset(path) or is_mcap_dataset(path):
                    split = _split_for_jax_process(split)
                else:
                    split = tfds.split_for_jax_process(split)
            ds, ds_stats = load_dataset(
                path,
                split,
                standardization_transform=transform_fn,
                structure=structure,
                dataset_statistics=dataset_statistics_path,
                recompute_statistics=recompute_statistics,
                filter_fn=ModuleSpec.instantiate(ds_config.get("train_filter", ds_config.get("filter"))),
                num_parallel_reads=ds_config.get("num_parallel_reads", num_parallel_reads),
                num_parallel_calls=ds_config.get("num_parallel_calls", num_parallel_calls),
                shuffle=shuffle_size > 0,
            )
            train_datasets[ds_name] = ds
            weights[ds_name] = ds_config.get("weight", 1.0)
            dataset_statistics[ds_name] = ds_stats

        # Add val split if present
        if ds_config.get("val_split"):
            # Val dataloading is the same except we limit the number of workers
            ds, ds_stats = load_dataset(
                path,
                ds_config["val_split"],  # Do not split validation set.
                standardization_transform=transform_fn,
                structure=structure,
                dataset_statistics=dataset_statistics_path,
                recompute_statistics=ds_config.get("train_split") is None and recompute_statistics,
                filter_fn=ModuleSpec.instantiate(ds_config.get("val_filter", ds_config.get("filter"))),
                num_parallel_reads=VAL_PARALLEL_CALLS,
                num_parallel_calls=VAL_PARALLEL_CALLS,
                shuffle=shuffle_size > 0,
            )
            # No weights are used for validation datasets.
            val_datasets[ds_name] = ds
            dataset_statistics[ds_name] = ds_stats

    if repeat and repeat_early:
        # Repeat here, otherwise will repeat with shuffling for fused op.
        train_datasets = {k: v.repeat(count=repeat_count) for k, v in train_datasets.items()}
        val_datasets = {k: v.repeat(count=repeat_count) for k, v in val_datasets.items()}

    # Next, chunk all of the datasets and dump steps if needed.
    # We'll also add the dataset ID here for fun.
    def _stepify(ep, dataset_id):
        ep = transforms.concatenate(ep)
        ep = transforms.add_dataset_id(ep, dataset_id)
        # Add goal conditioning first (no sequence)
        if goal_conditioning == "uniform":
            ep = transforms.uniform_goal_relabeling(ep)
        elif goal_conditioning == "last":
            ep = transforms.last_goal_relabeling(ep)
        # Add the initial observaiton if needed (also no sequence)
        if add_initial_observation:
            ep = transforms.add_initial_observation(ep)
        # Add sequences
        ep = transforms.chunk(ep, n_obs, n_action)
        # Add next observation if wanted, uses sequence
        if n_step is not None:
            ep = transforms.add_next_observation(ep, n_step)
        # cut the last transition -- its terminal.
        ep = tf.nest.map_structure(lambda x: x[:-1], ep)

        # Shuffle and discard.
        if discard_fraction > 0:
            ep_len = tf.shape(tf.nest.flatten(ep)[0])[0]
            num_to_keep = tf.maximum(tf.cast(ep_len, tf.float32) * (1 - discard_fraction), 1)
            num_to_keep = tf.cast(num_to_keep, tf.int32)
            idxs = tf.random.shuffle(tf.range(ep_len))[:num_to_keep]
            ep = tf.nest.map_structure(lambda x: tf.gather(x, idxs), ep)

        return ep

    dataset_ids = sorted(list(set(list(train_datasets.keys()) + list(val_datasets.keys()))))
    dataset_ids = {k: v for v, k in enumerate(dataset_ids)}

    train_datasets = {
        k: v.map(
            functools.partial(_stepify, dataset_id=dataset_ids[k]),
            num_parallel_calls=datasets[k].get("num_parallel_calls", num_parallel_calls),
            deterministic=shuffle_size > 0,
        )
        for k, v in train_datasets.items()
    }
    val_datasets = {
        k: v.map(
            functools.partial(_stepify, dataset_id=dataset_ids[k]),
            num_parallel_calls=VAL_PARALLEL_CALLS,
            deterministic=shuffle_size > 0,
        )
        for k, v in val_datasets.items()
    }

    # Now flatten the datasets.
    def _flatten_dataset(ds, num_parallel_calls):
        if use_parallel_flatten and shuffle_size > 0:
            return ds.interleave(
                lambda ep: tf.data.Dataset.from_tensor_slices(ep),
                cycle_length=num_parallel_calls,
                num_parallel_calls=num_parallel_calls,
                deterministic=False,
            )
        return ds.flat_map(tf.data.Dataset.from_tensor_slices)

    train_datasets = {
        k: _flatten_dataset(v, datasets[k].get("num_parallel_calls", num_parallel_calls))
        for k, v in train_datasets.items()
    }
    val_datasets = {k: _flatten_dataset(v, VAL_PARALLEL_CALLS) for k, v in val_datasets.items()}

    # Combine the train datasets into one dataset
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    print("\n######################################################################################")
    print(f"# Loading the following {len(train_datasets)} datasets (incl. sampling weight):{'': >24} #")
    for ds_name in train_datasets:
        pad = 80 - len(ds_name)
        print(f"# {ds_name}: {weights[ds_name]:=>{pad}f} #")
    print("######################################################################################\n")

    if len(train_datasets) > 1:
        order = sorted(list(train_datasets.keys()))
        train_dataset = tf.data.Dataset.sample_from_datasets(
            [train_datasets[p] for p in order],
            weights=[weights[p] for p in order],
            stop_on_empty_dataset=repeat and not cache,  # only stop if we are repeating.
            rerandomize_each_iteration=shuffle_size > 0,
        )
    else:
        train_dataset = train_datasets[next(iter(train_datasets.keys()))]

    if cache:
        train_dataset = train_dataset.cache()
        val_datasets = {k: v.cache() for k, v in val_datasets.items()}

    # Shuffle the datasets
    if shuffle_size > 0:
        train_dataset = train_dataset.shuffle(shuffle_size)
        # The val shuffle size is automatically set to 1/10 that of the train set.
        val_datasets = {
            k: v.shuffle(
                max(1, int(shuffle_size * weights.get(k, 1 / len(val_datasets)) // 10)),
            )
            for k, v in val_datasets.items()
        }

    if repeat and not repeat_early:
        train_dataset = train_dataset.repeat(count=repeat_count)
        val_datasets = {k: v.repeat(count=repeat_count) for k, v in val_datasets.items()}

    # Decode and augment the images
    augment_kwargs = dict() if augment_kwargs is None else augment_kwargs
    _decode_and_augment = functools.partial(transforms.decode_and_augment, structure=structure, **augment_kwargs)
    train_dataset = train_dataset.map(
        functools.partial(_decode_and_augment, train=True),
        num_parallel_calls=num_parallel_calls,
        deterministic=shuffle_size > 0,
    )
    val_datasets = {
        k: v.map(
            functools.partial(_decode_and_augment, train=False),
            num_parallel_calls=2 * VAL_PARALLEL_CALLS,
            deterministic=shuffle_size > 0,
        )
        for k, v in val_datasets.items()
    }

    # Finally, batch the datasets
    train_dataset = train_dataset.batch(batch_size, num_parallel_calls=None, drop_remainder=True)
    val_datasets = {
        k: v.batch(batch_size, num_parallel_calls=None, drop_remainder=True) for k, v in val_datasets.items()
    }

    # Then, add memory limits for autotune.
    if restrict_memory:
        train_options = tf.data.Options()
        train_options.autotune.ram_budget = int(4 * 1024 * 1024 * 1024)  # GB -> Bytes
        train_dataset = train_dataset.with_options(train_options)
        val_options = tf.data.Options()
        val_options.autotune.ram_budget = int(1 * 1024 * 1024 * 1024)  # GB -> Bytes
        val_datasets = {k: v.with_options(val_options) for k, v in val_datasets.items()}

    # finally add prefetch as desired
    train_dataset = train_dataset.prefetch(prefetch)
    return train_dataset, val_datasets, dataset_statistics, dataset_ids

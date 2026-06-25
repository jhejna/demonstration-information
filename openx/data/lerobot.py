"""
Loader for LeRobot-format datasets (parquet + mp4 videos).

LeRobot format layout:
    <dataset_root>/
        meta/
            info.json        -- dataset metadata, feature specs, split ranges
            episodes.jsonl   -- per-episode task/length records
            stats.json       -- pre-computed statistics
            tasks.jsonl      -- task descriptions
        data/
            chunk-000/
                episode_000000.parquet
                ...
        videos/
            chunk-000/
                <video_key>/
                    episode_000000.mp4
                    ...

Also provides a stub loader for MCAP (ROS 2 bag) format that converts to the
same internal representation.  Install `mcap-ros2-support` and `mcap` to use it:
    pip install mcap mcap-ros2-support
"""

import json
import os
import re
from typing import Callable, Dict, List, Optional, Union

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tensorflow as tf

from .utils import NormalizationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_lerobot_dataset(path: str) -> bool:
    """Return True when *path* looks like a LeRobot dataset root."""
    if not isinstance(path, str):
        return False
    return os.path.exists(os.path.join(path, "meta", "info.json"))


def _parse_split(split_str: str, info: dict) -> List[int]:
    """
    Parse a split descriptor into a list of episode indices.

    Accepted formats:
      "train"              -- named split from info["splits"]
      "all"                -- every episode
      "train[0%:50%]"      -- jax-process-sharded slice of a named split
      "0:142"              -- raw episode range (start inclusive, end exclusive)
    """
    named = info.get("splits", {})
    total = info["total_episodes"]

    # Match  name  or  name[slice]
    m = re.match(r"^(\w+)(\[[\d%:]+\])?$", split_str)
    if m:
        name, slice_str = m.group(1), m.group(2)
        if name in named:
            lo, hi = map(int, named[name].split(":"))
            indices = list(range(lo, hi))
        elif name == "all":
            indices = list(range(total))
        else:
            raise ValueError(f"Unknown split name {name!r}. Available: {list(named)}")

        if slice_str:
            m2 = re.match(r"\[(\d*)%?:(\d*)%?\]", slice_str)
            if m2:
                s, e = m2.group(1), m2.group(2)
                n = len(indices)
                s_idx = int(s) * n // 100 if s else 0
                e_idx = int(e) * n // 100 if e else n
                indices = indices[s_idx:e_idx]
        return indices

    # Bare "start:end" range
    if re.match(r"^\d+:\d+$", split_str):
        lo, hi = map(int, split_str.split(":"))
        return list(range(lo, hi))

    raise ValueError(f"Unrecognised split format: {split_str!r}")


def _read_episode(path: str, ep_idx: int, info: dict, video_keys: List[str]) -> dict:
    """Read one episode from disk and return a dict of numpy arrays."""
    chunk_idx = ep_idx // info["chunks_size"]

    parquet_path = os.path.join(
        path,
        info["data_path"].format(chunk_index=chunk_idx, episode_index=ep_idx),
    )
    df = pd.read_parquet(parquet_path)

    # Downsample from 30 fps to 5 fps
    df = df.iloc[::SUBSAMPLE_FACTOR].reset_index(drop=True)

    state = np.stack(df["observation.state"].values).astype(np.float32)
    action = np.stack(df["action"].values).astype(np.float32)
    is_last = df["next.done"].values.astype(bool)
    is_last[-1] = True  # guarantee the last kept frame ends the episode
    is_first = np.zeros(len(df), dtype=bool)
    is_first[0] = True
    frame_indices = df["frame_index"].values

    episode = {
        "observation.state": state,
        "action": action,
        "is_first": is_first,
        "is_last": is_last,
    }

    for vk in video_keys:
        video_path = os.path.join(
            path,
            info["video_path"].format(
                chunk_index=chunk_idx, video_key=vk, episode_index=ep_idx
            ),
        )
        frames = iio.imread(video_path, plugin="pyav")  # (T_vid, H, W, 3) uint8
        jpeg_frames = []
        for fi in frame_indices:
            frame = frames[fi] if fi < len(frames) else frames[-1]
            frame = _center_crop_and_resize(frame)  # HxH centre crop → 256×256
            jpeg = tf.image.encode_jpeg(frame, quality=95).numpy()
            jpeg_frames.append(jpeg)
        episode[vk] = jpeg_frames  # list of bytes

    return episode


# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------

SUBSAMPLE_FACTOR = 6        # 30 fps → 5 fps
OUTPUT_IMAGE_SIZE = 256     # pixels (square)


def _center_crop_and_resize(frame: np.ndarray, output_size: int = OUTPUT_IMAGE_SIZE) -> np.ndarray:
    """Crop to a centre square (min(H,W) × min(H,W)) then resize to output_size²."""
    h, w = frame.shape[0], frame.shape[1]
    crop = min(h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    frame = frame[y0:y0 + crop, x0:x0 + crop]
    resized = tf.image.resize(frame[None], [output_size, output_size], method="lanczos3")[0]
    return tf.cast(resized, tf.uint8).numpy()


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_lerobot_dataset(
    path: str,
    split: str,
    standardization_transform: Callable,
    structure: Optional[dict] = None,
    dataset_statistics: Optional[Union[str, Dict]] = None,
    recompute_statistics: bool = False,
    num_parallel_calls: Optional[int] = tf.data.AUTOTUNE,
    shuffle: bool = True,
    filter_fn: Optional[Callable] = None,
    minimum_length: int = 3,
):
    """
    Load a LeRobot dataset and return it in the same format as
    ``core.load_dataset``:  a ``tf.data.Dataset`` of standardised episode
    dicts plus the dataset-statistics dict.

    Each element of the returned dataset is a *standardised* episode dict
    (output of ``standardization_transform``) with per-step arrays along
    the first axis.
    """
    from .core import filter_dataset_statistics_by_structure, load_dataset_statistics
    from .transforms import normalize

    with open(os.path.join(path, "meta", "info.json")) as f:
        info = json.load(f)

    video_keys = ["observation.images.cam_head"]
    episode_indices = _parse_split(split, info)

    # Build output_signature so TF knows the shape/dtype before running the generator.
    state_dim = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]

    output_signature = {
        "observation.state": tf.TensorSpec(shape=(None, state_dim), dtype=tf.float32),
        "action": tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
        "is_first": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "is_last": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "ep_idx": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "quality_score": tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
    for vk in video_keys:
        output_signature[vk] = tf.TensorSpec(shape=(None,), dtype=tf.string)

    # -----------------------------------------------------------------------
    # Python generator – one yield per episode
    # -----------------------------------------------------------------------
    # We capture episode_indices in a closure; shuffling happens outside.
    if shuffle:
        np.random.shuffle(episode_indices)

    def _generator():
        for ep_idx in episode_indices:
            try:
                raw = _read_episode(path, ep_idx, info, video_keys)
            except Exception as exc:
                print(f"[lerobot] Warning: skipping episode {ep_idx}: {exc}")
                continue
            ep_len = len(raw["is_first"])
            # Convert to tensors matching output_signature
            ep_tf = {
                "observation.state": tf.constant(raw["observation.state"], dtype=tf.float32),
                "action": tf.constant(raw["action"], dtype=tf.float32),
                "is_first": tf.constant(raw["is_first"], dtype=tf.bool),
                "is_last": tf.constant(raw["is_last"], dtype=tf.bool),
                "ep_idx": tf.fill([ep_len], tf.constant(ep_idx, dtype=tf.int32)),
                "quality_score": tf.zeros([ep_len], dtype=tf.int32),
            }
            for vk in video_keys:
                ep_tf[vk] = tf.constant(raw[vk], dtype=tf.string)
            yield ep_tf

    dataset = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)

    # -----------------------------------------------------------------------
    # Filter (operates on raw episode dicts before standardisation)
    # -----------------------------------------------------------------------
    if filter_fn is not None:
        dataset = dataset.filter(filter_fn())

    # -----------------------------------------------------------------------
    # Standardisation transform
    # -----------------------------------------------------------------------
    def _standardize(ep):
        from .core import filter_by_structure
        ep_idx_val = ep.get("ep_idx")
        quality_score_val = ep.get("quality_score")
        ep = standardization_transform(ep)
        if ep_idx_val is not None:
            ep["ep_idx"] = ep_idx_val
        if quality_score_val is not None:
            ep["quality_score"] = quality_score_val
        if structure is not None:
            ep = filter_by_structure(ep, structure)
        return ep

    dataset = dataset.map(_standardize, num_parallel_calls=num_parallel_calls, deterministic=not shuffle)

    # Drop episodes that are too short
    dataset = dataset.filter(
        lambda ep: tf.shape(tf.nest.flatten(ep["action"])[0])[0] >= minimum_length
    )

    # -----------------------------------------------------------------------
    # Dataset statistics (used for normalisation)
    # -----------------------------------------------------------------------
    if structure is not None:
        state_keys = tf.nest.flatten(structure["observation"].get("state", NormalizationType.NONE))
        action_keys = tf.nest.flatten(structure["action"])
        needs_stats = any(n != NormalizationType.NONE for n in state_keys + action_keys)
    else:
        needs_stats = False

    if needs_stats:
        if dataset_statistics is None:
            dataset_statistics = _compute_lerobot_statistics(path, info)
        elif isinstance(dataset_statistics, str):
            dataset_statistics = load_dataset_statistics(dataset_statistics)
        if structure is not None:
            dataset_statistics = filter_dataset_statistics_by_structure(dataset_statistics, structure)
        dataset = dataset.map(
            lambda ep: normalize(ep, structure, dataset_statistics),
            num_parallel_calls=num_parallel_calls,
            deterministic=not shuffle,
        )
    else:
        dataset_statistics = None

    assert "observation" in dataset.element_spec
    assert "action" in dataset.element_spec

    return dataset, dataset_statistics


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats_from_json(obj):
    """Recursively convert a JSON-loaded dict to numpy arrays at the leaves."""
    if isinstance(obj, dict):
        return {k: _stats_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return np.array(obj, dtype=np.float32)
    return obj  # int / float scalars (num_ep, num_steps)

def _compute_lerobot_statistics(
    path: str,
    info: dict,
    standardization_transform: Callable,
    recompute: bool = False,
) -> dict:
    """
    Compute dataset statistics (mean/std/min/max) in the post-transform structure.

    Results are cached at meta/dataset_statistics_openx.json so subsequent
    training runs load instantly.
    """
    cache_path = os.path.join(path, "meta", "dataset_statistics_openx.json")
    if not recompute and os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return _stats_from_json(raw)

    print("[lerobot] Computing dataset statistics (one-time, will be cached) ...")

    # Build a lightweight dataset without video – only parquet data.
    total = info["total_episodes"]
    state_dim = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]

    sig_noimg = {
        "observation.state": tf.TensorSpec(shape=(None, state_dim), dtype=tf.float32),
        "action": tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
        "is_first": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "is_last": tf.TensorSpec(shape=(None,), dtype=tf.bool),
    }

    def _gen_noimg():
        for ep_idx in range(total):
            chunk_idx = ep_idx // info["chunks_size"]
            p = os.path.join(
                path, info["data_path"].format(chunk_index=chunk_idx, episode_index=ep_idx)
            )
            df = pd.read_parquet(p)
            df = df.iloc[::SUBSAMPLE_FACTOR].reset_index(drop=True)
            state = np.stack(df["observation.state"].values).astype(np.float32)
            action = np.stack(df["action"].values).astype(np.float32)
            is_last = df["next.done"].values.astype(bool)
            is_last[-1] = True
            is_first = np.zeros(len(df), dtype=bool)
            is_first[0] = True
            yield {
                "observation.state": tf.constant(state),
                "action": tf.constant(action),
                "is_first": tf.constant(is_first),
                "is_last": tf.constant(is_last),
            }

    raw_ds = tf.data.Dataset.from_generator(_gen_noimg, output_signature=sig_noimg)

    # Apply the standardization transform to get the nested structure.
    # Images are absent; transforms must tolerate missing image keys.
    raw_ds = raw_ds.map(standardization_transform, num_parallel_calls=1)

    # Build initial accumulators from the element spec.
    elem_spec = raw_ds.element_spec
    sa_spec = {}
    if "state" in elem_spec.get("observation", {}):
        sa_spec["state"] = elem_spec["observation"]["state"]
    sa_spec["action"] = elem_spec["action"]

    state0 = dict(
        num_steps=0,
        num_ep=0,
        mean=tf.nest.map_structure(lambda s: tf.zeros(s.shape[1:], tf.float32), sa_spec),
        var=tf.nest.map_structure(lambda s: 1e-5 * tf.ones(s.shape[1:], tf.float32), sa_spec),
        min=tf.nest.map_structure(lambda s: 1e10 * tf.ones(s.shape[1:], tf.float32), sa_spec),
        max=tf.nest.map_structure(lambda s: -1e10 * tf.ones(s.shape[1:], tf.float32), sa_spec),
    )

    def _reduce(old, ep):
        sa = {}
        if "state" in ep.get("observation", {}):
            sa["state"] = ep["observation"]["state"]
        sa["action"] = ep["action"]

        n = tf.shape(tf.nest.flatten(sa)[0])[0] - 1
        batch_mean = tf.nest.map_structure(lambda x: tf.reduce_mean(x[:-1], 0), sa)
        batch_var  = tf.nest.map_structure(lambda x: tf.math.reduce_variance(x[:-1], 0), sa)

        count_f, n_f = tf.cast(old["num_steps"], tf.float32), tf.cast(n, tf.float32)
        total_f = count_f + n_f
        delta  = tf.nest.map_structure(lambda m, mb: mb - m, old["mean"], batch_mean)
        new_mean = tf.nest.map_structure(
            lambda m, d: m + d * n_f / total_f, old["mean"], delta
        )
        new_m2 = tf.nest.map_structure(
            lambda v, vb, d: v * count_f + vb * n_f + tf.square(d) * count_f * n_f / total_f,
            old["var"], batch_var, delta,
        )
        return dict(
            num_steps=old["num_steps"] + n,
            num_ep=old["num_ep"] + 1,
            mean=new_mean,
            var=tf.nest.map_structure(lambda m2: m2 / total_f, new_m2),
            min=tf.nest.map_structure(
                lambda x, m: tf.minimum(tf.reduce_min(x[:-1], 0), m), sa, old["min"]
            ),
            max=tf.nest.map_structure(
                lambda x, m: tf.maximum(tf.reduce_max(x[:-1], 0), m), sa, old["max"]
            ),
        )

    result = raw_ds.reduce(state0, _reduce)
    stats = dict(
        num_ep=int(result["num_ep"].numpy()),
        num_steps=int(result["num_steps"].numpy()),
        mean=tf.nest.map_structure(lambda x: x.numpy(), result["mean"]),
        std=tf.nest.map_structure(lambda x: tf.math.sqrt(x).numpy(), result["var"]),
        min=tf.nest.map_structure(lambda x: x.numpy(), result["min"]),
        max=tf.nest.map_structure(lambda x: x.numpy(), result["max"]),
    )

    # Cache to disk
    serialisable = tf.nest.map_structure(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x, stats
    )
    with open(cache_path, "w") as f:
        json.dump(serialisable, f, default=float, indent=2)
    print(f"[lerobot] Statistics cached to {cache_path}")

    return stats


# ---------------------------------------------------------------------------
# MCAP loader
# ---------------------------------------------------------------------------
#
# Expected on-disk layout (produced by convert_to_episodes.py):
#
#   <root>/
#     train/
#       episode_000000/
#         episode_000000.mcap            topics: /observation/state, /action, /episode/metadata
#         episode_000000_cam_head.mp4
#         episode_000000_cam_wrist.mp4
#       episode_000001/
#         ...
#     test/
#       ...

def is_mcap_dataset(path: str) -> bool:
    """
    Return True when *path* is an MCAP dataset root.
    Detected by finding at least one *.mcap file one or two levels below *path*.
    """
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    for entry in os.scandir(path):
        if not entry.is_dir():
            continue
        # Direct episode dirs (path/episode_XXXXXX/*.mcap)
        for sub in os.scandir(entry.path):
            if sub.name.endswith(".mcap"):
                return True
            # One level deeper (path/train/episode_XXXXXX/*.mcap)
            if sub.is_dir():
                for subsub in os.scandir(sub.path):
                    if subsub.name.endswith(".mcap"):
                        return True
    return False


def _mcap_split_dirs(root: str, split: str) -> List[str]:
    """
    Return sorted list of episode directories for a given split.

    Handles split descriptors:
      "train"          -> root/train/episode_*/
      "test"           -> root/test/episode_*/
      "train[0%:50%]"  -> first 50% of root/train/episode_*/
      "all"            -> all episode dirs directly under root/
    """
    m = re.match(r"^(\w+)(\[[\d%:]+\])?$", split)
    if not m:
        raise ValueError(f"Unrecognised MCAP split format: {split!r}")

    name, slice_str = m.group(1), m.group(2)

    split_dir = os.path.join(root, name)
    if os.path.isdir(split_dir):
        dirs = sorted(
            e.path for e in os.scandir(split_dir)
            if e.is_dir() and os.path.exists(os.path.join(e.path, e.name + ".mcap"))
        )
    elif name == "all":
        dirs = sorted(
            e.path for e in os.scandir(root)
            if e.is_dir() and any(f.endswith(".mcap") for f in os.listdir(e.path))
        )
    else:
        raise ValueError(f"MCAP split directory not found: {split_dir!r}")

    if slice_str:
        m2 = re.match(r"\[(\d*)%?:(\d*)%?\]", slice_str)
        if m2:
            s, e = m2.group(1), m2.group(2)
            n = len(dirs)
            s_idx = int(s) * n // 100 if s else 0
            e_idx = int(e) * n // 100 if e else n
            dirs = dirs[s_idx:e_idx]

    return dirs


def _read_mcap_episode(ep_dir: str, video_cameras: List[str]) -> dict:
    """
    Read one episode from an MCAP + MP4 directory.

    On first call the decoded episode is cached as ``<ep_dir>/<ep_name>.npz``
    so subsequent loads skip video decoding entirely (~0.05 s vs ~6 s).
    """
    from mcap.reader import make_reader

    ep_name = os.path.basename(ep_dir)

    # ---- fast path: load from npz cache ----
    cache_path = os.path.join(ep_dir, ep_name + "_cached.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        episode = {
            "observation.state": data["state"],
            "action":            data["action"],
            "is_first":          data["is_first"],
            "is_last":           data["is_last"],
        }
        for cam in video_cameras:
            key = f"observation.images.{cam}"
            if key in data:
                episode[key] = list(data[key])
        return episode
    # ----------------------------------------
    mcap_path = os.path.join(ep_dir, ep_name + ".mcap")

    states, actions, timestamps, frame_indices, dones = [], [], [], [], []

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        state_buf, action_buf, meta_buf = {}, {}, {}

        for _schema, channel, message in reader.iter_messages():
            msg = json.loads(message.data)
            fi = msg["frame_index"]
            if channel.topic == "/observation/state":
                state_buf[fi] = msg
            elif channel.topic == "/action":
                action_buf[fi] = msg
            elif channel.topic == "/episode/metadata":
                meta_buf[fi] = msg

    frame_indices_sorted = sorted(state_buf.keys())
    for fi in frame_indices_sorted:
        states.append(state_buf[fi]["data"])
        actions.append(action_buf[fi]["data"])
        timestamps.append(state_buf[fi]["timestamp"])
        frame_indices.append(fi)
        dones.append(meta_buf[fi]["next_done"])

    # Subsample 30 fps → 5 fps
    idx = list(range(0, len(states), SUBSAMPLE_FACTOR))
    states    = [states[i]    for i in idx]
    actions   = [actions[i]   for i in idx]
    timestamps= [timestamps[i]for i in idx]
    frame_idx_sub = [frame_indices[i] for i in idx]
    dones     = [dones[i]     for i in idx]

    ep_len = len(states)
    state_arr  = np.array(states,  dtype=np.float32)
    action_arr = np.array(actions, dtype=np.float32)
    is_last    = np.array(dones,   dtype=bool)
    is_last[-1] = True
    is_first   = np.zeros(ep_len, dtype=bool)
    is_first[0] = True

    episode = {
        "observation.state": state_arr,
        "action":            action_arr,
        "is_first":          is_first,
        "is_last":           is_last,
    }

    for cam in video_cameras:
        mp4_path = os.path.join(ep_dir, f"{ep_name}_{cam}.mp4")
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(f"Missing video: {mp4_path}")
        frames = iio.imread(mp4_path, plugin="pyav")  # (T, H, W, 3)
        jpeg_frames = []
        for fi in frame_idx_sub:
            frame = frames[fi] if fi < len(frames) else frames[-1]
            frame = _center_crop_and_resize(frame)
            jpeg  = tf.image.encode_jpeg(frame, quality=95).numpy()
            jpeg_frames.append(jpeg)
        # Store under the same key the transform expects
        episode[f"observation.images.{cam}"] = jpeg_frames

    # ---- save npz cache for fast future loads ----
    save_dict = {
        "state":    episode["observation.state"],
        "action":   episode["action"],
        "is_first": episode["is_first"],
        "is_last":  episode["is_last"],
    }
    for cam in video_cameras:
        key = f"observation.images.{cam}"
        save_dict[key] = np.array(episode[key], dtype=object)
    np.savez_compressed(cache_path, **save_dict)
    # -----------------------------------------------

    return episode


def _compute_mcap_statistics(
    path: str,
    split: str,
    video_cameras: List[str],
    standardization_transform: Callable,
    recompute: bool = False,
) -> dict:
    """
    One-pass streaming statistics over all MCAP episodes (no video decoding).
    Cached at <path>/dataset_statistics_openx.json.
    """
    cache_path = os.path.join(path, "dataset_statistics_openx.json")
    if not recompute and os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return _stats_from_json(raw)

    print("[mcap] Computing dataset statistics (one-time, will be cached) ...")

    from mcap.reader import make_reader

    # Use all train episodes for statistics regardless of split slice
    base_split = re.match(r"^(\w+)", split).group(1)
    all_dirs = _mcap_split_dirs(path, base_split)

    def _gen_noimg():
        for ep_dir in all_dirs:
            ep_name = os.path.basename(ep_dir)
            mcap_path = os.path.join(ep_dir, ep_name + ".mcap")
            states, actions, dones = [], [], {}
            try:
                with open(mcap_path, "rb") as f:
                    for _s, channel, message in make_reader(f).iter_messages():
                        msg = json.loads(message.data)
                        fi = msg["frame_index"]
                        if channel.topic == "/observation/state":
                            states.append((fi, msg["data"]))
                        elif channel.topic == "/action":
                            actions.append((fi, msg["data"]))
                        elif channel.topic == "/episode/metadata":
                            dones[fi] = msg["next_done"]
            except Exception as exc:
                print(f"[mcap] stats: skipping {ep_dir}: {exc}")
                continue

            states.sort(key=lambda x: x[0])
            actions.sort(key=lambda x: x[0])
            idx = list(range(0, len(states), SUBSAMPLE_FACTOR))
            s_arr = np.array([states[i][1] for i in idx], dtype=np.float32)
            a_arr = np.array([actions[i][1] for i in idx], dtype=np.float32)
            is_last = np.zeros(len(idx), dtype=bool); is_last[-1] = True
            is_first = np.zeros(len(idx), dtype=bool); is_first[0] = True
            raw_ep = {
                "observation.state": tf.constant(s_arr),
                "action":            tf.constant(a_arr),
                "is_first":          tf.constant(is_first),
                "is_last":           tf.constant(is_last),
            }
            for cam in video_cameras:
                raw_ep[f"observation.images.{cam}"] = tf.constant(
                    [b""] * len(idx), dtype=tf.string
                )
            yield raw_ep

    # Infer dims from first episode
    first = next(_gen_noimg())
    first = standardization_transform(first)
    sa_spec = {}
    if "state" in first.get("observation", {}):
        sa_spec["state"] = tf.nest.map_structure(lambda x: tf.TensorSpec(x.shape, x.dtype), first["observation"]["state"])
    sa_spec["action"] = tf.nest.map_structure(lambda x: tf.TensorSpec(x.shape, x.dtype), first["action"])

    state0 = dict(
        num_steps=0, num_ep=0,
        mean=tf.nest.map_structure(lambda s: tf.zeros(s.shape[1:] if s.shape.rank > 1 else s.shape, tf.float32), sa_spec),
        var =tf.nest.map_structure(lambda s: 1e-5 * tf.ones(s.shape[1:] if s.shape.rank > 1 else s.shape, tf.float32), sa_spec),
        min =tf.nest.map_structure(lambda s:  1e10 * tf.ones(s.shape[1:] if s.shape.rank > 1 else s.shape, tf.float32), sa_spec),
        max =tf.nest.map_structure(lambda s: -1e10 * tf.ones(s.shape[1:] if s.shape.rank > 1 else s.shape, tf.float32), sa_spec),
    )

    def _reduce(old, raw_ep):
        ep = standardization_transform(raw_ep)
        sa = {}
        if "state" in ep.get("observation", {}):
            sa["state"] = ep["observation"]["state"]
        sa["action"] = ep["action"]
        n = tf.shape(tf.nest.flatten(sa)[0])[0] - 1
        bm = tf.nest.map_structure(lambda x: tf.reduce_mean(x[:-1], 0), sa)
        bv = tf.nest.map_structure(lambda x: tf.math.reduce_variance(x[:-1], 0), sa)
        cf, nf = tf.cast(old["num_steps"], tf.float32), tf.cast(n, tf.float32)
        tf_ = cf + nf
        d   = tf.nest.map_structure(lambda m, mb: mb - m, old["mean"], bm)
        nm  = tf.nest.map_structure(lambda m, dd: m + dd * nf / tf_, old["mean"], d)
        nm2 = tf.nest.map_structure(
            lambda v, vb, dd: v * cf + vb * nf + tf.square(dd) * cf * nf / tf_,
            old["var"], bv, d)
        return dict(
            num_steps=old["num_steps"] + n, num_ep=old["num_ep"] + 1,
            mean=nm, var=tf.nest.map_structure(lambda m2: m2 / tf_, nm2),
            min=tf.nest.map_structure(lambda x, m: tf.minimum(tf.reduce_min(x[:-1], 0), m), sa, old["min"]),
            max=tf.nest.map_structure(lambda x, m: tf.maximum(tf.reduce_max(x[:-1], 0), m), sa, old["max"]),
        )

    state = state0
    for raw_ep in _gen_noimg():
        state = _reduce(state, raw_ep)

    stats = dict(
        num_ep=int(state["num_ep"]) if isinstance(state["num_ep"], (int, np.integer)) else int(state["num_ep"].numpy()),
        num_steps=int(state["num_steps"]) if isinstance(state["num_steps"], (int, np.integer)) else int(state["num_steps"].numpy()),
        mean=tf.nest.map_structure(lambda x: x.numpy() if hasattr(x, "numpy") else x, state["mean"]),
        std =tf.nest.map_structure(lambda x: (tf.math.sqrt(x)).numpy() if hasattr(x, "numpy") else x, state["var"]),
        min =tf.nest.map_structure(lambda x: x.numpy() if hasattr(x, "numpy") else x, state["min"]),
        max =tf.nest.map_structure(lambda x: x.numpy() if hasattr(x, "numpy") else x, state["max"]),
    )

    ser = tf.nest.map_structure(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, stats)
    with open(cache_path, "w") as f:
        json.dump(ser, f, default=float, indent=2)
    print(f"[mcap] Statistics cached to {cache_path}")
    return stats


def load_mcap_dataset(
    path: str,
    split: str,
    standardization_transform: Callable,
    structure: Optional[dict] = None,
    dataset_statistics: Optional[Union[str, Dict]] = None,
    recompute_statistics: bool = False,
    num_parallel_calls: Optional[int] = tf.data.AUTOTUNE,
    shuffle: bool = True,
    filter_fn: Optional[Callable] = None,
    minimum_length: int = 3,
):
    """
    Load an MCAP dataset and return it in the same format as ``core.load_dataset``.

    *path* is the dataset root (contains train/ and test/ subdirs).
    *split* is one of: "train", "test", "train[0%:50%]", "all".
    """
    from .core import filter_dataset_statistics_by_structure, load_dataset_statistics
    from .transforms import normalize

    try:
        from mcap.reader import make_reader  # noqa: F401
    except ImportError as exc:
        raise ImportError("pip install mcap") from exc

    # Cameras to load (short names matching the mp4 filename suffix)
    video_cameras = ["cam_head"]

    ep_dirs = _mcap_split_dirs(path, split)
    if not ep_dirs:
        raise RuntimeError(f"No MCAP episodes found for split {split!r} in {path}")

    if shuffle:
        import random
        random.shuffle(ep_dirs)

    # Infer state/action dims from first episode
    _sample = _read_mcap_episode(ep_dirs[0], video_cameras)
    state_dim  = _sample["observation.state"].shape[1]
    action_dim = _sample["action"].shape[1]

    output_signature = {
        "observation.state": tf.TensorSpec(shape=(None, state_dim),  dtype=tf.float32),
        "action":            tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
        "is_first":          tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "is_last":           tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "ep_idx":            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "quality_score":     tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }
    for cam in video_cameras:
        output_signature[f"observation.images.{cam}"] = tf.TensorSpec(
            shape=(None,), dtype=tf.string
        )

    def _generator():
        for global_ep_idx, ep_dir in enumerate(ep_dirs):
            try:
                raw = _read_mcap_episode(ep_dir, video_cameras)
            except Exception as exc:
                print(f"[mcap] Warning: skipping {ep_dir}: {exc}")
                continue
            ep_len = len(raw["is_first"])
            ep_tf = {
                "observation.state": tf.constant(raw["observation.state"], dtype=tf.float32),
                "action":            tf.constant(raw["action"],            dtype=tf.float32),
                "is_first":          tf.constant(raw["is_first"],          dtype=tf.bool),
                "is_last":           tf.constant(raw["is_last"],           dtype=tf.bool),
                "ep_idx":            tf.fill([ep_len], tf.constant(global_ep_idx, dtype=tf.int32)),
                "quality_score":     tf.zeros([ep_len], dtype=tf.int32),
            }
            for cam in video_cameras:
                ep_tf[f"observation.images.{cam}"] = tf.constant(
                    raw[f"observation.images.{cam}"], dtype=tf.string
                )
            yield ep_tf

    dataset = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)

    if filter_fn is not None:
        dataset = dataset.filter(filter_fn())

    def _standardize_mcap(ep):
        from .core import filter_by_structure
        # Preserve auxiliary fields not returned by the standardization transform
        ep_idx_val = ep.get("ep_idx")
        quality_score_val = ep.get("quality_score")
        ep = standardization_transform(ep)
        if ep_idx_val is not None:
            ep["ep_idx"] = ep_idx_val
        if quality_score_val is not None:
            ep["quality_score"] = quality_score_val
        if structure is not None:
            ep = filter_by_structure(ep, structure)
        return ep

    dataset = dataset.map(
        _standardize_mcap,
        num_parallel_calls=num_parallel_calls,
        deterministic=not shuffle,
    )
    dataset = dataset.filter(
        lambda ep: tf.shape(tf.nest.flatten(ep["action"])[0])[0] >= minimum_length
    )

    # Statistics
    if structure is not None:
        state_keys  = tf.nest.flatten(structure["observation"].get("state", NormalizationType.NONE))
        action_keys = tf.nest.flatten(structure["action"])
        needs_stats = any(n != NormalizationType.NONE for n in state_keys + action_keys)
    else:
        needs_stats = False

    if needs_stats:
        if dataset_statistics is None:
            dataset_statistics = _compute_mcap_statistics(
                path, split, video_cameras, standardization_transform,
                recompute=recompute_statistics,
            )
        elif isinstance(dataset_statistics, str):
            dataset_statistics = load_dataset_statistics(dataset_statistics)
        if structure is not None:
            dataset_statistics = filter_dataset_statistics_by_structure(dataset_statistics, structure)
        dataset = dataset.map(
            lambda ep: normalize(ep, structure, dataset_statistics),
            num_parallel_calls=num_parallel_calls,
            deterministic=not shuffle,
        )
    else:
        dataset_statistics = None

    assert "observation" in dataset.element_spec
    assert "action"      in dataset.element_spec

    return dataset, dataset_statistics

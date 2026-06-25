#!/usr/bin/env python3
"""
Convert LeRobot dataset (clean_data/) into episode-wise directories,
each containing a .mcap file (state/action/metadata) and .mp4 files
(one per camera).

Output layout:
  episode_data/
    train/
      episode_000000/
        episode_000000.mcap
        episode_000000_cam_head.mp4
        episode_000000_cam_wrist.mp4
    test/
      episode_000000/
        ...
"""

import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mcap.writer import Writer


CLEAN_DATA = Path("/data/demonstration-information/clean_data")
OUT_ROOT = Path("/data/demonstration-information/episode_data")

SPLITS = {
    "train": CLEAN_DATA / "clean_train",
    "test": CLEAN_DATA / "clean_test",
}

# All jitter/jerk subfolders combined into one split
JITTER_SOURCES = [
    Path("/data/demonstration-information/jitter_data/jitter_train"),
    Path("/data/demonstration-information/jitter_data/jitter_test"),
    Path("/data/demonstration-information/jitter_data/jerk_train"),
    Path("/data/demonstration-information/jitter_data/jerk_test"),
]

VIDEO_KEYS = [
    "observation.images.cam_head",
    "observation.images.cam_wrist",
]

# Short names used for output filenames
VIDEO_SHORT = {
    "observation.images.cam_head": "cam_head",
    "observation.images.cam_wrist": "cam_wrist",
}


def ndarray_to_list(val):
    """Convert numpy array or list to plain Python list."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return val


def write_mcap(parquet_path: Path, out_mcap: Path, episode_index: int):
    df = pd.read_parquet(parquet_path)

    schema_defs = {
        "observation_state": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "frame_index": {"type": "integer"},
                "episode_index": {"type": "integer"},
                "data": {"type": "array", "items": {"type": "number"}},
            },
        },
        "action": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "frame_index": {"type": "integer"},
                "episode_index": {"type": "integer"},
                "data": {"type": "array", "items": {"type": "number"}},
            },
        },
        "episode_metadata": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "number"},
                "frame_index": {"type": "integer"},
                "episode_index": {"type": "integer"},
                "task_index": {"type": "integer"},
                "next_done": {"type": "boolean"},
                "next_success": {"type": "boolean"},
            },
        },
    }

    out_mcap.parent.mkdir(parents=True, exist_ok=True)

    with open(out_mcap, "wb") as f:
        writer = Writer(f)
        writer.start(library="lerobot-converter", profile="")

        schema_ids = {}
        for name, schema in schema_defs.items():
            schema_id = writer.register_schema(
                name=name,
                encoding="jsonschema",
                data=json.dumps(schema).encode(),
            )
            schema_ids[name] = schema_id

        channel_ids = {
            "observation_state": writer.register_channel(
                topic="/observation/state",
                message_encoding="json",
                schema_id=schema_ids["observation_state"],
            ),
            "action": writer.register_channel(
                topic="/action",
                message_encoding="json",
                schema_id=schema_ids["action"],
            ),
            "episode_metadata": writer.register_channel(
                topic="/episode/metadata",
                message_encoding="json",
                schema_id=schema_ids["episode_metadata"],
            ),
        }

        fps = 30.0
        for _, row in df.iterrows():
            ts_sec = float(row["timestamp"])
            ts_ns = int(ts_sec * 1e9)
            frame_idx = int(row["frame_index"])
            ep_idx = int(row["episode_index"])

            state_msg = json.dumps({
                "timestamp": ts_sec,
                "frame_index": frame_idx,
                "episode_index": ep_idx,
                "data": ndarray_to_list(row["observation.state"]),
            }).encode()
            writer.add_message(
                channel_id=channel_ids["observation_state"],
                log_time=ts_ns,
                data=state_msg,
                publish_time=ts_ns,
            )

            action_msg = json.dumps({
                "timestamp": ts_sec,
                "frame_index": frame_idx,
                "episode_index": ep_idx,
                "data": ndarray_to_list(row["action"]),
            }).encode()
            writer.add_message(
                channel_id=channel_ids["action"],
                log_time=ts_ns,
                data=action_msg,
                publish_time=ts_ns,
            )

            meta_msg = json.dumps({
                "timestamp": ts_sec,
                "frame_index": frame_idx,
                "episode_index": ep_idx,
                "task_index": int(row["task_index"]),
                "next_done": bool(row["next.done"]),
                "next_success": bool(row["next.success"]),
            }).encode()
            writer.add_message(
                channel_id=channel_ids["episode_metadata"],
                log_time=ts_ns,
                data=meta_msg,
                publish_time=ts_ns,
            )

        writer.finish()


def convert_split(split_name: str, split_path: Path):
    out_split = OUT_ROOT / split_name

    data_dir = split_path / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))

    print(f"\n[{split_name}] Converting {len(parquet_files)} episodes → {out_split}")

    for parquet_path in parquet_files:
        ep_stem = parquet_path.stem  # e.g. episode_000000
        ep_idx = int(ep_stem.split("_")[1])

        ep_out = out_split / ep_stem
        ep_out.mkdir(parents=True, exist_ok=True)

        # --- MCAP ---
        mcap_path = ep_out / f"{ep_stem}.mcap"
        if not mcap_path.exists():
            write_mcap(parquet_path, mcap_path, ep_idx)

        # --- MP4s ---
        for vkey in VIDEO_KEYS:
            src_mp4 = split_path / "videos" / "chunk-000" / vkey / f"{ep_stem}.mp4"
            short = VIDEO_SHORT[vkey]
            dst_mp4 = ep_out / f"{ep_stem}_{short}.mp4"
            if src_mp4.exists() and not dst_mp4.exists():
                shutil.copy2(src_mp4, dst_mp4)
            elif not src_mp4.exists():
                print(f"  WARNING: missing video {src_mp4}")

        print(f"  {ep_stem} done", flush=True)


def convert_combined(sources, split_name):
    """Combine multiple LeRobot source dirs into one MCAP split, renumbering episodes."""
    out_split = OUT_ROOT / split_name
    ep_counter = 0
    for source in sources:
        data_dir = source / "data" / "chunk-000"
        parquet_files = sorted(data_dir.glob("episode_*.parquet"))
        print(f"  [{source.name}] {len(parquet_files)} episodes starting at {ep_counter:06d}")
        for parquet_path in parquet_files:
            orig_stem = parquet_path.stem
            ep_stem = f"episode_{ep_counter:06d}"
            ep_out = out_split / ep_stem
            ep_out.mkdir(parents=True, exist_ok=True)

            mcap_path = ep_out / f"{ep_stem}.mcap"
            if not mcap_path.exists():
                write_mcap(parquet_path, mcap_path, ep_counter)

            for vkey in VIDEO_KEYS:
                src_mp4 = source / "videos" / "chunk-000" / vkey / f"{orig_stem}.mp4"
                short = VIDEO_SHORT[vkey]
                dst_mp4 = ep_out / f"{ep_stem}_{short}.mp4"
                if src_mp4.exists() and not dst_mp4.exists():
                    shutil.copy2(src_mp4, dst_mp4)
                elif not src_mp4.exists():
                    print(f"  WARNING: missing video {src_mp4}")

            print(f"  {ep_stem} ({source.name}/{orig_stem}) done", flush=True)
            ep_counter += 1
    print(f"[{split_name}] Total: {ep_counter} episodes → {out_split}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split_name, split_path in SPLITS.items():
        if not split_path.exists():
            print(f"Skipping {split_name} (not found)")
            continue
        convert_split(split_name, split_path)

    print("\n[jitter] Combining jitter/jerk subfolders...")
    convert_combined(JITTER_SOURCES, "jitter")
    print("\nDone.")


if __name__ == "__main__":
    main()

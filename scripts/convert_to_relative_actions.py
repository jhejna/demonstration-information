#!/usr/bin/env python3
"""
Convert episode_data/ to episode_data_relative/ with relative (delta) actions.

Relative action definition:
    relative_action[0] = 0                          (no displacement at first step)
    relative_action[t] = action[t] - action[t-1]    (delta from previous command)

All other data (state, video, metadata) is copied unchanged.

Usage:
    python scripts/convert_to_relative_actions.py
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer

SRC_ROOT = Path("/data/demonstration-information/episode_data")
DST_ROOT = Path("/data/demonstration-information/episode_data_relative")
SPLITS = ["train", "test"]


def read_mcap(mcap_path: Path):
    """Read state, action, and metadata from an MCAP file."""
    states, actions, meta = {}, {}, {}
    with open(mcap_path, "rb") as f:
        for _schema, channel, message in make_reader(f).iter_messages():
            msg = json.loads(message.data)
            fi = msg["frame_index"]
            if channel.topic == "/observation/state":
                states[fi] = msg
            elif channel.topic == "/action":
                actions[fi] = msg
            elif channel.topic == "/episode/metadata":
                meta[fi] = msg
    return states, actions, meta


def write_mcap(mcap_path: Path, states: dict, actions: dict, meta: dict):
    """Write state, action, and metadata dicts to an MCAP file."""
    frame_indices = sorted(states.keys())
    with open(mcap_path, "wb") as f:
        writer = Writer(f)
        writer.start()
        state_chan = writer.register_channel(topic="/observation/state", message_encoding="json", schema_id=0)
        action_chan = writer.register_channel(topic="/action", message_encoding="json", schema_id=0)
        meta_chan = writer.register_channel(topic="/episode/metadata", message_encoding="json", schema_id=0)

        for i, fi in enumerate(frame_indices):
            ts = i * 1_000_000  # synthetic timestamp in ns
            writer.add_message(state_chan, log_time=ts, data=json.dumps(states[fi]).encode(), publish_time=ts)
            writer.add_message(action_chan, log_time=ts, data=json.dumps(actions[fi]).encode(), publish_time=ts)
            writer.add_message(meta_chan, log_time=ts, data=json.dumps(meta[fi]).encode(), publish_time=ts)

        writer.finish()


def convert_episode(src_dir: Path, dst_dir: Path):
    ep_name = src_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Read source MCAP
    src_mcap = src_dir / f"{ep_name}.mcap"
    states, actions, meta = read_mcap(src_mcap)

    frame_indices = sorted(states.keys())
    action_arr = np.array([actions[fi]["data"] for fi in frame_indices], dtype=np.float32)

    # Compute relative actions: delta from previous action, zeros at first step
    relative = np.zeros_like(action_arr)
    relative[1:] = action_arr[1:] - action_arr[:-1]

    # Write modified actions back into dict
    rel_actions = {}
    for i, fi in enumerate(frame_indices):
        rel_actions[fi] = dict(actions[fi])
        rel_actions[fi]["data"] = relative[i].tolist()

    # Write new MCAP
    dst_mcap = dst_dir / f"{ep_name}.mcap"
    write_mcap(dst_mcap, states, rel_actions, meta)

    # Copy video files (unchanged)
    for f in src_dir.iterdir():
        if f.suffix == ".mp4":
            shutil.copy2(f, dst_dir / f.name)


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src_split = SRC_ROOT / split
        dst_split = DST_ROOT / split
        if not src_split.exists():
            print(f"Skipping missing split: {src_split}")
            continue

        ep_dirs = sorted(src_split.iterdir())
        print(f"Converting {len(ep_dirs)} episodes in '{split}' split...")

        for i, ep_dir in enumerate(ep_dirs):
            if not ep_dir.is_dir():
                continue
            convert_episode(ep_dir, dst_split / ep_dir.name)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(ep_dirs)} done")

        print(f"  {len(ep_dirs)}/{len(ep_dirs)} done")

    # Copy dataset_statistics_openx.json if present (will need recomputation)
    stats_src = SRC_ROOT / "dataset_statistics_openx.json"
    if stats_src.exists():
        print("NOTE: dataset_statistics_openx.json not copied — stats must be recomputed for relative actions.")

    print(f"\nDone. Relative action dataset saved to: {DST_ROOT}")


if __name__ == "__main__":
    main()

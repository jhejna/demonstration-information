"""
Pre-process all MCAP episodes to .npz cache files so training loads instantly.

Usage:
    conda run -n openx python3 scripts/preprocess_episodes.py \
        --root episode_data --splits train test --workers 8
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def process_one(ep_dir: str) -> tuple:
    from openx.data.lerobot import _read_mcap_episode
    ep_name = os.path.basename(ep_dir)
    cache = os.path.join(ep_dir, ep_name + "_cached.npz")
    if os.path.exists(cache):
        return ep_dir, 0.0, True  # already cached
    t = time.time()
    _read_mcap_episode(ep_dir, ["cam_head"])
    return ep_dir, time.time() - t, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default="episode_data")
    parser.add_argument("--splits",  nargs="+", default=["train", "test"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ep_dirs = []
    for split in args.splits:
        split_dir = os.path.join(args.root, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split_dir} (not found)")
            continue
        for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
            if entry.is_dir():
                ep_dirs.append(entry.path)

    total = len(ep_dirs)
    print(f"Processing {total} episodes with {args.workers} workers...\n")

    done = skipped = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, d): d for d in ep_dirs}
        for fut in as_completed(futures):
            ep_dir, elapsed, was_cached = fut.result()
            ep_name = os.path.basename(ep_dir)
            done += 1
            if was_cached:
                skipped += 1
                tag = "cached"
            else:
                tag = f"{elapsed:.1f}s"
            pct = done / total * 100
            print(f"  [{done:3d}/{total}  {pct:5.1f}%]  {ep_name}  ({tag})", flush=True)

    new = done - skipped
    print(f"\nDone. {new} newly cached, {skipped} already cached. "
          f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

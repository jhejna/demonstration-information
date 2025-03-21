"""
A script that computes scores for robomimic datasets.

Example command:

python scripts/quality/estimate_quality.py \
    --obs_ckpt path/to/vae \
    --action_ckpt path/to/vae \
    --batch_size 1024 \
    --estimator ksg
"""

import os
import pickle
import pprint

import jax
import numpy as np
import quality_estimators
import tensorflow as tf
from absl import app, flags
from jax.experimental import compilation_cache, multihost_utils
from matplotlib import pyplot as plt
from scipy import stats

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "The path to save the results if desired.", required=False)
flags.DEFINE_string("obs_ckpt", None, "Path to the obs logs and checkpoints.", required=False)
flags.DEFINE_string("action_ckpt", None, "Path to the action logs and checkpoints.", required=False)
flags.DEFINE_string("obs_action_ckpt", None, "Path to the obs action logs and checkpoints.", required=False)
flags.DEFINE_enum(
    "estimator",
    None,
    ["biksg", "ksg", "kl", "nce", "vip", "random", "l2", "mine", "stddev", "inv_stddev", "compatibility"],
    "Type of quality estimator",
    required=True,
)
flags.DEFINE_integer(
    "batch_size", 1024, "The batch size for the dataset, by default override the config.", required=False
)


def main(_):
    # Initialize experimental jax compilation cache
    compilation_cache.compilation_cache.set_cache_dir(os.path.expanduser("~/.jax_compilation_cache"))

    # Define Shardings
    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")
    dp_spec = jax.sharding.PartitionSpec("batch")
    dp_sharding = jax.sharding.NamedSharding(mesh, dp_spec)
    rep_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def shard(batch):
        batch = jax.tree.map(lambda x: x._numpy(), batch)
        return multihost_utils.host_local_array_to_global_array(batch, mesh, dp_spec)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    ds, pred_fn, dataset_ids = quality_estimators.get_dataset_and_score_fn(
        FLAGS.estimator, FLAGS.batch_size, FLAGS.obs_ckpt, FLAGS.action_ckpt, FLAGS.obs_action_ckpt,
        sharding=rep_sharding
    )
    ds = map(shard, ds)  # Apply sharding to the batches
    jitted_pred_fn = jax.jit(pred_fn, in_shardings=(dp_sharding, None), out_shardings=(rep_sharding, rep_sharding))
    rng = jax.random.key(jax.process_index())
    scores = quality_estimators.estimate_quality(ds, jitted_pred_fn, dataset_ids, rng)

    pprint.pprint(scores)

    if jax.process_index() == 0:
        if FLAGS.path is not None:
            tf.io.gfile.makedirs(FLAGS.path)

        for ds_name, ds_scores in scores.items():
            # Save
            if FLAGS.path is not None:
                with tf.io.gfile.GFile(os.path.join(FLAGS.path, ds_name + ".pkl"), "wb") as f:
                    pickle.dump(ds_scores, f)

            # If we can, visualize
            if "quality_by_ep_idx" in ds_scores:
                # Compute statistics
                idxs = list(ds_scores["ep_idx"].keys())
                x = np.array([ds_scores["quality_by_ep_idx"][idx] for idx in idxs])
                y = np.array([ds_scores["ep_idx"][idx] for idx in idxs])
                r = stats.pearsonr(x, y)
                print("R:", r)

                # If we have statistics and a path, then save a histogram and curve.
                if FLAGS.path is not None:
                    _, ax = plt.subplots(1, 1, figsize=(3, 3))
                    for v in np.sort(np.unique(x)):
                        ax.hist(y[x == v], bins=20, alpha=0.5, label=f"Quality {v}")
                    ax.legend()
                    title = f"r={r[0]:.2f}"
                    ax.set_title(title)
                    plt.tight_layout()
                    with tf.io.gfile.GFile(os.path.join(FLAGS.path, ds_name + "_hist.png"), "wb") as f:
                        plt.savefig(f)

                    # Now plot the quality curve
                    plt.clf()
                    sort_idx = np.argsort(y)  # Sorts from low score to high score episodes.
                    rev_sorted_quality_labels = x[sort_idx][::-1]  # The sorted quality labels
                    total_quality_labels = np.cumsum(rev_sorted_quality_labels)
                    num_data_points = 1 + np.arange(total_quality_labels.shape[0])
                    avg_quality_label = total_quality_labels / num_data_points
                    # Finally, re-reverse to set the axes back to num data points removed.
                    plt.plot(np.arange(avg_quality_label.shape[0]), avg_quality_label[::-1], label="method")
                    # Plot the oracle strategy
                    oracle_labls = np.cumsum(np.sort(x)[::-1]) / num_data_points
                    plt.plot(np.arange(oracle_labls.shape[0]), oracle_labls[::-1], color="gray", label="oracle")
                    plt.gca().hlines(np.mean(x), xmin=0, xmax=oracle_labls.shape[0], color="red", linestyles="dashed")

                    plt.xlabel("Episodes Removed")
                    plt.ylabel("Average Quality Label")
                    plt.legend(frameon=False)
                    plt.ylim(np.min(x), np.max(x))

                    plt.title(title)
                    plt.tight_layout()
                    with tf.io.gfile.GFile(os.path.join(FLAGS.path, ds_name + "_curve.png"), "wb") as f:
                        plt.savefig(f)


if __name__ == "__main__":
    app.run(main)

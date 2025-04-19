import functools
import json
import os
from typing import List

import jax
import numpy as np
import tensorflow as tf
import tqdm
from jax import numpy as jnp
from jax.scipy.special import digamma
from ml_collections import ConfigDict

from openx.data.dataloader import make_dataloader
from openx.utils.evaluate import load_checkpoint

AGGREGATION_KEYS = [
    "ep_idx",
    "quality_score",
    # "quality_score_continuous", # Enable only if using RoboCrowd.
    "dataset_id",
]

"""
Quality Estimators
"""


def _l2_dists(z):
    return jnp.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)  # (B, B)


def _knn(z, ks):
    # Faster implementations exist, but this is OK for now since we are streaming.
    dist = _l2_dists(z)
    return jnp.sort(dist, axis=-1)[:, ks]


def ksg_estimator(batch, rng, ks, obs_alg, obs_state, action_alg, action_state):
    # Get the state and action encoding
    obs_rng, action_rng = jax.random.split(rng)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)

    obs_dist = _l2_dists(z_obs)
    action_dist = _l2_dists(z_action)

    # Use the InfNorm on Z
    joint_dist = jnp.maximum(obs_dist, action_dist)
    joint_knn_dists = jnp.sort(joint_dist, axis=-1)[:, ks]

    obs_count = jnp.sum(obs_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)
    action_count = jnp.sum(action_dist[:, :, None] < joint_knn_dists[:, None, :], axis=1)

    return -jnp.mean(digamma(obs_count) + digamma(action_count), axis=-1)


def biksg_estimator(batch, rng, ks, obs_alg, obs_state, action_alg, action_state):
    # Get the state and action encoding
    obs_rng, action_rng = jax.random.split(rng)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)
    z_joint = jnp.concatenate((z_obs, z_action), axis=-1)

    joint_knn_dists = _knn(z_joint, ks)  # (B, k)

    # Now we need counts for each of the marginals, omitting the self count.
    obs_count = jnp.sum(_l2_dists(z_obs)[:, :, None] <= joint_knn_dists[:, None, :], axis=1) - 1
    action_count = jnp.sum(_l2_dists(z_action)[:, :, None] <= joint_knn_dists[:, None, :], axis=1) - 1

    return -jnp.mean(jnp.log(obs_count) + jnp.log(action_count), axis=-1)


def kl_estimator(
    batch,
    rng,
    ks,
    obs_alg,
    obs_state,
    action_alg,
    action_state,
    obs_action_alg=None,
    obs_action_state=None,
    obs_weight: float = 1.0,
):
    obs_rng, action_rng, joint_rng = jax.random.split(rng, num=3)
    z_obs = obs_alg.predict(obs_state, batch, obs_rng)
    z_action = action_alg.predict(action_state, batch, action_rng)
    if obs_action_alg is not None:
        z_joint = obs_action_alg.predict(obs_action_state, batch, joint_rng)
    else:
        z_joint = jnp.concatenate((z_obs, z_action), axis=-1)

    # I(S;A) = H(S) + H(A) - H(S, A)
    h_obs = z_obs.shape[-1] * jnp.log(_knn(z_obs, ks))  # (B, k)
    h_action = z_action.shape[-1] * jnp.log(_knn(z_action, ks))  # (B, k)
    h_joint = z_joint.shape[-1] * jnp.log(_knn(z_joint, ks))  # (B, k)

    return jnp.mean(obs_weight * h_obs + h_action - h_joint, axis=-1)


def nce_estimator(batch, rng, obs_action_alg, obs_action_state):
    return obs_action_alg.predict(obs_action_state, batch, rng)


def vip_estimator(batch, rng, obs_alg, obs_state):
    return obs_alg.predict(obs_state, batch, rng)


def mine_estimator(batch, rng, alg, state):
    return alg.predict(state, batch, rng)


def l2_loss_estimator(batch, rng, alg, state):
    pred = alg.predict(state, batch, rng)
    loss = jnp.square(pred - batch["action"]).sum(axis=-1)  # (B, T, D) --> (B, T)
    return -jnp.sum(loss * batch["mask"], axis=-1) / (jnp.sum(batch["mask"], axis=-1))


def stddev_estimator(batch, rng, alg, state):
    # We need to get the prediction and the stddev
    _, stddev = alg.predict_with_confidence(state, batch, rng)
    return stddev  # For entropy estimators, higher stddev is supposedly good


def inverse_stddev_estimator(batch, rng, alg, state):
    _, stddev = alg.predict_with_confidence(state, batch, rng)
    return -stddev  # For entropy estimators, higher stddev is supposedly good


def compatibility_estimator(batch, rng, alg, state):
    # We need to get the prediction and the stddev
    pred, stddev = alg.predict_with_confidence(state, batch, rng)
    loss = jnp.square(pred - batch["action"]).sum(axis=-1)  # (B, T, D) --> (B, T)
    loss = jnp.sum(loss * batch["mask"], axis=-1) / (jnp.sum(batch["mask"], axis=-1))
    eta = 0.05
    lambd = 4
    return jnp.where(stddev < eta, 1 - jnp.minimum(loss / lambd, 1.0), loss)


def random_estimator(batch, rng):
    # determine the batch size somehow
    batch_size = jax.tree.leaves(batch["observation"])[0].shape[0]
    return jax.random.uniform(rng, shape=(batch_size,))


def get_dataset_and_score_fn(
    estimator: str,
    batch_size: int | None = None,
    obs_ckpt: str | None = None,
    action_ckpt: str | None = None,
    obs_action_ckpt: str | None = None,
    datasets: List[str] | None = None,
    sharding: jax.sharding.Sharding | None = None,
):
    # First get the prediction function, and set the correct dataset parameters for it.
    if estimator == "random":
        pred_fn = random_estimator
        repeat, discard_fraction = 1, 0.99  # Remove almost all the data, don't repeat
    elif estimator == "ksg":
        assert obs_ckpt is not None and action_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        action_alg, action_state, _, _ = load_checkpoint(action_ckpt, sharding=sharding)
        pred_fn = functools.partial(
            ksg_estimator,
            ks=np.arange(5, 8),
            obs_alg=obs_alg,
            obs_state=obs_state,
            action_alg=action_alg,
            action_state=action_state,
        )
        repeat, discard_fraction = 4, 0.5
    elif estimator == "biksg":
        assert obs_ckpt is not None and action_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        action_alg, action_state, _, _ = load_checkpoint(action_ckpt, sharding=sharding)
        pred_fn = functools.partial(
            biksg_estimator,
            ks=np.arange(2, 5),
            obs_alg=obs_alg,
            obs_state=obs_state,
            action_alg=action_alg,
            action_state=action_state,
        )
        repeat, discard_fraction = 4, 0.5
    elif estimator == "kl":
        assert obs_ckpt is not None and action_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        action_alg, action_state, _, _ = load_checkpoint(action_ckpt, sharding=sharding)
        if obs_action_ckpt is not None:
            obs_action_alg, obs_action_state, _, _ = load_checkpoint(obs_action_ckpt, sharding=sharding)
        else:
            obs_action_alg, obs_action_state = None, None
        pred_fn = functools.partial(
            kl_estimator,
            ks=np.arange(1, 4),
            obs_alg=obs_alg,
            obs_state=obs_state,
            action_alg=action_alg,
            action_state=action_state,
            obs_action_alg=obs_action_alg,
            obs_action_state=obs_action_state,
            obs_weight=1.0,
        )
        repeat, discard_fraction = 4, 0.5
    elif estimator == "nce":
        assert obs_action_ckpt is not None
        obs_action_alg, obs_action_state, _, _ = load_checkpoint(obs_action_ckpt, sharding=sharding)
        pred_fn = functools.partial(nce_estimator, obs_action_alg=obs_action_alg, obs_action_state=obs_action_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "vip":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        pred_fn = functools.partial(vip_estimator, obs_alg=obs_alg, obs_state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "l2":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        pred_fn = functools.partial(l2_loss_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "stddev":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        pred_fn = functools.partial(stddev_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "inv_stddev":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        pred_fn = functools.partial(inverse_stddev_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "compatibility":
        assert obs_ckpt is not None
        obs_alg, obs_state, _, _ = load_checkpoint(obs_ckpt, sharding=sharding)
        pred_fn = functools.partial(compatibility_estimator, alg=obs_alg, state=obs_state)
        repeat, discard_fraction = 1, 0.0
    elif estimator == "mine":
        assert obs_action_ckpt is not None
        obs_action_alg, obs_action_state, _, _ = load_checkpoint(obs_action_ckpt, sharding=sharding)
        pred_fn = functools.partial(mine_estimator, alg=obs_action_alg, state=obs_action_state)
        repeat, discard_fraction = 1, 0.0
    else:
        raise ValueError("Invalid estimator type called: " + estimator)

    def wrapped_pred_fn(batch, rng):
        return pred_fn(batch, rng), {k: batch[k] for k in AGGREGATION_KEYS}

    # Try to load the config from one of the models
    path = next(iter(p for p in (obs_action_ckpt, obs_ckpt, action_ckpt) if p is not None))
    if os.path.basename(os.path.normpath(path)).isdigit():
        path = os.path.dirname(os.path.normpath(path))
    with tf.io.gfile.GFile(tf.io.gfile.join(path, "config.json"), "r") as f:
        config = json.load(f)
        config = ConfigDict(config)

    # Next get the dataloader from the config, set the parameters appropriately.
    dataloader_config = config.dataloader.to_dict()
    dataloader_config["repeat"] = repeat
    dataloader_config["discard_fraction"] = discard_fraction
    if batch_size is not None:
        dataloader_config["batch_size"] = batch_size
    assert dataloader_config["batch_size"] % jax.device_count() == 0

    # Subset the datasets if desired
    if datasets is not None:
        dataloader_config["datasets"] = {k: dataloader_config["datasets"] for k in datasets}
    for ds in dataloader_config["datasets"]:
        if "val_split" in dataloader_config["datasets"][ds]:
            del dataloader_config["datasets"][ds]["val_split"]

    if dataloader_config.get("goal_conditioning", None) is not None:
        dataloader_config["goal_conditioning"] = "last"  # Always set to last frame for preds.
    # Don't recompute statistics for quality estimation, they should already be there.
    dataloader_config["recompute_statistics"] = False

    # Add aggregation keys to structure.
    structure = config.structure.to_dict()
    for k in AGGREGATION_KEYS:
        if k == "dataset_id":
            continue
        structure[k] = None  # Add to the structure with empty field so we include it.

    ds, _, _, dataset_ids = make_dataloader(**dataloader_config, structure=structure, split_for_jax=True)

    return ds, wrapped_pred_fn, dataset_ids


def _aggregate_stats(stats, attrs):
    scores = dict()
    for attr_name, attr in attrs.items():
        scores[attr_name] = dict()
        for attr_val in np.unique(attr).tolist():
            scores[attr_name][attr_val] = np.mean(stats[attr == attr_val])

    if "ep_idx" in attrs and "quality_score" in attrs:
        scores["quality_by_ep_idx"] = dict()
        for ep_idx in np.unique(attrs["ep_idx"]):
            scores["quality_by_ep_idx"][ep_idx] = np.mean(attrs["quality_score"][attrs["ep_idx"] == ep_idx])

    if "ep_idx" in attrs and "quality_score_continuous" in attrs:
        scores["quality_continuous_by_ep_idx"] = dict()
        for ep_idx in np.unique(attrs["ep_idx"]):
            scores["quality_continuous_by_ep_idx"][ep_idx] = np.mean(
                attrs["quality_score_continuous"][attrs["ep_idx"] == ep_idx]
            )

    return scores


def estimate_quality(ds, pred_fn, dataset_ids, rng):
    """
    Note that this is a separate function to allow for jitting of pred_fn and sharding of the ds.
    """
    stats = []
    attributes = {attr: [] for attr in AGGREGATION_KEYS}

    for i, batch in tqdm.tqdm(enumerate(ds), dynamic_ncols=True):
        rng = jax.random.fold_in(rng, i)
        pred, attrs = pred_fn(batch, rng)
        no_nan_idx = ~jnp.isnan(pred)
        stats.append(pred[no_nan_idx])
        for k, v in attrs.items():
            attributes[k].append(v[no_nan_idx])

    # Concatenate everything
    stats = jnp.concatenate(stats, axis=0)
    attributes = {k: jnp.concatenate(v, axis=0) for k, v in attributes.items()}

    # Normalize
    stats = jnp.clip(stats, a_min=jnp.percentile(stats, 1), a_max=jnp.percentile(stats, 99))
    stats = (stats - jnp.mean(stats)) / jnp.std(stats)

    # Do only a single conversion to numpy
    stats = np.array(stats)
    attributes = {k: np.array(v) for k, v in attributes.items()}

    # Now aggregate per dataset
    scores = dict()
    for ds_name, ds_id in dataset_ids.items():
        mask = attributes["dataset_id"] == ds_id
        scores[ds_name] = _aggregate_stats(stats[mask], {k: v[mask] for k, v in attributes.items()})

    return scores

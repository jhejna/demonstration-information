# Additional Dependencies

Packages present in the active `openx` conda environment that are **not listed in `requirements.txt`** or the README installation steps.  
Environment: Python 3.11, CUDA 12, GPU instance.  
Captured: 2026-06-25.

---

## Directly Required (must install explicitly)

These are packages directly imported by the codebase — particularly the MCAP loader, LeRobot dataset loader, and training scripts. They are not pulled in automatically by `pip install -r requirements.txt`.

| Package | Version | Why needed |
|---|---|---|
| `mcap` | 1.4.0 | MCAP (ROS2 bag) file format — `openx/data/lerobot.py` |
| `av` | 17.1.0 | PyAV video decoding — `imageio.imread(..., plugin="pyav")` for MP4 reading |
| `imageio-ffmpeg` | 0.6.0 | ffmpeg backend for imageio — MP4 video decoding |
| `pyarrow` | 24.0.0 | Parquet file reading — LeRobot dataset loader (`pandas.read_parquet`) |
| `pandas` | 3.0.3 | DataFrame operations — LeRobot parquet loading |
| `einops` | 0.8.2 | Tensor manipulation — used in network components |
| `simple-parsing` | 0.1.8 | CLI argument parsing — `scripts/train.py` |
| `orbax-checkpoint` | 0.11.5 | JAX model checkpointing — `openx/utils/evaluate.py` |
| `chex` | 0.1.90 | JAX shape/type assertions and utilities |
| `h5py` | 3.16.0 | HDF5 file reading — used by TF datasets / robomimic data |
| `tensorstore` | 0.1.74 | Required by orbax-checkpoint for array storage |

**Install with:**
```bash
pip install mcap av imageio-ffmpeg pyarrow pandas einops simple-parsing \
            orbax-checkpoint chex h5py tensorstore
```

---

## Version Notes

Packages that are in `requirements.txt` but installed at a **different or more specific version** than specified.

| Package | requirements.txt spec | Installed version | Notes |
|---|---|---|---|
| `numpy` | `<2.0` | `1.26.4` | Pinned — do not upgrade to 2.x (legacy deps) |
| `jax` | `==0.4.37` | `0.4.37` | Matches |
| `jaxlib` | *(not listed)* | `0.4.36` | **One patch behind jax 0.4.37** — installed via the CUDA wheel; must match GPU driver |
| `jax-cuda12-pjrt` | *(not listed)* | `0.4.36` | CUDA 12 JAX plugin — GPU only |
| `jax-cuda12-plugin` | *(not listed)* | `0.4.36` | CUDA 12 JAX plugin — GPU only |
| `scikit-learn` | unpinned | `1.9.0` | Works fine |
| `scipy` | unpinned | `1.17.1` | Works fine |
| `gymnasium` | unpinned | `1.3.0` | Works fine |
| `matplotlib` | unpinned | `3.11.0` | Works fine |
| `seaborn` | unpinned | `0.13.2` | Works fine |
| `wandb` | unpinned | `0.28.0` | Works fine |
| `imageio` | unpinned | `2.37.3` | Works fine |
| `tqdm` | unpinned | `4.68.3` | Works fine |
| `absl-py` | unpinned | `1.4.0` | Works fine |

**Critical note on jax/jaxlib mismatch:**  
`jax==0.4.37` and `jaxlib==0.4.36` are one patch apart. This is because the CUDA 12 wheel (`jax[cuda12_pip]==0.4.37`) installs `jaxlib` at 0.4.36 — the GPU plugin packages (`jax-cuda12-pjrt`, `jax-cuda12-plugin`) bridge the gap. This is the expected behaviour for the GPU install path in the README.

---

## GPU / CUDA Runtime Packages

Installed automatically with `jax[cuda12_pip]`. Listed for reference — do not install manually.

| Package | Version |
|---|---|
| `nvidia-cublas-cu12` | 12.9.2.10 |
| `nvidia-cuda-cupti-cu12` | 12.9.79 |
| `nvidia-cuda-nvcc-cu12` | 12.9.86 |
| `nvidia-cuda-nvrtc-cu12` | 12.9.86 |
| `nvidia-cuda-runtime-cu12` | 12.9.79 |
| `nvidia-cudnn-cu12` | 9.23.2.1 |
| `nvidia-cufft-cu12` | 11.4.1.4 |
| `nvidia-cusolver-cu12` | 11.7.5.82 |
| `nvidia-cusparse-cu12` | 12.5.10.65 |
| `nvidia-nccl-cu12` | 2.30.7 |
| `nvidia-nvjitlink-cu12` | 12.9.86 |

---

## Transitive Dependencies

Auto-installed as dependencies of the above — no manual installation needed.

| Package | Version | Pulled in by |
|---|---|---|
| `annotated-types` | 0.7.0 | pydantic |
| `array_record` | 0.8.3 | tensorflow-datasets |
| `astunparse` | 1.6.3 | tensorflow |
| `attrs` | 26.1.0 | various |
| `certifi` | 2026.6.17 | requests |
| `cfgv` | 3.5.0 | pre-commit |
| `charset-normalizer` | 3.4.7 | requests |
| `click` | 8.4.2 | wandb |
| `cloudpickle` | 3.1.2 | orbax / gym |
| `contourpy` | 1.3.3 | matplotlib |
| `cycler` | 0.12.1 | matplotlib |
| `decorator` | 5.3.1 | various |
| `distlib` | 0.4.3 | virtualenv |
| `dm-tree` | 0.1.10 | tensorflow / jax |
| `docstring_parser` | 0.18.0 | simple-parsing |
| `etils` | 1.14.0 | jax |
| `Farama-Notifications` | 0.0.6 | gymnasium |
| `filelock` | 3.29.4 | various |
| `flatbuffers` | 25.12.19 | tensorflow |
| `fonttools` | 4.63.0 | matplotlib |
| `fsspec` | 2026.6.0 | tensorflow-datasets / pandas |
| `gast` | 0.7.0 | tensorflow |
| `gitdb` | 4.0.12 | GitPython |
| `GitPython` | 3.1.50 | wandb |
| `google-pasta` | 0.2.0 | tensorflow |
| `googleapis-common-protos` | 1.73.0 | grpcio |
| `grpcio` | 1.81.1 | tensorflow / wandb |
| `humanize` | 4.15.0 | wandb |
| `identify` | 2.6.19 | pre-commit |
| `idna` | 3.18 | requests |
| `immutabledict` | 4.3.1 | jax |
| `joblib` | 1.5.3 | scikit-learn |
| `keras` | 3.15.0 | tensorflow |
| `kiwisolver` | 1.5.0 | matplotlib |
| `libclang` | 18.1.1 | tensorflow |
| `lz4` | 4.4.5 | tensorstore |
| `Markdown` | 3.10.2 | tensorboard |
| `markdown-it-py` | 4.2.0 | rich |
| `MarkupSafe` | 3.0.3 | Werkzeug |
| `mdurl` | 0.1.2 | markdown-it-py |
| `ml-dtypes` | 0.4.1 | jax / tensorflow |
| `namex` | 0.1.0 | keras |
| `narwhals` | 2.22.1 | pandas |
| `nest-asyncio` | 1.6.0 | tensorflow-datasets |
| `networkx` | 3.6.1 | tensorflow-datasets |
| `nodeenv` | 1.10.0 | pre-commit |
| `OpenEXR` | 3.4.13 | tensorflow-graphics |
| `opt_einsum` | 3.4.0 | jax / tensorflow |
| `optree` | 0.19.1 | jax / optax |
| `packaging` | 26.2 | various |
| `pillow` | 12.2.0 | imageio |
| `platformdirs` | 4.10.0 | virtualenv |
| `promise` | 2.3 | tensorflow-datasets |
| `protobuf` | 4.25.9 | tensorflow |
| `psutil` | 7.2.2 | wandb |
| `pydantic` | 2.13.4 | wandb / sentry-sdk |
| `pydantic_core` | 2.46.4 | pydantic |
| `Pygments` | 2.20.0 | rich |
| `pyparsing` | 3.3.2 | matplotlib |
| `python-dateutil` | 2.9.0.post0 | pandas |
| `python-discovery` | 1.4.2 | simple-parsing |
| `PyYAML` | 6.0.3 | wandb / pre-commit |
| `requests` | 2.34.2 | wandb / tensorflow-datasets |
| `rich` | 15.0.0 | wandb |
| `sentry-sdk` | 2.63.0 | wandb |
| `simplejson` | 4.1.1 | wandb |
| `six` | 1.17.0 | various |
| `smmap` | 5.0.3 | gitdb |
| `tensorboard` | 2.17.1 | tensorflow |
| `tensorboard-data-server` | 0.7.2 | tensorboard |
| `tensorflow-addons` | 0.23.0 | tensorflow-datasets |
| `tensorflow-io-gcs-filesystem` | 0.37.1 | tensorflow |
| `tensorflow-metadata` | 1.15.0 | tensorflow-datasets |
| `termcolor` | 3.3.0 | tensorflow |
| `threadpoolctl` | 3.6.0 | scikit-learn |
| `toml` | 0.10.2 | pre-commit |
| `toolz` | 1.1.0 | jax / optax |
| `trimesh` | 4.12.2 | tensorflow-graphics |
| `typeguard` | 2.13.3 | tensorflow-datasets |
| `typing_extensions` | 4.15.0 | various |
| `typing-inspection` | 0.4.2 | pydantic |
| `urllib3` | 2.7.0 | requests |
| `virtualenv` | 21.5.1 | pre-commit |
| `Werkzeug` | 3.1.8 | tensorboard |
| `wrapt` | 2.2.2 | tensorflow |
| `zipp` | 4.1.0 | various |
| `zstandard` | 0.25.0 | tensorstore / mcap |

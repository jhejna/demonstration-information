# Override with correct values for your system if non-default.
CONDA_PATH=path/to/miniconda3/bin/activate
ENV_NAME=openx
REPO_PATH=path/to/demonstration-info
USE_MUJOCO_PY=true # For using mujoco py with RoboMimic
WANDB_API_KEY="" # If you want to use wandb, set this to your API key.

# Setup Conda
source $CONDA_PATH
conda activate $ENV_NAME
cd $REPO_PATH
unset DISPLAY # Make sure display is not set or it will prevent scripts from running in headless mode.

if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Using WandB."
    export WANDB_API_KEY=$WANDB_API_KEY
fi

if $USE_MUJOCO_PY; then
    echo "Using mujoco_py"
    if [ -d "/usr/lib/nvidia" ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
fi

# First check if we have a GPU available
if nvidia-smi | grep "CUDA Version"; then
    if [ -d "/usr/local/cuda-12.3" ]; then
        export PATH=/usr/local/cuda-12.3/bin:$PATH
    elif [ -d "/usr/local/cuda-12.2" ]; then
        export PATH=/usr/local/cuda-12.2/bin:$PATH
    elif [ -d "/usr/local/cuda-12.1" ]; then
        export PATH=/usr/local/cuda-12.1/bin:$PATH # This is the lowest compatible version with current jax.
    elif [ -d "/usr/local/cuda" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        echo "Using default CUDA, compatibility should be verified."
    else
        echo "Warning: Could not find a CUDA version but GPU was found."
    fi
    export MUJOCO_GL="egl"
    # Setup any GPU specific flags
else
    echo "GPU was not found, assuming CPU setup."
    export MUJOCO_GL="osmesa" # glfw doesn't support headless rendering
fi

if nvidia-smi | grep "No devices were found"; then
    export JAX_PLATFORMS=cpu
fi

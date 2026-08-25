# TACC Setup for World Model Training

## Files To Upload

Upload these code files to the same folder on TACC:

```text
sample_paired_h5ad_dataloader.py
five_level_cell_world_model.py
chreode_loss.py
train_five_level_world_model.py
```

Upload these notes as documentation:

```text
model architectures.md
sample_paired_h5ad_dataloader_notes.md
TACC_SETUP.md
```

Also upload or place the paired h5ad dataset folder on TACC, for example:

```text
paired_training_h5ad_50k/
  paired_h5ad_manifest.tsv
  genes.tsv
  shards/
    paired_expr_shard_00000.h5ad
    ...
```

## Conda Environment

Recommended environment name:

```text
worldmodeltraining
```

Important Grace Hopper note:

```text
Vista GH nodes use an NVIDIA Grace CPU, so Conda reports:

Platform: linux-aarch64
```

That is expected. It is not an error. It means the CPU side is ARM/aarch64, not x86_64. Do not force `linux-64` while running on a GH node.

If `conda create` fails with:

```text
Failed to resolve repo.anaconda.com
```

that is a network/DNS issue, not an architecture issue. Create/install the environment from a network-enabled login node, use TACC-provided modules, or use a TACC-supported container workflow.

Create it when network access is available:

```bash
conda create -n worldmodeltraining python=3.11 -y
conda activate worldmodeltraining
python -m pip install --upgrade pip setuptools wheel
```

### If Conda Cannot Reach The Internet

If you see:

```text
Failed to resolve repo.anaconda.com
```

you are probably on a compute node without outbound DNS/network access. Leave the compute node and create the environment from a login node, or clone the already-working `moscot_env`.

Option A, clone existing environment:

```bash
conda create -n worldmodeltraining --clone moscot_env
conda activate worldmodeltraining
python -m pip install --upgrade pip setuptools wheel
```

Then check what is already available:

```bash
python - <<'PY'
mods = ["torch", "anndata", "pandas", "numpy", "scipy", "scvi", "scanpy"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "ok", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "missing", e)
PY
```

Option B, create in project storage from a login node with conda-forge:

```bash
mkdir -p $WORK/conda_envs
conda create -p $WORK/conda_envs/worldmodeltraining \
  -c conda-forge --override-channels \
  python=3.11 pip numpy pandas scipy anndata h5py tqdm -y
conda activate $WORK/conda_envs/worldmodeltraining
python -m pip install --upgrade pip setuptools wheel
```

Use `conda-forge` because GH reports `linux-aarch64`, and conda-forge usually has better aarch64 coverage than defaults.

## PyTorch For H200

For Vista Grace Hopper / H200 nodes, prefer a TACC-provided PyTorch module or NVIDIA/PyTorch container built for Grace Hopper. This avoids ARM/aarch64 wheel problems.

First check available modules:

```bash
module spider pytorch
module spider cuda
module spider python
```

If TACC provides a PyTorch module, use that instead of installing PyTorch manually:

```bash
module load pytorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you must install with pip, only do this after confirming the wheel supports linux-aarch64 on the GH node:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If that fails because no compatible aarch64 wheel is available, use the TACC module/container path.

Check GPU visibility:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

Expected GPU should be an NVIDIA H200.

## Python Packages

Install packages needed by the current dataloader and training scripts:

```bash
pip install numpy pandas scipy anndata h5py tqdm
```

Install scVI tooling for the next scVI latent pipeline:

```bash
pip install scvi-tools scanpy
```

Optional but useful:

```bash
pip install matplotlib seaborn tensorboard
```

## Quick Import Check

Run this from the folder containing the uploaded Python files:

```bash
python - <<'PY'
import torch
import anndata
import pandas
import scipy
import scvi
from sample_paired_h5ad_dataloader import build_paired_h5ad_loader
from five_level_cell_world_model import build_model
from chreode_loss import ChreodeLossConfig
print("imports ok")
print("cuda:", torch.cuda.is_available())
PY
```

## First Smoke Test

Use a small step count first:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 2 \
  --batch-size 32 \
  --max-steps 20 \
  --sinkhorn-iters 20
```

Expected outputs:

```text
checkpoints_level2/run_config.json
checkpoints_level2/step_losses.csv
checkpoints_level2/epoch_losses.csv
checkpoints_level2/world_model_level2_step0000020.pt
```

## First Model-Selection Runs

Level 2 DiT residual:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 2 \
  --batch-size 64 \
  --epochs 5 \
  --max-steps -1
```

Level 3 gated DiT residual:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 3 \
  --batch-size 64 \
  --epochs 5 \
  --max-steps -1
```

Outputs are automatically level-marked:

```text
checkpoints_level2/
checkpoints_level3/
```

Each folder contains:

```text
run_config.json
step_losses.csv
epoch_losses.csv
world_model_level{level}_step{step}.pt
```

## Suggested H200 Settings

The H200 has enough memory to try larger batches after the smoke test:

```text
batch_size: 128 or 256
model_dim: 256
depth: 6
heads: 4
num_tokens: 4
```

Example Tiny-like DiT setting:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 3 \
  --batch-size 128 \
  --model-dim 256 \
  --depth 6 \
  --heads 4 \
  --num-tokens 4 \
  --epochs 5 \
  --max-steps -1
```

## What To Inspect

For model selection, compare:

```text
checkpoints_level2/epoch_losses.csv
checkpoints_level3/epoch_losses.csv
```

The first goal is simple:

```text
loss should trend downward across epochs.
loss_mmd, loss_w2, and loss_drift should not diverge or become NaN.
```

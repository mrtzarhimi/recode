# Start-to-finish Colab instructions

After pushing this repository to GitHub, open a fresh Colab notebook and use
these cells. Replace `YOUR_GITHUB_USER/YOUR_REPO` and the Drive file paths.

## 1. Clone the pushed repository

```bash
!git clone https://github.com/YOUR_GITHUB_USER/YOUR_REPO.git
%cd YOUR_REPO
```

## 2. Mount Drive and check the inputs

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!ls -lh "/content/drive/MyDrive/nomad"
```

The folder must contain the trained Day 0 checkpoint (`model_day0.pt`) and one
prepared format-6 `.npz` reference/target archive for each Day K session. The
NPZs need to match the checkpoint's Day 0 reference and preprocessing. Raw
session data alone is not enough for this sweep; prepare the archives first
with this repository's `dataloader.py` if needed.

## 3. Install dependencies and confirm GPU

```bash
!python -m pip install -r requirements.txt
!nvidia-smi
```

In Colab, select **Runtime → Change runtime type → T4 GPU** (or another CUDA
GPU) before starting the sweep. Training and per-checkpoint validation are
compute- and storage-intensive; write outputs to Drive so they survive a
runtime reset.

## 4. Run the study

```bash
!python run_reconstruction_sweep.py \
  --ckpt0 "/content/drive/MyDrive/nomad/model_day0.pt" \
  --target "Jango_20150731=/content/drive/MyDrive/nomad/target_0731.npz" \
  --target "Jango_20150809=/content/drive/MyDrive/nomad/target_0809.npz" \
  --target "Jango_20150827=/content/drive/MyDrive/nomad/target_0827.npz" \
  --target "Jango_20150908=/content/drive/MyDrive/nomad/target_0908.npz" \
  --target "Jango_20151029=/content/drive/MyDrive/nomad/target_1029.npz" \
  --target "Jango_20151102=/content/drive/MyDrive/nomad/target_1102.npz" \
  --out_dir "/content/drive/MyDrive/nomad/reconstruction_sweep" \
  --weights 0,1,10,100 \
  --include_reconstruction_only \
  --seeds 0,1,2 \
  --split_seed 0 \
  --epochs 100 \
  --device auto
```

For a short pipeline check, use one target, `--seeds 0`, and `--epochs 2` in a
separate output directory. Do not use that run for scientific conclusions.

## Copyable request for a Colab coding assistant

> Starting from a fresh Colab runtime, clone my pushed repository at
> `https://github.com/YOUR_GITHUB_USER/YOUR_REPO.git`, mount Google Drive, and
> change into the cloned repository. My Day 0 checkpoint and prepared format-6
> target archives are in `/content/drive/MyDrive/nomad`. Install the repository
> requirements, confirm that the checkpoint and all six NPZ paths exist, and
> confirm a GPU is available. If an input is missing or incompatible, stop and
> report exactly which file is needed; do not substitute data or fabricate
> results. Run the command in this file with all six sessions, reconstruction
> weights 0, 1, 10, and 100 at alignment weight 1, plus the reconstruction-only
> endpoint (alignment weight 0, reconstruction weight 10), and paired seeds
> 0, 1, and 2. Keep split seed 0 across every condition. Save all output to
> `/content/drive/MyDrive/nomad/reconstruction_sweep`. Choose candidates using
> validation only, then score the neural-selected, behavior-selected, and
> NoMAD-default candidate on test trials. Do not change the protocol after
> inspecting test scores. Preserve `sweep_results.json`, per-checkpoint
> validation reports, selected test reports, and the run logs. Report each
> session's test R² means, seed variation, and paired differences from the
> NoMAD default, and state any failed or incomplete sessions.

The output includes `sweep_results.json`, validation reports, selected test
reports, and saved epoch checkpoints. The primary runner does not yet implement
the fresh Day K decoder probe, masked-channel reconstruction analysis, or
pseudo-session control described in the study outline.

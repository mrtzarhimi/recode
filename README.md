# PyTorch NoMAD

This repository ports the executable LFADS/NoMAD design at upstream commit [`aa3c8faadf053d0b22160458854c076dcb13eb7e`](https://github.com/snel-repo/nomad/tree/aa3c8faadf053d0b22160458854c076dcb13eb7e). Source findings, paper/demo/default distinctions, deliberate extensions, and remaining differences are in [ORIGINAL_ALIGNMENT.md](ORIGINAL_ALIGNMENT.md).

The workflow prepares each XDS session once, trains one Day-0 model, reuses its exact checkpoint for independent Day-K alignments, and reports unchanged transfer, normalization-only adaptation, and full alignment on identical held-out target trials and bins. Day-K behavior is not used for default unsupervised alignment or decoder fitting.

## Install

```bash
python -m pip install -r requirements.txt
```

MATLAB v7.3 loading uses `h5py`; XDS is loaded from the local `limblab/xds` checkout. Preparing raw data requires a supported XDS `.mat` archive. Old processed archives are not compatible with this pipeline.

## Prepare data

Prepare a reference-only archive and one pair archive per target. The reference file, binning, unit order, behavior names, and preprocessing options must be identical in every pair archive.

```bash
python dataloader.py --xds_path /path/to/xds --data_dir /path/to/data \
  --day0 Reference.mat --no_emg --out data/reference.npz

python dataloader.py --xds_path /path/to/xds --data_dir /path/to/data \
  --day0 Reference.mat --dayk Target1.mat --no_emg --out data/target1.npz

python dataloader.py --xds_path /path/to/xds --data_dir /path/to/data \
  --day0 Reference.mat --dayk Target2.mat --no_emg --out data/target2.npz
```

The pinned demo uses 20 ms bins, continuous-session 600 ms windows, and 120 ms overlap. The paper specifies a 500th-order Chebyshev anti-alias filter for continuous-valued data before downsampling. The paper does not state Chebyshev type, passband ripple, or cutoff in the accessible Methods text. This implementation uses a zero-phase SOS Chebyshev Type I filter with 0.05 dB ripple and cutoff at 0.8 times the target Nyquist frequency, then samples the native behavior grid at the neural timestamps. The order is source-specified; the filter type, ripple, cutoff, and zero-phase operation are explicit implementation choices. Settings are stored in each archive and included in its fingerprint.

Filtering is performed separately over each trial-owned continuous interval, including its following intertrial gap. Thus no filter state or padding can draw data from another interval; since whole intervals belong to a single train, validation, or test partition, filtering cannot cross partition boundaries. Zero-phase filtering uses interval-local odd reflection with a fixed 500-native-sample pad. Intervals of 500 samples or fewer are rejected rather than filtered with a silently changed order or fabricated edge values. Pad length is an implementation choice and is recorded in the archive. Neural spike counts continue to be summed into larger bins, as described by the paper.

Force derivatives are backward differences divided by the bin duration, in XDS force units per second. They include intertrial activity and reset to zero at each trial-owned interval start, so held-out values cannot affect training derivatives. The pinned repository contains prepared behavior arrays but does not publish their raw derivative construction. Exact equivalence to its derivative units is therefore unverified. The old ±100 derivative cutoff was applied without that verification and has been removed; it is not silently converted to a larger threshold.

The archive applies force cutoffs x `[-1500, 800]`, y `[-1100, 600]` to individual behavior training bins. The next bin is also masked when its derivative depends on an invalid preceding force sample, except at reset boundaries. Evaluation excludes whole trials containing force outliers. One bad bin does not mask the rest of its trial or its following intertrial activity during training. All spikes remain available to unsupervised reconstruction/alignment. Loading for unsupervised alignment does not require target behavior inliers; supervised training/validation and final evaluation still require eligible behavior observations at their respective stages. This force-based mask is an explicit extension while the source derivative convention remains unresolved.

The optional `--corr_thresh` removes highly correlated units after estimating correlations from Day-0 training intervals; it is a pipeline extension at the selected bin resolution. The default is disabled. Selected unit names and indices are saved with the reference checkpoint, and missing target units are zero-padded by unit name.

## Train and align multiple days

```bash
python cross_day.py --reference data/reference.npz \
  --targets data/target1.npz data/target2.npz --out_dir runs/multiday
```

This trains Day 0 once and keeps target splits, calibration, checkpoints, and reports in distinct per-day directories. To align additional days from an existing Day-0 checkpoint:

```bash
python cross_day.py --ckpt0 runs/multiday/reference/model_day0.pt \
  --targets data/target3.npz data/target4.npz --out_dir runs/additional
```

The default split is 60% training, 20% validation, and 20% test by whole trial. Each trial and its following intertrial interval are assigned to the same partition; continuous windows are then formed only inside a partition. This prevents overlapping windows or sequence context from crossing partitions. The validation partition selects checkpoints. The test partition is used only for final reporting.

Useful `cross_day.py` options include `--epochs0`, `--epochsk`, `--bs0`, `--bsk`, `--kl_warmup`, `--l2_warmup`, `--beh_warmup`, `--nll_warmupk`, `--kl_warmupk`, `--device`, `--threads`, and `--amp bf16`. Each training script also exposes loss weights, learning rates, optimizer settings, clipping, and model dimensions. Checkpoints record the complete CLI configuration.

The alignment width defaults to `readin_dim` (50 in the sourced demo configuration). An explicitly different `align_dim` is rejected because its output feeds the frozen encoders.

## Run the stages individually

```bash
python train_phase1.py --data data/reference.npz --out_dir runs/reference

python train_phase2.py --data data/target1.npz \
  --ckpt runs/reference/model_day0.pt --out_dir runs/target1

python evaluate.py --data data/target1.npz \
  --ckpt0 runs/reference/model_day0.pt \
  --ckptk runs/target1/model_dayk.pt \
  --out runs/target1/test_results.json
```

Use `evaluate.py --split validation` for development. Do not adjust settings after inspecting test results. Optional target-behavior alignment is supervised and is labeled as such in output; it is off by default.

## Evaluation and provenance

- **Unchanged Day-0 transfer** applies the unaligned Day-0 model using Day-0 neural normalization.
- **Normalization-only** applies that same model using target-training neural statistics. This preserves the previous unaligned baseline as a distinct, useful condition.
- **Full NoMAD** applies the aligned model using target-training statistics. The Day-0 ridge decoder is identical across all three conditions.
- Decoder ridge penalties are selected from Day-0 training trials only, with trial-grouped CV. Input lags run oldest-to-newest and include the prediction-time bin. The score uses the pinned notebook's variance-weighted R² convention, with finite handling for constant targets.
- All three target conditions use the same eligible held-out trials and bins. Behavior is scored in original units without clipping. Movement-onset selection is retrospective; features use causal windows wholly inside each trial.
- Day-0 and target fingerprints include timestamp arrays. Checkpoints also record channel identities, mappings, preprocessing settings, trial splits, normalization statistics, and the exact source-model fingerprint. Frozen LFADS dynamics are compared against the Day-0 state before evaluation.
- This split and decoder protocol is stricter than the paper demo: the demo reserves overlapping chopped segments for validation and randomly divides evaluation trials, while this port uses disjoint whole-trial partitions and a separate test partition. Causal inference also requires the full 30-bin context inside each evaluation trial; the notebook uses initial overlapping windows with lookahead for the first 29 bins.

## Build a multi-day results report

`generate_report.py` creates a static, offline dashboard from one or more `cross_day_summary.json` files, individual `evaluate.py` reports, or a previously exported report bundle. It does not retrain models or recalculate scores. Open the generated `index.html` directly in a browser, then use **Add results** to append evaluation JSONs or summaries. Browser additions can be downloaded as a normalized JSON bundle and reused later.

For a cross-day run with one model seed:

```bash
python generate_report.py \
  --summary runs/multiday/cross_day_summary.json \
  --out runs/report
```

For separate seed runs, pass each run summary. Day-K labels and evaluation seeds come from their embedded reports:

```bash
python generate_report.py \
  --summary runs/seed0/cross_day_summary.json \
  --summary runs/seed1/cross_day_summary.json \
  --summary runs/seed2/cross_day_summary.json \
  --out runs/report
```

For standalone reports or to explicitly set a run seed, use `DAYK@SEED=PATH`. The `@SEED` override is optional; without it the generator uses `args.seed` in that evaluation JSON. This matters when `args.seed` is the behavior-decoder CV seed rather than the NoMAD training seed.

```bash
python generate_report.py \
  --result 'Jango_20150731@0=runs/target1/evaluation.json' \
  --result 'Jango_20150731@1=runs/target1_seed1/evaluation.json' \
  --result 'Jango_20150809@0=runs/target2/evaluation.json' \
  --out runs/report
```

Add a downloaded bundle with `--bundle runs/nomad-results-bundle.json`, or combine bundles and new reports by repeating inputs. The dashboard keeps test and validation results separate. Its default test view reports each Day-K's mean and sample SD across seeds. Overall performance is the mean of the Day-K means, with equal weight for every Day-K; its SD is across Day-K means. When only one sample contributes, SD is shown as undefined (`—`). A missing condition stays missing; it is never imputed.

The comparison keeps unchanged Day-0 transfer separate from target-training normalization and full alignment. Normalization-only uses target training statistics and is not zero-shot. Overall NoMAD is labeled supervised if its source reports identify supervised alignment. Results remain descriptive: compare matching days, splits, metrics, readouts, and eligible trials before interpreting a gain.

## Migration and tests

Format-3/4/5 archives and checkpoints must be regenerated and retrained using format 6. Rerun every preparation command, including the reference-only archive and every reference/target pair, with the same reference and options. Use a fresh output directory such as `runs/multiday_v6`. Format 6 records derivative definition, behavior mask policy, the anti-alias filter settings, timestamps, and split provenance; these enter the dataset fingerprint. Existing arrays cannot be relabeled as equivalent.

Regression tests are in `test_pipeline.py`. They cover anti-alias attenuation, isolation between filtered intervals, native-rate passthrough, split isolation, preprocessing invariance, derivative units and boundaries, sample versus trial masks, target loading without behavior inliers, timestamp gaps and XDS behavior, objective reductions and zero regularization, KL reference values, alignment widths/freezing, causal boundaries, normalization conditions, and reference reuse. A user-supplied Colab run of the previous version reported four errors and one dtype assertion failure; those issues were addressed in source. This update did not execute project code or rerun the tests, as requested; no passing result or real-data performance is claimed.

```bash
python -m unittest test_pipeline -v
```

## Reconstruction versus decoder preservation experiment

`run_reconstruction_sweep.py` runs matched Day-K alignments over target sessions,
seeds, and objective weights. It saves every epoch checkpoint, evaluates
unweighted spike reconstruction and the frozen Day-0 Wiener force decoder on
validation trials, selects candidates independently by each validation score,
then reports test force R² for those selections and the prespecified NoMAD
default. The test split is evaluated only after validation selection.

See [COLAB_RUN_PROMPT.md](COLAB_RUN_PROMPT.md) for a Colab prompt and complete
command template. The sweep defaults to `align_w=1` and `rec_w=0,1,10,100`; add
`--include_reconstruction_only` for `align_w=0, rec_w=10`. Paired seeds and a
fixed `--split_seed` keep target trial partitions common across conditions.
`--epochs` sets a common update budget; use a smaller value for a smoke run and
the prespecified budget for the study.

The current runner implements the primary fixed-decoder comparison. A fresh
Day-K decoder probe, masked-channel reconstruction analysis, and Day-0
pseudo-session control from the study outline remain follow-up analyses; the
sweep does not include them, and its results cannot support those claims.

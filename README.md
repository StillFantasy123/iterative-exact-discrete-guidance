# Iterative Exact Discrete Guidance

This is the evaluation release for *Iterative Exact Discrete Guidance* (IEDG).
It contains the selected inference checkpoints, frozen evaluation inputs,
checkpoint-compatible evaluation configurations, and the code needed to
recompute the released metrics or rerun the formal checkpoint evaluations.
Training code will be released soon.

## Install

The pinned GPU environment is defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate iedg-paper
pip install -e . --no-build-isolation
```

All artifacts are bundled under `artifacts/`; evaluation requires no download.
`artifacts/manifest.yaml` is the machine-readable source of truth for result
IDs, configurations, checkpoints, seeds, sample budgets, and test inputs.
Run the commands below from the repository root.

## Level 1: recompute metrics from frozen samples

This CPU command validates all 20 released checkpoint payloads, strictly loads
their IEDG/One-shot state dictionaries into the registered models,
and recomputes the paper metrics for the six IEDG `16x16` lattice results from
their frozen generated/reference sample pairs:

```bash
iedg-validate-release
```

It fails if a checkpoint does not match its registered model, if a manifest
path or protocol field is inconsistent, or if a recomputed metric differs from
the bundled metric record. The emitted JSON includes the recomputed metrics.

## Level 2: resample released checkpoints

These commands perform new terminal rollouts with the manifest-registered test
protocol. A CUDA GPU is recommended. Output paths must be new.

### Ising and Potts `16x16`

```bash
for result in \
  ising16_beta028 ising16_beta04407 ising16_beta06 \
  potts16_beta05 potts16_beta1005 potts16_beta12 \
  one_shot_ising16_beta028 one_shot_ising16_beta04407 one_shot_ising16_beta06 \
  one_shot_potts16_beta05 one_shot_potts16_beta1005 one_shot_potts16_beta12
do
  iedg-lattice-evaluate \
    --result "$result" \
    --device cuda \
    --output-dir "evaluation/$result"
done
```

### Exact Ising `4x4`

```bash
for result in one_shot_ising4_beta06 iedg_ising4_beta06
do
  iedg-ising4-evaluate \
    --result "$result" \
    --device cuda \
    --output "evaluation/${result}.json"
done
```

### Max-Cut

```bash
for result in \
  one_shot_maxcut_ba20 iedg_maxcut_ba20 \
  one_shot_maxcut_ba40 iedg_maxcut_ba40 \
  one_shot_maxcut_ba100 iedg_maxcut_ba100
do
  iedg-maxcut-evaluate \
    --result "$result" \
    --device cuda \
    --output-dir "evaluation/$result"
done

iedg-maxcut-uniform \
  --output-dir evaluation/maxcut_uniform
```

The Max-Cut test manifests contain 32 fixed graphs per scale and their certified
optima, so Gurobi is not required for evaluation.

## Scope and reproducibility

The six released IEDG lattice checkpoints were trained on NVIDIA RTX A6000 or
RTX PRO 6000 GPUs as recorded in the manifest. Reported evaluation rollouts
used NVIDIA RTX A6000 GPUs. Exact paper values are defined by the frozen sample
payloads in Level 1; Level 2 verifies that the checkpoints can independently
generate and evaluate new samples under the same protocol. Bitwise equality
across GPU architectures is not expected.

## RoPE provenance

The mixed two-dimensional rotary-position utilities in
`edg_experiment/lattice/models/backbone_mdns_rope_deit2d.py` are adapted from
the [RoPE-ViT implementation by NAVER AI Lab](https://github.com/naver-ai/rope-vit). 
They are integrated here with the lattice backbone used for categorical-state inference.

## Layout

```text
artifacts/                 checkpoints, frozen samples, and release manifest
configs/                   compact evaluation configurations referenced by manifest
edg_experiment/            lattice inference, metrics, and release validation
ising4_experiment/         exact Ising 4x4 evaluation
ising16_experiment/        Ising 16x16 metrics
potts16_experiment/        Potts 16x16 metrics
CO_experiment/             Max-Cut inference, metrics, and fixed test graphs
```

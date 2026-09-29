# NEPTUNE v1.2.3-H1 — Local Server Run Guide

This README describes the minimal workflow for running **NEPTUNE v1.2.3-H1** on a local GPU server such as an RTX 3090.

## 1. Recommended directory structure

```text
NEPTUNE/
├── code/
│   ├── neptune-v1.2.3-h1.zip
│   └── neptune/                    # unpacked source code
│
├── data/
│   ├── raw/
│   │   └── ml-25m/
│   │       ├── ratings.csv
│   │       ├── movies.csv
│   │       ├── genome-scores.csv
│   │       └── ...
│   │
│   └── processed/
│       └── h1_preview_2027/
│           └── 4f223a6dc0c3/
│               ├── events.parquet
│               ├── item_features.npy
│               ├── movie_ids.npy
│               ├── popularity.npy
│               ├── cold_items.npy
│               ├── warm_items.npy
│               ├── user_ids.npy
│               └── manifest.json
│
├── profiles/
│   └── h1_preview_2027.yaml
│
├── launch/
│   ├── run_one.py
│   ├── run_matrix.sh
│   └── README.md
│
└── artifacts/
    └── runs/
        └── ...
```

`artifacts/` stores complete run outputs, including checkpoints. A separate `checkpoint/` directory is not necessary.

---

## 2. What is required to train

For training from an already prepared dataset, the server only needs:

1. NEPTUNE source code.
2. Python environment with dependencies installed.
3. Processed dataset directory.
4. Profile YAML.
5. Run parameters: `M`, `h`, `sigma`, `seed`, negative sampler, time mode.
6. Output directory.

The raw ML-25M files are **not required for training** if the processed dataset already exists.

The actual training call conceptually is:

```text
CODE + PROCESSED_DATA + PROFILE + SWEEP_CONFIG + OUTPUT_DIR -> TRAIN
```

---

## 3. First-time environment setup

Assume the project root is:

```bash
export NEPTUNE_ROOT=/home/user/NEPTUNE
```

Unpack the source code if needed:

```bash
cd $NEPTUNE_ROOT/code
unzip neptune-v1.2.3-h1.zip
```

Enter the repo:

```bash
cd $NEPTUNE_ROOT/code/neptune
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

Check CUDA / GPU:

```bash
nvidia-smi
```

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

For a GPU server this should print something like:

```text
True
NVIDIA GeForce RTX 3090
```

---

## 4. Verify the exact NEPTUNE code version

Run:

```bash
cd $NEPTUNE_ROOT/code/neptune
python scripts/fingerprint.py
```

For v1.2.3-H1 the expected implementation fingerprint is:

```text
a7eddb3ac27bb5aeea1e8f2aa764b4a03fa5b8baa114e0603eaa135086ae0fa3
```

Do not compare experimental results across machines if they use different implementation fingerprints unless that difference is intentional and documented.

---

## 5. Exploratory H1 profile

Example file:

```text
$NEPTUNE_ROOT/profiles/h1_preview_2027.yaml
```

Contents:

```yaml
name: h1_preview_2027
claim_eligible: false

may_override:
  - data.n_users

overrides:
  data:
    n_users: 4000
```

This is an exploratory profile and is not a claim-eligible preregistered H1 run.

---

## 6. Processed data

Recommended path:

```text
$NEPTUNE_ROOT/data/processed/h1_preview_2027/4f223a6dc0c3/
```

It should contain at least:

```text
events.parquet
item_features.npy
movie_ids.npy
popularity.npy
cold_items.npy
warm_items.npy
user_ids.npy
manifest.json
```

If these files already exist, **do not preprocess ML-25M again**.

Set the path once:

```bash
export DATA=$NEPTUNE_ROOT/data/processed/h1_preview_2027/4f223a6dc0c3
export OUT=$NEPTUNE_ROOT/artifacts
```

---

## 7. Preparing data only when necessary

If no processed dataset exists, raw ML-25M should be at:

```text
$NEPTUNE_ROOT/data/raw/ml-25m/
```

with at least:

```text
ratings.csv
movies.csv
genome-scores.csv
```

Then run:

```bash
cd $NEPTUNE_ROOT/code/neptune
source .venv/bin/activate

python scripts/prepare_ml25m.py \
  --profile $NEPTUNE_ROOT/profiles/h1_preview_2027.yaml \
  --raw-dir $NEPTUNE_ROOT/data/raw/ml-25m \
  --data $NEPTUNE_ROOT/data/processed/h1_preview_2027/4f223a6dc0c3 \
  --out $NEPTUNE_ROOT/artifacts
```

For multi-machine training, prepare the dataset once and copy the processed artifact to each machine.

---

## 8. Running one experiment

Assuming `launch/run_one.py` has been created, activate the environment:

```bash
export NEPTUNE_ROOT=/home/user/NEPTUNE
export DATA=$NEPTUNE_ROOT/data/processed/h1_preview_2027/4f223a6dc0c3
export OUT=$NEPTUNE_ROOT/artifacts

source $NEPTUNE_ROOT/code/neptune/.venv/bin/activate
```

Example: `M=16`, `h=0.20`, `sigma=0.30`, `seed=2027`:

```bash
python $NEPTUNE_ROOT/launch/run_one.py \
  --data $DATA \
  --out $OUT \
  --M 16 \
  --h 0.20 \
  --sigma 0.30 \
  --seed 2027
```

Change only the command-line values to launch another sweep point.

Examples:

```bash
--M 1
--M 2
--M 4
--M 8
--M 16
```

```bash
--h 0.20
--h 0.35
--h 0.50
```

```bash
--sigma 0.20
--sigma 0.30
--sigma 0.45
```

```bash
--seed 2027
--seed 2028
--seed 2029
```

---

## 9. Running a matrix of experiments

Example matrix:

```text
M     = {1, 16}
h     = {0.20, 0.50}
sigma = 0.30
seed  = 2027
```

Use:

```bash
for M in 1 16; do
  for H in 0.20 0.50; do
    echo "=========================================="
    echo "RUN M=$M h=$H sigma=0.30 seed=2027"
    echo "=========================================="

    python $NEPTUNE_ROOT/launch/run_one.py \
      --data "$DATA" \
      --out "$OUT" \
      --M "$M" \
      --h "$H" \
      --sigma 0.30 \
      --seed 2027
  done
done
```

Or run:

```bash
$NEPTUNE_ROOT/launch/run_matrix.sh
```

if `run_matrix.sh` contains the matrix loop.

---

## 10. Checkpoints and resume

Each run creates its own directory, for example:

```text
artifacts/runs/h1_preview_2027/4f223a6dc0c3/
└── neptune_M16_h0.20_sg0.30_uniform_ord_s2027/
    ├── checkpoint_last.pt
    ├── config.yaml
    ├── provenance.json
    ├── environment.json
    ├── hardware.json
    ├── metrics.jsonl
    ├── metrics.parquet
    ├── peruser_val.parquet
    ├── sealed/
    └── summary.json
```

`checkpoint_last.pt` is used automatically when `resume=True`.

If a process or server stops during training, run the **same command again**. The runner will resume from the latest checkpoint.

If `summary.json` already exists and provenance matches, the completed run is reused instead of retrained.

Therefore it is safe to restart the same matrix script after an interruption:

```bash
$NEPTUNE_ROOT/launch/run_matrix.sh
```

Completed runs are reused; interrupted runs resume; remaining runs start normally.

---

## 11. Running safely over SSH

Use `tmux` so training is not tied to the SSH session.

Create a session:

```bash
tmux new -s neptune
```

Run the experiment or matrix inside it:

```bash
$NEPTUNE_ROOT/launch/run_matrix.sh
```

Detach:

```text
Ctrl+B, then D
```

Reconnect later:

```bash
tmux attach -t neptune
```

---

## 12. Parallel training across machines

Use the **same processed dataset** and **same implementation fingerprint** on every machine.

Example:

```text
Colab GPU:
  M=1, h=0.20

RTX 3090 server:
  M=16, h=0.20
```

Each machine writes a different run directory because the run key includes `M`, `h`, `sigma`, negative sampler, time mode, and seed.

If storage is not shared, copy the completed run directories back to one common `artifacts/runs/...` tree afterward.

Do not let two machines train the exact same run key into the same output directory at the same time.

---

## 13. Configuration rule of thumb

Use **sweep arguments** for experimental dimensions:

```text
M
h
sigma
seed
negative sampler
time mode
```

Use a **profile YAML** for engineering/runtime changes such as:

```text
n_users
epochs
batch_size
learning rate
warmup steps
checkpoint frequency
```

Do not edit the frozen source code merely to change a run configuration.

---

## 14. Minimal daily workflow

After the server has been set up once:

```bash
export NEPTUNE_ROOT=/home/user/NEPTUNE
export DATA=$NEPTUNE_ROOT/data/processed/h1_preview_2027/4f223a6dc0c3
export OUT=$NEPTUNE_ROOT/artifacts

source $NEPTUNE_ROOT/code/neptune/.venv/bin/activate
```

Then one run is simply:

```bash
python $NEPTUNE_ROOT/launch/run_one.py \
  --data $DATA \
  --out $OUT \
  --M 16 \
  --h 0.20 \
  --sigma 0.30 \
  --seed 2027
```

Or an entire matrix:

```bash
$NEPTUNE_ROOT/launch/run_matrix.sh
```

That is the intended operational workflow.

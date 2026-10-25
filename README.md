# kronecker-moe-gate

A small test harness for the ERA V5 Capstone G5 (Architecture) group. It trains a tiny GPT-2 with a Kronecker embedding and a 4-expert top-1 MoE on English+Hindi text, and checks that runs are repeatable across 1 GPU, CPU, 4-GPU DDP and 4-GPU DeepSpeed ZeRO-3. It is a determinism and plumbing check, not a model-quality testbed.

Status: test setup. The V5 architecture is not decided; the MoE is an assumption.

## Layout

| Path | What it is |
|---|---|
| `configs/config_v0.yaml` | Every setting with a default and a source comment (a V4 file and key, or "G5 gate value"). Size keys are TBD until TSAI (31 Oct). |
| `harness/train.py` | Trainer. Seeded, deterministic, writes `losses.json`, `eval.json`, `metrics.jsonl`, `manifest.json`. |
| `harness/check_config.py` | Reports config keys with no value or no source comment. |
| `data/make_mixed_sample.py` | Builds the en+hi text from the XNLI test split (train) and validation split (eval). |
| `deepspeed/zero3_4gpu.json` | ZeRO-3 config for the 4-GPU run. |
| `experiments/plan_v0.md` | Draft experiment plan: arms, metric, pass bar, GPU hours (proposals for review). |

## Setup

Needs Python 3.10 or newer (`kronecker-embeddings` requires it).

```
python -m pip install torch transformers kronecker-embeddings pyyaml pandas pyarrow datasets
python -m pip install deepspeed   # only for --deepspeed
python data/make_mixed_sample.py                    # data/mixed_en_hi_xnli.txt
python data/make_mixed_sample.py --split validation # data/mixed_en_hi_xnli_val.txt
```

Run everything from the repo root. The data files are not committed.

## Run

```
python harness/check_config.py configs/config_v0.yaml --allow-tbd ke2_shape_tbd size_tbd_1 size_tbd_2

# single process (CPU, or cuda:0 if visible); twice, then diff
python harness/train.py --config configs/config_v0.yaml --out runs/gate1
python harness/train.py --config configs/config_v0.yaml --out runs/gate2
diff runs/gate1/losses.json runs/gate2/losses.json && echo DETERMINISTIC

# 4-GPU DDP (set training.batch to a multiple of 4 in the config)
python -m torch.distributed.run --nproc_per_node=4 harness/train.py --config <cfg> --out runs/ddp1

# 4-GPU DeepSpeed ZeRO-3
python -m torch.distributed.run --nproc_per_node=4 harness/train.py --config <cfg> --out runs/ds --deepspeed
```

A missing Kronecker package, tokenizer or data file exits with code 4. `--allow_fallback` permits a debug-only fallback and the run is invalid. Exit code 3 means the kill rule fired (NaN, or after step 100 the mean of the last 20 steps above the mean of the first 20).

## Repeatability rules

- Same setup repeated (same hardware, config, launch): `losses.json` and `eval.json` byte-identical.
- Different setups (1 vs 4 GPU, GPU vs CPU, DDP vs DeepSpeed): `loss_ntp` within 1e-5 on every step.
- Gradient clipping is 1.0 on every path. DeepSpeed clips by default, so a path without clipping diverges from it.

## Training settings

Taken from V4 reference files: gradient clip 1.0, AdamW betas 0.9/0.95, eps 1e-10, weight decay 0.0, peak lr 1e-4 (V4 stage `3B_MOE_PRODUCTION`). Each is cited in the config comments.

## Results so far (toy scale, one machine)

Tiny model, 500 steps, batch 2, seed 1234. Last-20-step mean training loss, with held-out validation loss at the end:

| lr | last-20 mean | final val loss |
|---|---|---|
| 1e-5 | 10.09 | 9.93 |
| 1e-4 | 6.93 | 6.64 |
| 2e-4 | 5.45 | 5.14 |
| 1e-3 | 4.87 | 4.87 |

The 1e-3 run on seeds 1234, 1235, 1236 gives last-20 means of 4.87, 4.89, 4.91. A tiny model over 500 steps favours a high learning rate, so this is not a recommendation for V5.

DDP vs DeepSpeed over 500 steps (clipping on in both): maximum per-step difference 9.5e-07.

## Limits

- 2 layers, 128 hidden, 32-token sequences, 20 to 500 steps.
- Every expert runs on every token (keeps ZeRO-3 collectives in order), so MoE throughput here is not representative.
- No router load-balancing loss, checkpoint save/resume, MTP loss, or LR hold phase yet.
- At the gate's batch size each rank gets one row, so 4 GPUs are slower than 1.
- Config keys not read by the code yet: OPUS settings, always-on Indic fraction, checkpoint cadence, ZeRO stage.

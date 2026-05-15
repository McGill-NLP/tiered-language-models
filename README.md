# Tiered Language Models (TLMs)

Reference implementation for "Toward Open Weight Models Without Risks: Separating Public and Private Capabilities in LLMs".

A TLM is a single set of weights that supports two (or more) computation graphs over the same parameters. A compact secret key specifies a permutation over a small parameter subset (about 5% of total parameters): without the key the model runs in its public configuration, and with the key it runs in a keyed configuration that exposes additional capabilities. Public and keyed models share the same parameter values and differ only in how those values are arranged within the weight tensors.

## Install

```bash
pip install -e .
```

Requires Python 3.9 or newer, PyTorch 2.6 or newer, CUDA 12 or newer, and at least one NVIDIA H100 80GB (or equivalent). All paper experiments used 8x H100 with bf16 mixed precision and PyTorch FSDP.

For the AlpacaEval judging step (Section 4 / Figure 2 right), `alpaca-eval` is installed on first run by the eval pipeline; it is not declared as a hard dependency.

## Repository layout

```
.
├── src/tiered/
│   ├── model/                                            ── GPT-Neo backbone with apply_key / unapply_key methods
│   ├── permutation/                                      ── Key dataclass, swap plans, GPT-Neo and Qwen-style permutations
│   ├── data/                                             ── Dataset preparation
│   │   ├── prepare_fineweb.py                              · English pretraining corpus
│   │   ├── prepare_fineweb2_multilingual.py                · Spanish, German, Turkish private corpora
│   │   ├── prepare_alpaca.py                               · Stanford Alpaca instruction-tuning data
│   │   └── generate_synthetic_bios.py                      · 400-person memorization dataset
│   └── train/
│       ├── inference.py                                  ── Generation under any (key, configuration)  (Figs 16-18)
│       ├── pretrain/
│       │   ├── pretrain.py                                 · Non-tiered baseline                        (§C.3)
│       │   ├── tiered_pretrain.py                          · 2-tier asymmetric joint pretraining        (§3.2)
│       │   ├── tiered_pretrain_c2k.py                      · Periodic-keyed-pass variant                (§5, Fig 3 right)
│       │   └── cumulative_mult_tiered_pretrain.py          · Cumulative multi-tier pretraining          (§7, Fig 15)
│       └── finetune/
│           ├── private_finetune.py                         · Private finetune with KL anchor            (§3.2, Fig 2)
│           ├── private_finetune_memorization.py            · Memorization variant                       (Fig 8)
│           ├── baseline_finetune.py                        · Non-tiered finetune                        (§C.3, Fig 9 right)
│           ├── lora_private_finetune.py                    · 1% LoRA matched-performance baseline       (§C.5, Fig 11, Tab 2)
│           ├── multi_stage_private_finetune.py             · Sequential multi-tier finetune             (§7, Fig 5/6)
│           └── extraction_attack.py                        · Finetuning-based extraction attack         (§6, Fig 4 left)
│
├── scripts/
│   ├── keys/
│   │   ├── generate_key.py                                 · Random 5%-by-default key generator
│   │   └── benchmark_permutation_speed.py                  · Permutation latency benchmark              (Tab 3)
│   └── eval/
│       ├── partial_key_recovery_memorization_per_module.py · Partial-key access                         (§6, Fig 4 right)
│       ├── qwen_key_destruction_ablation.py                · Random permutation on Qwen-3-8B / MMLU     (§C.1, Fig 7 left)
│       ├── analyze_c1_keyed_magnitudes.py                  · Magnitude-ratio heatmaps                   (§C.4, Fig 10)
│       ├── attack_magnitude_ranking.py                     · Magnitude-based key recovery               (§C.4, Tab 1)
│       ├── llm_judge_c1_c2.py                              · Generate C1/C2 outputs + gpt-oss-120b judge
│       ├── export_llm_judge_outputs_to_alpaca_eval.py      · Format for AlpacaEval CLI
│       ├── log_llm_judge_checkpoint_to_wandb.py            · Push judge results to W&B
│       └── alpaca_eval_configs/
│           └── gpt_oss_120b_local.yaml                     · AlpacaEval annotator config
│
├── launchers/                                            ── Thin shell wrappers (25 files)
│   ├── data/                                               · Data preparation
│   ├── keys/                                               · Key generation, latency benchmark
│   ├── pretrain/                                           · 180M and 650M pretraining
│   ├── finetune/                                           · Private finetune variants
│   ├── eval/                                               · Adversarial and analysis evaluations
│   └── inference.sh                                        · Qualitative generation
│
├── configs/
│   ├── keys/180m/both/key_5pct.json                        · Bundled default 5% key (180M)
│   └── alpaca_eval/chatgpt/basic_prompt.txt                · Judge prompt referenced by the AlpacaEval YAML
│
├── pyproject.toml
├── README.md
└── LICENSE
```

## Compute requirements

Reproducing the headline numbers requires 8x NVIDIA H100 80GB. See Table 4 (pretraining) and Table 5 (finetuning) in the paper for batch sizes, learning rates, schedules, and total step or token counts per experiment. Pretraining the 180M TLM uses about 18B tokens; the 650M TLM uses about 65B tokens; private finetunes range from 2B to 6B tokens depending on dataset.

All training scripts log to Weights & Biases. Set `WANDB_API_KEY` (and optionally `WANDB_ENTITY`) before launching, or pass `--wandb_project <name>` when invoking the underlying script directly.

## Reproducing the paper

Each launcher is a thin wrapper with sensible defaults baked in. Override any parameter via env var, for example:

```bash
NGPUS=4 OUTPUT_DIR=./checkpoints/run1 SEED=1 bash launchers/pretrain/180m_tiered.sh
```

All commands below assume the repository root as the working directory.

### 1. Data preparation

```bash
bash launchers/data/prepare_fineweb.sh                # English pretraining corpus (sample-100BT, 100B tokens by default)
bash launchers/data/prepare_fineweb2_multilingual.sh  # Spanish, German, Turkish private corpora (5B tokens per language)
bash launchers/data/prepare_alpaca.sh                 # Stanford Alpaca (set DATA_PATH to alpaca_data.json)
bash launchers/data/generate_synbios.sh               # 400-person bios; writes tokenized/, tokenized_full/, bios_metadata.json
```

### 2. Key generation

A 5% key for the 180M architecture is bundled at `configs/keys/180m/both/key_5pct.json` and is used by every two-tier 180M experiment. To regenerate it (or generate the three disjoint keys required for the multi-tier setting in Section 7):

```bash
bash launchers/keys/generate_key.sh                                  # 180M, 5%, seed=42
SEED=43 OUTPUT_PATH=./key_5pct_2.json bash launchers/keys/generate_key.sh
SEED=44 OUTPUT_PATH=./key_5pct_3.json bash launchers/keys/generate_key.sh
```

For 650M keys, override architecture parameters:

```bash
NUM_LAYERS=16 NUM_HEADS=16 HIDDEN_SIZE=1344 MLP_DIM=10752 \
  OUTPUT_PATH=./key_5pct_650m.json bash launchers/keys/generate_key.sh
```

### 3. Pretraining

```bash
bash launchers/pretrain/180m_tiered.sh        # 180M TLM (Section 4, Table 4)
bash launchers/pretrain/180m_c2k.sh           # f sweep variant (Section 5, Figure 3 right). Set C2_EVERY_K to vary K.
bash launchers/pretrain/180m_baseline.sh      # Non-tiered baseline (Section C.3, Figures 3 left and 9 right)
bash launchers/pretrain/180m_cumulative.sh    # Cumulative multi-tier pretraining (Section 7, Figure 15). Requires three 5% keys.
bash launchers/pretrain/650m_tiered.sh        # 650M TLM (Section 4 Figure 2 right, Table 4). Requires a 650M-architecture key.
```

### 4. Finetuning

```bash
bash launchers/finetune/180m_spa.sh           # 180M Spanish private finetune (Section 4, Figure 2 left)
bash launchers/finetune/180m_synbios.sh       # 180M memorization on synthetic bios (Section 4, Figure 8)
bash launchers/finetune/180m_baseline_spa.sh  # Non-tiered baseline finetune for comparison (Section C.3, Figure 9 right)
bash launchers/finetune/180m_lora_spa.sh      # 1% LoRA matched-performance baseline (Section C.5, Figure 11, Table 2)
bash launchers/finetune/180m_multi_stage.sh   # 3-stage cumulative finetune over German -> Turkish -> Spanish (Section 7, Figure 5/6)
bash launchers/finetune/650m_alpaca.sh        # 650M instruction tuning on Alpaca (Section 4, Figure 2 right). 3 epochs by default.
```

### 5. Adversarial evaluation

```bash
bash launchers/eval/extraction_attack.sh         # Finetuning-based extraction attack (Section 6, Figure 4 left)
bash launchers/eval/partial_key_recovery.sh      # Partial-key access (Section 6, Figure 4 right). 100 random subsets per percentage by default.
bash launchers/eval/qwen_key_destruction.sh      # Post-hoc random permutation on Qwen-3-8B (Section C.1, Figure 7 left)
bash launchers/eval/analyze_magnitudes.sh        # Magnitude-ratio heatmaps (Section C.4, Figure 10)
bash launchers/eval/attack_magnitude_ranking.sh  # Magnitude-based key recovery (Section C.4, Table 1)
```

### 6. AlpacaEval (Figure 2 right)

Two phases. First, generate C1 and C2 responses on AlpacaEval prompts and judge them with gpt-oss-120b (loaded locally via HuggingFace transformers):

```bash
bash launchers/eval/llm_judge_c1_c2.sh
```

Second, optionally produce the canonical AlpacaEval scoring via the third-party `alpaca_eval` CLI. The CLI is installed on first run via pip if not already present:

```bash
bash launchers/eval/alpaca_eval_c1_c2.sh
```

The annotator YAML's `prompt_template: chatgpt/basic_prompt.txt` resolves against `configs/alpaca_eval/`. Run from the repo root, or set `ALPACAEVAL_PROMPTS_DIR=configs/alpaca_eval` (or symlink) so alpaca_eval finds the template.

### 7. Permutation latency benchmark (Table 3)

```bash
bash launchers/keys/benchmark_perm_speed.sh
```

### 8. Qualitative examples (Figures 16, 17, 18)

```bash
bash launchers/inference.sh
```

`inference.py` accepts multiple `--checkpoint` paths and one or more keys, evaluates the prompt under each (key, configuration) pair, and prints results side by side. Use the launcher for the simple case; invoke `python -m tiered.train.inference` directly for multi-checkpoint or multi-key sweeps.

## Notes

* Training scripts save checkpoints at `--save_interval` (default 1000 steps for pretraining, 1000 to 2000 for finetuning) plus a `final/` checkpoint at the end. Finetuning additionally saves `best/` based on validation loss.
* Resuming a run: pass the previous checkpoint directory via `--checkpoint` (pretraining) or `--resume_from` (private finetuning). Optimizer state, scheduler state, W&B run id, data epoch, and cumulative wall time are restored from `training_state.pt`.
* The 180M experiments in the paper used `configs/keys/180m/both/key_5pct.json` (bundled). The multi-tier (Section 7) and 650M experiments require additional keys generated as described in Section 2 above.
* The synthetic-biography dataset (Sections 4, 6, C.2) is regenerated deterministically by `launchers/data/generate_synbios.sh` with `SEED=42`.

## License

See `LICENSE`.

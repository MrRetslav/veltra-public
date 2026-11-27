# Model card: reference model files

**This repository contains no model weights.** The numbers in the README were measured with model files built from public checkpoints. Their
licences stay with their authors.

## `qwen3-30b-a3b` reference file (called `hf30` in the logs)
| | |
|---|---|
| Base model | Qwen3-30B-A3B (July 2025 instruct release) by the Qwen team, Apache-2.0 |
| Architecture | 48 layers, hidden size 2048, 32 heads (4 KV heads), 128 experts, 8 per token, expert size 768, vocabulary 151,936 |
| Expert weights | transcoded losslessly from Unsloth's `Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf`: gate/up Q2_K; down Q3_K in 38 layers, Q4_K in 10 |
| Attention | own export from the original weights: `q` (48 layers) and `o` (42 layers) 3-bit with GPTQ; `k`, `v` and `o` of 6 layers 4-bit with GPTQ |
| Embedding and output head | 8-bit |
| File size | 11.96 GB |
| Format | `.vlt` (`docs/INTERFACES.md`, `docs/qk_format.md`, `docs/i3_format.md`) |
| Tools | `python/export_qwen3_moe_stream.py`, `python/quant_i3.py`, `python/export_i3_attn.py`, `python/export_qwen3_moe_qk.py` |

The exact file is **not published**, and a one-command rebuild of it does not exist yet: the attention step needs calibration activations
(`python/calib_text_i3.py`) and several runs of the tools above. Numbers measured on another export can differ.

## Quality
160-task public set (`benchmarks/eval_public/`), greedy, thinking off. The set is small and checks arithmetic with traps, short Python functions
(run in a sandbox), JSON extraction and facts hidden in long prompts. It is **not** a general intelligence benchmark: use it to compare
settings of one model, not to rank models.

Scores per setting and engine: see the README and `benchmarks/results/`.

## Limits
- Quantised models make more mistakes than the original checkpoint. The original was not run on this task set.
- `--moe-k` and `--pareto-max` change the output of the model and cost quality, mostly on code and JSON.
- The model can be confidently wrong. Check anything that matters.
- Training data, languages and biases: see the Qwen3 model card of the original authors.

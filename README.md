# Compressed Agent Communication

Non-natural-language latent communication between LLM-based agents. Instead of exchanging full text or unbounded KV caches, agents communicate through **compressed latent representations** — slot-attention bottlenecks, deterministic bottlenecks, or VAE bottlenecks — with a frozen LLM backbone.

## Key Results

| Scenario | Method | Message Size | Accuracy | Comparison |
|----------|--------|-------------|----------|------------|
| GSM8K (4-agent reasoning) | SlotMAS (4×64, trained) | **512 B/agent** | 91% | = Baseline (91%), ~LatentMAS (95%) |
| ARC-Challenge | SlotMAS (4×d_model, random) | 40 KB/agent | **95%** | > Baseline (92%), > LatentMAS (93%) |
| Hidden Profile (must-comm) | Det Bottleneck dim=8 | **16 B** | 81.5% (comm: 64.5%) | vs Full Pool 10KB: 90.6% |
| GPQA-Diamond | LatentMAS (full KV) | ~MB | **26.8%** | >> Baseline (8.1%) |
| QASPER Long-Doc | Prefix-256 (text) | 1.35 KB | **54.0%** | > Full text 29.7KB (33.0%) |

> See [EXPERIMENTS.md](EXPERIMENTS.md) for complete results, analysis, and insights.

## Project Structure

```
compressed-agent-communication/
├── README.md               ← This file
├── EXPERIMENTS.md           ← Full experiment results & insights
├── src/
│   ├── LatentMAS/           ← Multi-agent reasoning (baseline/text/latent/slot)
│   │   ├── run.py           ← Main evaluation entry point
│   │   ├── train_compressor.py  ← Slot compressor training (v2, slot-aligned)
│   │   ├── methods/
│   │   │   ├── slot_mas.py  ← SlotMAS: slot-attention compressed communication
│   │   │   ├── latent_mas.py ← LatentMAS: full KV cache communication
│   │   │   ├── text_mas.py  ← TextMAS: natural language communication
│   │   │   └── baseline.py  ← Single-agent baseline
│   │   ├── models.py        ← Model wrapper (HuggingFace + vLLM)
│   │   ├── prompts.py       ← Agent prompt templates
│   │   ├── data.py          ← Dataset loaders
│   │   └── utils.py         ← Utilities
│   ├── latent_communication/     ← Synthetic controlled channel experiments
│   │   └── run_experiment.py
│   └── latent_communication_llm/ ← Hidden Profile + QASPER experiments
│       ├── run_hidden_profile.py
│       ├── run_qasper_long_context_eval.py
│       └── ...
├── results/                ← Experiment result files
│   ├── hidden_profile/
│   ├── qasper/
│   ├── multi_agent_reasoning/
│   └── synthetic/
└── doc/
    └── scripts/                 ← Env, reproduce, monitor, benchmark helpers
        ├── env_autodl.sh
        ├── start_up.sh
        ├── reproduce_autodl.sh
        ├── monitor_repro.sh
        ├── stop_repro.sh
        ├── run_benchmark.sh
        ├── run_llm_smoke.sh
        └── train_slot_compressor.sh
```

## Quick Start

### Requirements

```bash
pip install torch transformers datasets tqdm vllm
```

### 1. Run Baseline Comparison

```bash
# Compare baseline / latent_mas / slot_mas on GSM8K, ARC, GPQA, etc.
bash doc/scripts/run_benchmark.sh /path/to/Qwen3-14B 0,1,2
```

### 2. Train Slot Compressor

```bash
# Collect hidden states + train compressor (slot-aligned v2 pipeline)
bash doc/scripts/train_slot_compressor.sh /path/to/Qwen3-14B gsm8k 0,1,2
```

### 3. Evaluate Trained Compressor

```bash
python src/LatentMAS/run.py \
    --method slot_mas \
    --model_name /path/to/Qwen3-14B \
    --task gsm8k \
    --latent_steps 10 \
    --num_slots 4 \
    --slot_dim 64 \
    --think \
    --compressor_path results/trained_compressor/compressor_s4_d64.pt \
    --max_samples 100
```

## Methods

### SlotMAS (This Work)

Agents communicate through **slot-attention compressed representations**:

1. Each non-judger agent runs a forward pass + latent reasoning steps
2. The hidden sequence is compressed to `num_slots × slot_dim` via cross-attention
3. Compressed slots are transmitted (the bottleneck) and decompressed for the next agent
4. The judger receives all decoded slots as soft prefix and generates the answer

**Communication cost**: `num_slots × slot_dim × 2 bytes` (fp16)
- 4 slots × 64 dim = **512 bytes/agent** (vs ~MB for full KV)

### Training Pipeline (v2, Recommended)

The key insight: hidden states must be collected under **the same conditions as inference** (slot prefix chain), not under full KV chain (which creates a distribution mismatch).

1. **Slot-aligned collection**: Each agent receives decoded slots from previous agents (matching inference)
2. **Multi-objective loss**: Cross-attention step reconstruction + slot-level target + last-step MSE + pooled cosine + diversity
3. **Optional teacher**: Use a pre-trained compressor for more stable prefix chains during collection

## Citation

```
@misc{compressed-agent-communication,
  title={Compressed Latent Communication for Multi-Agent LLM Systems},
  year={2026}
}
```

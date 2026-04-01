# Qwen3 Latent Handoff

This directory contains a practical latent-communication benchmark built on top
of frozen `Qwen3-14B` hidden states.

The task is natural-language profile QA:

- The sender reads a full profile paragraph.
- The receiver only reads the question.
- Communication happens through a learned continuous latent message derived from
  the sender's hidden states.

The pipeline has four parts:

1. Generate profile/question text from structured records.
2. Extract frozen Qwen3 hidden states for documents and questions.
3. Train latent handoff models with three stages:
   - `stage1_high_band`
   - `stage2_purified`
   - `stage3_compressed`
4. Run a small direct-generation baseline on the same test split.

Run with the local conda C++ runtime visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python run_qwen_handoff.py \
  --output-dir results/full_run
```

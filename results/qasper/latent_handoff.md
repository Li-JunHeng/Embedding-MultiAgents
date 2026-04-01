# QASPER Long-Context Latent

## Setup
- Train/val/test sizes: 400/100/100
- Added distractor docs per example: 2
- Distractor chars per doc: 8000
- Sender mode: `query_select`
- Sender chunk config: words=64, stride=48, top_k=3

## Sender Stats
- Avg sender bytes on test: 1250.1
- Avg sender tokens on test: 254.8
- Avg full long-context bytes on test: 29689.1
- Avg full long-context tokens on test: 6083.9
- Avg selected chunks on test: 3.00
- Avg oracle chunk recall on test: 0.3247

## Accuracy
- Question-only scorer: 0.3400
- Long full-text generation: 0.3300
- Long query-select generation: 0.3900
- Long question-only generation: 0.3100
- qasper_high_band: 0.3100
- qasper_purified: 0.3200

## Main Table
| Method | Test Acc. | Message | Relative Bytes vs Long Full Text |
| --- | ---: | --- | ---: |
| Qwen Long Question Only | 0.310 | 0 B | 0.0% |
| Qwen Long Full-Text Handoff | 0.330 | 6083.9 tok / 29689.1 B | 100.0% |
| Qwen Long Query-Select Handoff | 0.390 | 254.8 tok / 1250.1 B | 4.2% |
| Long High Band | 0.310 | 1024 fp16 / 2048 B | 6.9% |
| Long Purified | 0.320 | 768 fp16 / 1536 B | 5.2% |
# QASPER Long-Context Eval

## Setup
- Base split sizes: 400/100/100
- Test examples evaluated: 100
- Added distractor documents per example: 2
- Distractor chars per document: 8000
- Prefix sender budget: 256 tokens
- Query-select sender: chunk_words=64, chunk_stride=48, top_k=3

## Message Stats
- Base full-doc size: 2839.8 tok / 13725.6 B
- Long full-doc size: 6083.9 tok / 29689.1 B
- Long prefix size: 256.0 tok / 1349.6 B
- Long query-select size: 254.8 tok / 1250.1 B

## Accuracy
- Full text: 0.3300
- Prefix-256: 0.5400
- Query-select: 0.3900
- Question only: 0.3100

## Main Table
| Method | Test Acc. | Message | Relative Bytes vs Long Full Text |
| --- | ---: | --- | ---: |
| Qwen Question Only | 0.310 | 0 B | 0.0% |
| Qwen Long Full-Text Handoff | 0.330 | 6083.9 tok / 29689.1 B | 100.0% |
| Qwen Long Prefix-256 Handoff | 0.540 | 256.0 tok / 1349.6 B | 4.5% |
| Qwen Long Query-Select Handoff | 0.390 | 254.8 tok / 1250.1 B | 4.2% |
# Latent Communication Experiment

## Task
- Factors: 8
- Semantic values per factor: 16
- Nuisance styles per factor: 8
- Query mix: 50% retrieval, 50% pairwise modular sum

## Results
### stage1_high_band
- Message size: 512 floats (1.00x smaller than stage 1)
- Clean answer accuracy: 1.0000
- Robust answer accuracy: 0.6833
- Avg KL rate: 0.0000
- State probe mean factor accuracy: 1.0000
- State probe exact match: 1.0000
- Style probe mean factor accuracy: 0.3520
- Best val accuracy: 1.0000 at epoch 7

### stage2_purified
- Message size: 384 floats (1.33x smaller than stage 1)
- Clean answer accuracy: 1.0000
- Robust answer accuracy: 0.6845
- Avg KL rate: 7.5699
- State probe mean factor accuracy: 1.0000
- State probe exact match: 1.0000
- Style probe mean factor accuracy: 0.1614
- Best val accuracy: 1.0000 at epoch 11

### stage3_compressed
- Message size: 128 floats (4.00x smaller than stage 1)
- Clean answer accuracy: 0.9995
- Robust answer accuracy: 0.6777
- Avg KL rate: 16.5784
- State probe mean factor accuracy: 1.0000
- State probe exact match: 1.0000
- Style probe mean factor accuracy: 0.1348
- Best val accuracy: 0.9995 at epoch 18

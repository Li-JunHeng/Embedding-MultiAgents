# Latent Communication Prototype

This directory contains a standalone sender/receiver communication benchmark for
testing whether continuous latent messages can replace natural-language
handoffs.

The benchmark is synthetic on purpose:

- The sender sees a hidden world state plus nuisance style variables.
- The receiver only sees a query.
- The receiver must answer by consuming the sender's latent message.

Three stages are implemented:

1. `stage1_high_band`: high-bandwidth continuous latent baseline.
2. `stage2_purified`: adds a variational bottleneck, slot dropout, state
   reconstruction, and a style adversary to purge nuisance information.
3. `stage3_compressed`: stronger bottleneck with fewer slots/dimensions and
   optional distillation from stage 2. The default config keeps factor-aligned
   slots and mainly compresses slot dimensionality, which turned out to be much
   lower loss than collapsing the number of slots.

The script reports:

- Task accuracy.
- Robust accuracy under channel corruption.
- Linear probe recovery of semantic state.
- Linear probe recovery of nuisance style.
- Message size and KL-rate proxies.

Run:

```bash
python run_experiment.py \
  --output-dir results/default_run
```

# Public Cache Schema

The reference methods communicate through NumPy `.npz` archives. Every row is
identified by an explicit `sample_ids` value. Consumers align rows by these IDs
and reject duplicate IDs, missing rows, label mismatches, or incompatible class
dimensions.

## Signal cache

A signal cache contains:

| Key | Shape | Meaning |
|---|---:|---|
| `iq` | `[N, 2, L]` | Real-valued in-phase and quadrature samples |
| `labels` | `[N]` | Integer class labels |
| `sample_ids` | `[N]` | Stable sample identifiers |
| `base_probabilities` | `[N, C]`, optional | Aligned primary probabilities |

The HCS, GAMC-inspired, pairwise, and long-spectral-lite entry points consume
this schema. Signal length `L` is not fixed by the interface.

## Single-predictor probability cache

A fitted predictor writes:

| Key | Shape | Meaning |
|---|---:|---|
| `probabilities` | `[N, C]` | Normalized class probabilities |
| `labels` | `[N]` | Integer class labels |
| `sample_ids` | `[N]` | Stable sample identifiers |
| `method` | scalar string | Method identifier |

## Candidate-pool cache

After merging the primary and auxiliary probability blocks, the candidate
cache contains:

| Key | Shape | Meaning |
|---|---:|---|
| `candidate_probabilities` | `[N, K, C]` | Primary plus `K-1` candidate distributions |
| `candidate_names` | `[K]` | Ordered candidate identifiers; primary is first |
| `labels` | `[N]` | Integer class labels |
| `sample_ids` | `[N]` | Stable sample identifiers |
| `meta_features` | `[N, F]`, optional | Additional observable evidence |

Train-OOF, validation, and optional held-out candidate caches must use the same
candidate order and class dimension. They contain different sample rows, so
row alignment is checked within each split rather than across splits.

## Merge example

```powershell
python merge_candidate_caches.py `
  --primary results/primary_train_oof.npz `
  --auxiliary iqcc=results/iqcc_train_oof.npz `
  --auxiliary hcs=results/hcs/hcs_train_oof.npz `
  --auxiliary pairwise=results/pairwise/pairwise_train_oof.npz `
  --auxiliary gamc=results/gamc/gamc_train_oof.npz `
  --output results/train_candidates.npz
```

Repeat the merge with aligned validation files and, after the policy has been
frozen, with held-out inference files.

# Dataset Requirements

No dataset is distributed with this repository. Users are responsible for
obtaining each dataset from its provider and complying with its terms.

## RML2016.10A

Expected local path:

```text
raw_data/RML2016.10a_dict.pkl
```

The public loaders expect the conventional Python dictionary keyed by
`(modulation, SNR)`, with two-channel I/Q arrays of length 128. The release does
not provide or fetch this pickle automatically.

## RML2016.10B

Expected local path:

```text
data/RML2016.10b.dat
```

The public 10B utilities also support an optional locally extracted helper
layout under `data/RML2016.10BGAMC/`. Neither form is redistributed.

## HisarMod2019.1

The paper evaluation uses the official 520,000/260,000 storage split and
length-1024 I/Q records. This selective public package provides aggregate audit
tables and protocol descriptions, but not the dataset, derived arrays, trained
models, or the full HisarMod experiment implementation.

The five robustness partitions reported by the project are **equal-sized test
storage blocks**, not authoritative physical-channel labels. They must not be
renamed as Rayleigh, Rician, Nakagami, or other channel families without
provider-supplied per-sample metadata.

## Split and alignment rule

All probability caches used together must share the same ordered sample indices,
class mapping, SNR mapping, and split seed. The evaluation scripts perform shape
and index checks where the source archive exposes indices. Never realign caches
by truncation, sorting probabilities, or using test labels.


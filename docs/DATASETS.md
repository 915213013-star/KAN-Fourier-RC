# Datasets

The repository does not redistribute RadioML2016.10A or RadioML2016.10B.
Obtain each dataset from an authorized source and comply with its terms.

## RML2016.10A

Expected default path:

```text
raw_data/RML2016.10a_dict.pkl
```

The loader expects the conventional Python dictionary keyed by
`(modulation, SNR)`, with arrays shaped as I/Q sequences of length 128.

## RML2016.10B

Expected portable paths:

```text
data/RML2016.10b.dat
data/RML2016.10BGAMC/
```

The second path is optional when the serialized `RML2016.10b.dat` file is
already available. The 10B compatibility loader searches the release-local
`dataprocessnew4.py` and the optional extracted helper directory.

## Generated artifacts

Feature caches belong in `feature_cache/`, model checkpoints in
`checkpoints/`, and probability/result archives in `results/`. These files are
ignored by Git because they are generated, potentially large, and may inherit
dataset redistribution restrictions.


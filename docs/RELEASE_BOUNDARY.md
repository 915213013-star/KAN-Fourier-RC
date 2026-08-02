# Public Scientific Interface

KAN-Fourier-RC is released as a focused academic reference implementation. The
public interface exposes the model-learning data flow, OOF protocol, candidate
cache contract, retain-or-correct formulation, aggregate statistical evidence,
complexity convention, robustness specification, and release tests needed to
inspect the paper's central claims.

Third-party datasets remain governed by their original licenses. Locally
generated checkpoints, feature caches, sample indices, and prediction archives
reside in ignored runtime directories and are identified by configuration and
artifact hashes when used for a frozen audit.

## Claim scope

The package does not claim that isolated ERU universally dominates alternative
OOF objectives, that HisarMod storage partitions are physical channels, or that
fresh training is bitwise identical across hardware. The reported contribution
is the complete primary-preserving residual-evidence and validation-frozen
action-policy design.

## Change policy

Protocol behavior, cache identifiers, and audit-table schemas are versioned.
New runs carry their own configuration fingerprints and output hashes; a cache
with incompatible protocol metadata is not silently reused.

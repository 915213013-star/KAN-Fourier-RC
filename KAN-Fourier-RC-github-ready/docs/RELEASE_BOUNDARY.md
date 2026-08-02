# Release Boundary and Intellectual Property

This repository exposes enough structure to inspect the paper's core data flow
and to audit aggregate claims while retaining a bounded set of research assets.

## Public scientific interface

The public interface includes:

- model and OOF protocol definitions;
- the retain-or-correct decision formulation;
- a generic aligned-candidate cache contract and reference routing path;
- aggregate rescue/harm and paired-significance evidence;
- protocol tests and provenance hashes.

## Retained implementation assets

The authors retain checkpoints, full prediction matrices, feature caches,
sample indices, paper-specific non-neural candidate implementations, private
experiment orchestration, and exhaustive search traces.
These materials can reveal dataset derivatives, unpublished variants, and
costly engineering decisions. They are not necessary for reading the source
protocol or checking the aggregate arithmetic.

## Claims this package does not make

- It is not a complete copy of the research workstation.
- It is not an unrestricted open-source release.
- It does not redistribute third-party datasets.
- It does not guarantee bitwise-identical retraining across hardware.
- It does not claim that isolated ERU universally dominates other OOF
  meta-objectives.
- It does not claim authoritative physical-channel labels for HisarMod storage
  partitions.

## Change policy

Protocol-hardening changes are recorded in `CHANGELOG.md`. A corrected public
trainer must never silently label newly generated caches as an older protocol.
Published aggregate paper records remain immutable provenance records; new runs
must carry their own protocol and configuration fingerprints.

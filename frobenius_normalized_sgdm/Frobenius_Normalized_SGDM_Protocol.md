# Frobenius-Normalized SGDM Study Protocol

## Frozen design

Task 3.4 freezes the first Frobenius-normalized SGDM study before any broad
Variant 3 result is inspected. The machine-readable source of truth is
`frobenius_normalized_sgdm/task_3_4_candidate_design.json`; the materialized
manifest is `experiment_manifests/task_3_4_frobenius_exploratory.json`.

The exploratory study is exactly the Cartesian product

```text
frobenius_learning_rate in {0.001, 0.003, 0.01, 0.03}
matrix_momentum          in {0.90, 0.95, 0.99}
seed                     = 1337
```

for 12 configurations. `frobenius_epsilon = 1e-12` and
`frobenius_shape_factor = 1.0` are fixed, not tuned. The normalization rule id
is `frobsqrtr-eps1em12-v1`, meaning

```text
sqrt(min(d_out, d_in)) * M / (||M||_F + 1e-12).
```

No bounded calibration was used. Every eligible GPT-2 124M projection matrix
has `min(d_out, d_in) = 768`, so the nominal normalized momentum norm is
`sqrt(768) = 27.712812921102035`. The four base LRs therefore search nominal
full-scale momentum-derived step norms of approximately `0.0277`, `0.0831`,
`0.2771`, and `0.8314`, before multiplication by the common schedule scale.
This fresh normalized-update grid is distinct from the raw Global SGDM grid.

Candidate order is LR-major and momentum-minor, in the exact order written
above. Run names encode optimizer, GPT-2 124M, Frobenius LR, momentum, matrix
weight decay, seed, and the normalization rule id. Re-materializing the design
must produce byte-identical JSON.

## Matched controls and budget

The Variant 3 config inherits `config/task_1_6_baseline.py`. The candidate
manifest copies the approved Variant 1 environment, dataset, and resource
locks. The design enumerates machine-checkable controls covering:

- GPT-2 124M architecture and seed-controlled initialization;
- OpenWebText, GPT-2 tokenizer, context length 1024, and matched data order;
- micro-batch 12, global accumulation 40, world size 2, and 491,520 tokens per
  attempted optimizer update;
- bfloat16, compilation, gradient clipping at 1.0, warmup 100, and the common
  cosine schedule shape through step 999;
- matrix decoupled weight decay 0.1 and the unchanged auxiliary AdamW settings;
- evaluations at steps 0, 333, 666, and 999 using 200 evaluation batches;
- the locked environment, dataset fingerprints, and RTX PRO 6000 Blackwell
  hardware class.

The selection checkpoint is step 999 at 491,028,480 processed tokens. A run
may attempt at most 1,000 updates (`max_iters = 999`) and process at most
491,520,000 tokens. Candidate ranking uses the unrounded step-999 validation
loss in ascending order. An exact tie is broken first by lower
`frobenius_learning_rate`, then by lower momentum.

Diagnostics remain disabled in broad study runs, so diagnostic overhead cannot
affect the comparison. A separate representative smoke in Task 3.5 must enable
and exercise diagnostics before the broad manifest can be authorized.

## Comparator and code-audit locks

The candidate design locks the selected Global SGDM and Static Per-Matrix SGDM
run ids, run names, exact candidate overrides, selection values, selection
records, recorded Git commits, source-manifest hashes, and selection-report
hashes. It additionally locks Static Per-Matrix SGDM mapping id
`2633929b6baf` and full fingerprint
`2633929b6bafc6556341f36a0c5318003a647fe2795298f030b9e26330c6b600`.
Manifest materialization fails if these artifacts or winner fields change.
The current semantic review is recorded in
`frobenius_normalized_sgdm/task_3_4_shared_code_audit.json` and its hash is
locked by the candidate design.

Before any Variant 3 launch, record a clean Variant 3 Git commit and audit the
shared paths listed in the candidate design from each comparator's recorded
commit through that commit. If a change affects a matched control or a
measurement, rerun the affected comparator or obtain explicit approval for a
documented limitation. Path changes alone do not waive this semantic audit.

## Outcome, resume, and confirmation policy

Every candidate retains an outcome: completed, failed, divergent, interrupted,
resumed, or numerically unstable. No candidate may be silently replaced, and
the grid may not change after results are viewed without a separately approved
follow-up protocol. Numerical events, clipping, epsilon-dominated events,
timing, and memory are reported separately from validation loss.

Because the current checkpoint path does not restore RNG state, a resumed run
is not eligible for matched data-order selection. Only completed uninterrupted
runs can win this study. A missing eligible result is reported; it is not
silently rerun.

After deterministic selection, the shared result utility will materialize an
unauthorized confirmation manifest for only seeds 2027 and 4099, preserving
the winner's LR, momentum, epsilon, shape factor, and rule identity. Together
with the valid seed-1337 winner, this gives three Variant 3 seeds.

A final H3 claim additionally requires three successful matched seeds for each
of the locked Global SGDM and Static Per-Matrix SGDM winners. Until all three
families satisfy that gate, one-seed observations may be reported but no final
three-family H3 conclusion is permitted.

## Authorization gates

The Task 3.4 exploratory manifest is deliberately unauthorized and its
materializer has no authorization option. Task 3.5 requires, in order:

1. approval of Tasks 3.1 through 3.4 and explicit smoke-compute authorization;
2. passing focused tests and one representative end-to-end smoke covering
   forward/backward, accumulation, clipping, both optimizer children,
   evaluation, diagnostics, checkpoint save, and deterministic next update;
3. completion of the comparator shared-code audit and recording the clean
   launch commit;
4. explicit authorization of the fixed 12-run compute budget;
5. separate approval before launching the generated confirmation manifest.

Task 3.4 itself launches no compute.

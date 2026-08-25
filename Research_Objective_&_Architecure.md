# Muon vs. SGDM Experiment Guide

## Project objective

This project investigates why Muon can outperform stochastic gradient descent with momentum (SGDM) when training Transformer language models.

The central question is:

> Can carefully scaled SGDM match Muon without changing the internal geometry of its matrix updates, or is Muon's singular-value transformation essential?

The purpose is not merely to rank optimizers. The experiments must causally separate two mechanisms:

1. **Per-matrix update magnitude:** different parameter matrices receive differently sized updates.
2. **Within-matrix update geometry:** Muon changes the relative strengths of singular directions inside each update matrix.

## Mathematical background

For an eligible matrix parameter `W_l`, let `G_l,t` be its stochastic gradient and `M_l,t` its momentum buffer:

```text
M_l,t = beta * M_l,t-1 + (1 - beta) * G_l,t
```

Ordinary SGDM applies:

```text
W_l,t+1 = W_l,t - eta * M_l,t
```

If the singular value decomposition of the momentum matrix is

```text
M_l,t = U_l,t * Sigma_l,t * V_l,t^T,
```

then Muon approximately replaces it with its polar factor:

```text
P_l,t = U_l,t * V_l,t^T.
```

A simplified Muon update is

```text
W_l,t+1 = W_l,t - eta_muon * s_l * P_l,t,
```

where `s_l` is any matrix-shape scaling used by the selected reference implementation.

Muon therefore changes both the total update magnitude and the singular-value structure within the matrix.

## Experimental principle

Change only one meaningful optimizer component at a time. Keep all other conditions matched:

- model architecture and initialization;
- dataset, tokenizer, sequence length, and data ordering;
- batch size, gradient accumulation, and total token budget;
- numerical precision and gradient clipping;
- learning-rate schedule and warmup;
- weight decay and other regularization;
- parameter groups and auxiliary optimizers;
- evaluation frequency;
- random seeds;
- hyperparameter-tuning budget.

Do not compare a poorly tuned SGDM baseline with a carefully tuned Muon baseline.

## Optimizer variants

### 1. Tuned global SGDM

All eligible matrices share one learning rate:

```text
Delta W_l,t = -eta * M_l,t
```

Properties:

- one global learning rate;
- no per-matrix scaling;
- dynamic momentum;
- preserves the original SGDM matrix direction;
- does not flatten singular values.

**Purpose:** establish how much performance careful global SGDM tuning alone can obtain.

Tune at least the learning rate and momentum. Tune weight decay and schedule parameters if they are not fixed by the common protocol.

### 2. Static per-matrix SGDM

Give each eligible matrix a fixed multiplier `a_l`:

```text
Delta W_l,t = -eta * a_l * M_l,t
eta_l = eta * a_l
```

The multipliers differ across matrices but remain fixed throughout training.

Prefer one multiplier per eligible weight matrix, or a clearly documented matrix-type grouping, rather than one rate for an entire Transformer block. A Transformer block contains multiple matrices, whereas Muon operates on individual eligible matrices.

Possible matrix groups include:

- attention QKV projection;
- attention output projection;
- MLP input projection;
- MLP output projection.

**Purpose:** determine whether persistent differences between matrix learning rates improve on the best global SGDM configuration.

Always report:

```text
best tuned global SGDM vs. best static per-matrix SGDM
```

This comparison is required before claiming that non-uniform rates are useful.

### 3. Frobenius-normalized SGDM

Normalize each momentum matrix while preserving its direction:

```text
M_hat_l,t = M_l,t / (||M_l,t||_F + epsilon)
Delta W_l,t = -eta * c_l * M_hat_l,t
```

A simple target scale is

```text
c_l = sqrt(r_l),
r_l = min(d_out_l, d_in_l),
```

which gives

```text
Delta W_l,t = -eta * sqrt(r_l) * M_l,t / (||M_l,t||_F + epsilon).
```

This is motivated by the exact polar-factor identity

```text
||U * V^T||_F = sqrt(r_l)
```

for a matrix with `r_l` retained singular directions.

Only add further shape scaling if it matches the chosen reference Muon implementation, and document it explicitly.

Properties:

- dynamic per-matrix scaling;
- does not need to compute Muon;
- preserves the SGDM direction;
- does not flatten singular values.

**Purpose:** test whether a cheap, independent matrix-normalization rule can reproduce Muon's magnitude-control benefit.

### 4. Muon-scaled SGDM using Muon as a norm oracle

For each eligible matrix:

1. Compute the SGDM momentum `M_l,t`.
2. Compute the Muon-transformed matrix `P_l,t`.
3. Compute the norm of the actual candidate Muon step.
4. Rescale the SGDM direction to have the same Frobenius norm.

If

```text
Delta W_muon_l,t = -eta_muon * s_l * P_l,t,
```

then use

```text
Delta W_oracle_l,t =
    -eta_muon * s_l
    * ||P_l,t||_F
    / (||M_l,t||_F + epsilon)
    * M_l,t.
```

This should satisfy

```text
||Delta W_oracle_l,t||_F ~= ||Delta W_muon_l,t||_F.
```

Properties:

- uses Muon only to obtain the target update norm;
- preserves SGDM's original matrix direction;
- discards Muon's transformed direction;
- changes only the overall scale of each matrix update.

**Purpose:** diagnose how much of Muon's performance is recovered when SGDM receives exactly Muon's per-matrix update magnitudes.

This is a scientific ablation, not necessarily a cheaper practical optimizer, because it may still compute the Newton-Schulz transformation before discarding its direction.

### 5. Full Muon

Use the unmodified reference implementation. For eligible matrices, its update is conceptually

```text
Delta W_muon_l,t = -eta_muon * s_l * P_l,t,
```

where `P_l,t` is the Newton-Schulz approximation to `U_l,t * V_l,t^T`.

Document exactly:

- momentum convention;
- whether Nesterov momentum is used;
- Newton-Schulz iteration count and coefficients;
- input normalization to Newton-Schulz;
- matrix-shape scaling;
- weight-decay implementation;
- eligible parameter groups;
- auxiliary optimizer for ineligible parameters.

**Purpose:** provide the reference containing both per-matrix magnitude control and within-matrix singular-value transformation.

## Mechanism summary

| Optimizer | Scale | Scale behavior | Preserves SGDM direction | Flattens singular values |
|---|---|---|---|---|
| Tuned global SGDM | Global | Static base rate | Yes | No |
| Static per-matrix SGDM | Per matrix | Static | Yes | No |
| Frobenius-normalized SGDM | Per matrix | Dynamic | Yes | No |
| Muon-scaled SGDM | Per matrix | Dynamic, obtained from Muon | Yes | No |
| Full Muon | Per matrix | Dynamic and geometry-dependent | No | Yes |

## Parameter grouping requirements

Use the same parameter partition in every experiment.

First identify all parameters eligible for Muon. These are normally two-dimensional hidden-layer weight matrices, such as attention and MLP projection matrices.

Parameters often excluded from Muon include:

- biases;
- normalization scales;
- embeddings;
- output head;
- other one-dimensional parameters.

All ineligible parameters must use the same auxiliary optimizer with identical settings in every experimental condition. For example, if AdamW handles embeddings and biases in the Muon run, it must handle those same named parameters in every SGDM variant.

Record explicitly:

- every named parameter and its group;
- the optimizer responsible for each group;
- treatment of tied input/output embeddings;
- weight-decay behavior for every group.

## Experimental phases

### Phase 0: Audit and reproduce the baselines

Before implementing new variants:

1. Inspect the existing training repository.
2. Locate the current SGDM, Muon, and auxiliary AdamW implementations.
3. Identify the exact parameter-group construction.
4. Reproduce tuned SGDM, full Muon, and relevant previous results.
5. Verify the fixed training configuration and parameter groups.

Do not proceed to causal comparisons until the baselines reproduce approximately and the update equations are understood.

### Phase 1: Unit-test the update mechanics

Use synthetic matrices or a tiny model.

For SGDM, verify

```text
Delta W = -eta * M.
```

For Frobenius-normalized SGDM, verify that the actual update norm matches the prescribed target.

For Muon-scaled SGDM, verify

```text
||Delta W_oracle||_F ~= ||Delta W_muon||_F.
```

Also verify that norm-matched SGDM preserves the SGDM direction. Define the Frobenius cosine similarity as

```text
cos_F(A, B) = <A, B>_F / (||A||_F * ||B||_F).
```

The following should hold, modulo the negative update sign:

```text
cos_F(Delta W_oracle, M) ~= -1.
```

Log

```text
cos_F(Delta W_muon, Delta W_sgdm)
```

to measure how strongly Muon's direction differs from SGDM.

### Phase 2: Tune the global baselines

Tune SGDM and Muon separately using comparable search budgets.

At minimum, consider:

- learning rate;
- momentum;
- weight decay, unless fixed;
- schedule and warmup parameters, unless fixed.

Select configurations using a predefined validation criterion. Do not reuse an SGDM learning rate for a fundamentally different normalized update rule.

### Phase 3: Compare all optimizer families

Run all five variants using:

- the same initialization seed;
- the same data order for each seed;
- the same token budget;
- the same evaluation checkpoints;
- the same eligible and auxiliary parameter groups.

One seed may be used for broad exploratory tuning. Use at least three seeds for the final promising configurations and any claimed differences.

### Phase 4: Test whether findings transfer

If compute permits, repeat the final comparison on:

- at least two model widths or depths;
- at least two batch sizes.

This tests whether conclusions are specific to a single small nanoGPT configuration.

## Required measurements

### Primary metric

Use validation loss after a fixed number of training tokens.

Do not compare only after a fixed number of iterations if effective batch sizes differ.

### Secondary metrics

Record:

- training loss;
- validation loss;
- processed tokens;
- wall-clock time;
- mean optimizer-step time;
- peak GPU memory;
- divergence or numerical instability;
- gradient-clipping frequency;
- sensitivity to learning rate and other key hyperparameters;
- mean and standard deviation across seeds.

### Per-matrix diagnostics

At selected checkpoints, log for every eligible matrix:

```text
||W_l,t||_F
||G_l,t||_F
||M_l,t||_F
||Delta W_l,t||_F
||Delta W_l,t||_F / (||W_l,t||_F + epsilon)
||Delta W_l,t||_spectral
effective per-matrix learning-rate multiplier
```

Also log

```text
cos_F(Delta W_sgdm, Delta W_muon).
```

For selected matrices and checkpoints, log the singular values of:

- the momentum matrix;
- the normalized SGDM update;
- the Muon update.

These diagnostics reveal whether methods obtain similar performance despite different within-matrix geometry.

Avoid logging full matrices or full singular spectra at every step. Use selected checkpoints to control storage and runtime overhead.

## Hypotheses and decision rules

### H1: Global tuning explains most of the difference

If tuned global SGDM matches static per-matrix SGDM and approaches Muon, earlier gaps were mainly caused by an under-tuned SGDM baseline.

### H2: Static matrix heterogeneity matters

If static per-matrix SGDM consistently beats the best global SGDM, persistent differences between matrix step sizes provide value.

### H3: Dynamic normalization matters

If Frobenius-normalized SGDM beats static per-matrix SGDM, update magnitudes benefit from adapting during training.

### H4: Muon's target-magnitude rule matters

If Muon-scaled SGDM beats generic Frobenius-normalized SGDM, Muon's particular target magnitude contains useful information beyond generic normalization.

### H5: Within-matrix geometry matters

If full Muon consistently beats Muon-scaled SGDM despite matched per-matrix update norms, Muon's singular-value transformation supplies an additional advantage.

Support this conclusion only after confirming:

- actual update norms are matched;
- parameter groups and auxiliary optimizers are identical;
- tuning budgets are comparable;
- results are consistent across seeds.

## Interpretation constraints

Do not claim that non-uniform step sizes are essential merely because one non-uniform configuration works well. Compare the best tuned uniform configuration against the best non-uniform configuration.

Do not claim that Muon's direction is irrelevant merely because norm-matched SGDM closes most of the loss gap. A remaining gap may arise from:

- Muon's singular-value transformation;
- momentum or Nesterov differences;
- regularization differences;
- parameter-group differences;
- schedules or warmup;
- stochastic variation.

Do not identify the best optimizer using training loss alone. Lower training loss can coexist with worse validation loss.

Do not describe a four-value grid as a search over all possible learning rates. It is exhaustive only over the specified discrete grid.

## Reproducibility requirements

Every run must save:

- complete configuration;
- Git commit hash;
- random seed;
- model parameter count and architecture settings;
- parameter-group listing;
- optimizer hyperparameters;
- dataset and tokenizer identifiers;
- sequence length, batch size, and gradient accumulation;
- total tokens and optimizer steps;
- precision settings;
- hardware information;
- train and validation metrics;
- timing and memory statistics;
- diagnostic update logs.

Use run names that encode at least:

```text
optimizer_model_lr_momentum_weight_decay_seed_scaling_rule
```

## Instructions for the coding agent

Before modifying code:

1. Inspect the repository structure and any local agent instructions.
2. Locate the training entry point, model definition, optimizer construction, configuration system, logging, and evaluation code.
3. Document the existing SGDM and Muon equations as implemented, not merely as expected from a paper.
4. Produce a complete parameter-group audit.
5. Identify all currently unmatched settings between optimizers.
6. Propose the smallest phased implementation that supports the five variants.
7. Add update-level unit tests before launching expensive runs.
8. Keep optimizer variants configurable from the same training entry point.
9. Preserve existing behavior by default.
10. Do not launch large sweeps until one short smoke test succeeds for every optimizer.

When reporting results, clearly separate:

- observed measurements;
- inferred explanations;
- unsupported or untested hypotheses.

## Desired scientific outcome

The experiments should answer two questions:

> How much of Muon's advantage comes from controlling the magnitude of each parameter-matrix update?

> After per-matrix update norms are matched, does Muon's within-matrix singular-value transformation still provide a consistent advantage?

The intended contribution is a causal decomposition of optimizer mechanisms, not only a benchmark ranking.

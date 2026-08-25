# Development Rules

These rules apply to all nanoGPT optimizer experiment work.

## 1. Approval and Git

- Do not commit, push, create a pull request, or modify a remote branch without explicit user approval.
- Local implementation and verification do not grant permission to commit.

## 2. Scope and minimal changes

- Make only the smallest changes required for the currently approved task.
- Preserve existing behavior by default.
- Avoid unrelated refactoring, formatting, dependency upgrades, or architectural changes.
- If the task requires a significant refactor, new dependency, changed experimental protocol, or other scope expansion, stop and request approval before proceeding.
- If the repository does not support a minimal solution, explain the limitation and propose the smallest viable alternatives before editing.

## 3. Sequential variants and tasks

Implement the optimizer variants in this order:

1. Tuned global SGDM.
2. Static per-matrix SGDM.
3. Frobenius-normalized SGDM.
4. Muon-scaled SGDM using Muon as a norm oracle.
5. Full Muon reference integration or verification.

Each variant is divided into separate tasks. Its task breakdown is stored inside the corresponding variant folder:

- `tuned_global_sgdm/Tuned_Global_SGDM_Tasks.md`;
- `static_per_matrix_sgdm/Static_Per_Matrix_SGDM_Tasks.md`;
- `frobenius_normalized_sgdm/Frobenius_Normalized_SGDM_Tasks.md`;
- `muon_scaled_sgdm/Muon_Scaled_SGDM_Tasks.md`;
- `full_muon/Full_Muon_Tasks.md`.

Work on exactly one approved task at a time. After completing that task:

1. Review the complete diff and confirm that it is limited to the task.
2. Verify the implementation against `Research_Objective_&_Architecure.md`.
3. Run appropriate unit tests, smoke tests, and existing checks.
4. Check for regressions, numerical issues, and unnecessary performance overhead.
5. Summarize:
   - what changed and which files changed;
   - how the implementation works;
   - which checks were run and their results;
   - any limitations or unresolved concerns.
6. Report: **Done and ready for approval.**
7. Stop and wait for explicit user approval.

Approval applies only to the completed task unless the user explicitly states otherwise. Do not start another task in the same variant, begin a later variant, implement future work in advance, or commit the completed task while awaiting approval. Complete every approved task in the current variant before moving to the next variant.

## 4. Prefer extension over modification

- Prefer a small function, class, module, wrapper, or configuration option that integrates with existing code over rewriting existing logic.
- Reuse existing training and optimizer pathways whenever practical.
- Modify existing code only where a small integration change is necessary.
- Do not duplicate large sections of the training loop to support a new variant.

## 5. Module and utility placement

- Put modules, functions, classes, or utilities that are genuinely reusable across two or more optimizer variants in `shared_utils/`.
- Put modules, functions, classes, or utilities specific to one optimizer variant in that variant's dedicated `utils/` folder:
  - tuned global SGDM: `tuned_global_sgdm/utils/`;
  - static per-matrix SGDM: `static_per_matrix_sgdm/utils/`;
  - Frobenius-normalized SGDM: `frobenius_normalized_sgdm/utils/`;
  - Muon-scaled SGDM: `muon_scaled_sgdm/utils/`;
  - full Muon: `full_muon/utils/`.
- Do not place variant-specific behavior in `shared_utils/` merely because it may be useful later.
- Do not duplicate shared logic between variant utility folders. If code becomes genuinely shared, move the smallest reusable portion to `shared_utils/` as part of an explicitly approved task.
- Do not import one variant's private utilities from another variant's `utils/` folder. Promote the common portion to `shared_utils/` instead.
- Keep integration changes in existing entry-point or model files minimal; substantive new optimizer logic belongs in `shared_utils/` or the appropriate variant's `utils/` folder.

## 6. Code quality

- Write readable, efficient, and maintainable code that follows the repository's existing style.
- Use clear names that reflect the optimizer equations and experimental intent.
- Keep functions focused and avoid unnecessary abstraction.
- Add concise comments where the mathematical purpose is not evident from the code.
- Avoid unnecessary runtime, memory, logging, or storage overhead.
- Add or update tests for new optimizer behavior where practical.

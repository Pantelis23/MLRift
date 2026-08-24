---
name: llm-regression-check
description: Use after touching codegen.mlr, ir.mlr, or any amdgpu emitter to confirm LLM inference byte-output is unchanged. The op72/ir.mlr edits made for GPU training MUST pass this before merge.
---

amdgpu-path compiler edits can silently corrupt LLM inference. Confirm byte-exactness:

1. Rebuild a known model gen: qwen3-0.6B and speck4. Current recorded emitted-`.co` md5s (`--emit-amdgpu-qwen3-megakernel-v2` / `--emit-amdgpu-qwen3-megakernel-speck4-v2`, compile-only, no GPU) are **a49219eb6cc5c1e11041086798048996** (qwen3) / **20dd24362e3dbb7290391c19d3584265** (speck4), re-confirmed on hardware 2026-08-24 at compiler HEAD 0f9a37d — confirmation recorded in commit 5f4533a. (The prior `ef399e4b…`/`6c0bc31a…` values were stale: `git log --all -S` shows no commit ever referenced them.) On a re-run, re-stamp the date here with `date +%F`. Use --target=amdgpu-native; ALWAYS hipkfd_teardown.
2. Separately, run a fixed-seed generation and md5 the token output — this is a distinct artifact from the `.co` md5s above, never conflate them. Diff vs the recorded token-output reference: qwen3 20-token md5 **ec879c18a4791f9888b5421fdc49f657**, observed 2026-08-24 at HEAD 016e14c against the in-tree PyTorch greedy reference. Identical = pass; any diff = regression — git bisect the codegen/ir change.
3. Spot-check Llama-1B / Mistral-7B if dims differ from qwen3 (HEAD_DIM 64 vs 128, FF sizes) — past bugs were dim-specific.
4. Report: which models, md5 match/diff, pass/fail. Fail = do not merge; revert the emitter change.
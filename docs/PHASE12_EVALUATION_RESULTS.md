# Phase 12 PR-summary evaluation incident

The loopback server returned 503 for read-only evaluation. An isolated
FastAPI reproduction returned 200 for preview and then raised
`RuntimeError: There is no Stream(gpu, 0) in current thread.` on the first
base-model generation. Preview initialized MLX-LM on the dedicated inference
worker. The evaluator called it from a FastAPI request thread. The error
occurred before loading the candidate for evaluation.

Evaluation now shares the inference worker, evicts the resident model before
evaluation, and serializes with ordinary inference. Failures log a redacted
traceback. The evaluation split, decoding, scoring and qualification gates
remain unchanged. The deterministic baseline is a diagnostic comparison.
The repaired real HTTP sequence returned 200 for both preview and evaluation.

The final adapter matched the step-28 checkpoint byte for byte (SHA-256
`8e37536c61a11854906dc5a21e04b3e86c178daa18251814c229d524665f5645`).
Its 56 tensors contain 659,456 parameters at rank 4 across four adapted layers.
MLX 0.32.4 and MLX-LM 0.30.6 evaluated it with the pinned 1.5B base. The six
cases in `data/v3/valid.jsonl` match the training validation export (SHA-256
`f2a01afb455699c18066b8e5af1307f0ec1859ec7fa5ced50ec8af0cff64e018`).
Results on this split cannot establish generalization.

| System | Gate passes | Mean identifier recall | Mean repeat ratio | Over repetition limit | Automated unsupported flags | Median generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen raw | 0/6 | 96.7% | 4.1% | 0/6 | 0/6 | 1,407 ms |
| Qwen scaffold | 4/6 | 87.8% | 4.7% | 0/6 | 2/6 | 1,122 ms |
| Trained adapter with scaffold | 3/6 | 77.8% | 8.6% | 2/6 | 0/6 | 1,093 ms |
| Deterministic PR summary | 6/6 | 100% | 0.0% | 0/6 | 0/6 | 0.12 ms |

Latencies exclude model loading. An AI-assisted inspection checked the core
behaviors in each patch. The raw and scaffold base responses completely
stated the core behavior without contradicting it on 4/6 cases each. The
adapter did so on 3/6 and deterministic output on 1/6. These spot-check
counts are separate from the automatic gate. Deterministic output mainly
lists patch evidence and made no unsupported factual claims in this review.

The adapter correctly described the v1 fallback, the two omitted log keys
and both quantity bounds. It omitted the default target `overview` and
the header value `nosniff`. On retention it repeated 14 days and contradicted
it with one week. Automated checks missed the retention contradiction and
the unsupported claim that no new vulnerabilities were introduced. Raw Qwen
claimed code review had completed and invented a header rename. Scaffold Qwen
invented a function rename and a replacement of nonexistent `merge!`.
The separate preview repeated the batch-size change and issued conflicting
risk labels.

The adapter failed qualification at 3/6 versus scaffold at 4/6. The run was
read-only. The registry still holds the candidate as inactive and
experimental. No artifacts were modified and the reserved final split
was not accessed. Start a fresh loopback server to pick up the fix and
use the new `watch` command to stop polling at completed or failed.

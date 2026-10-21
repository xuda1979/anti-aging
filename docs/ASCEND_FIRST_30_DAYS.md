# First 30 days: Ascend R&D work order

**David Xu, PhD; Peter Lee, PhD, MD; and Zhimin Zheng, PhD**

21 October 2026. Follow the [detailed plan](ASCEND_RND_PLAN.md) and [resource matrix](ASCEND_EXPERIMENT_MATRIX.json). This is a proposed work order; all NPU hardware qualification and training results remain UNKNOWN. Day numbers start when actual host access and responsible staff are available.

## First decisions

Assign an infrastructure owner, a data/split owner, an ML implementer and an independent evaluator. One person can cover development roles; critical scoring requires an independent check. Confirm the eight reported 910C and eight reported 910B devices against the runtime-visible device count and actual host topology. Keep their training groups separate initially.

The immediate objective is one verified lifecycle per pool. Large datasets, longer sequences and eight-rank jobs follow its gates. Existing CPU reports remain the scientific baseline.

## Read-only inventory on each host

Run these commands inside the intended existing NPU environment and retain their output privately with the host inventory. They do not install packages, modify drivers or train a model. Do not publish private host/network details from raw logs.

```bash
npu-smi info
```

```bash
python -m pip show torch torch_npu
```

```python
import json
import platform
import sys
import torch
import torch_npu

report = {
    "python": sys.version,
    "architecture": platform.machine(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_available": bool(torch.npu.is_available()),
}
report["visible_device_count"] = (
    int(torch.npu.device_count()) if report["npu_available"] else 0
)
print(json.dumps(report, indent=2))
```

Import failures and zero devices are failed/unknown environment evidence, not a reason to install arbitrary latest packages. The administrator additionally records supported product/OS, CANN, driver/firmware, HBM, CPU/RAM, storage, topology and network/collective qualification. Private credentials and addresses are outside the public report.

The existing CPU verification command, run in the repository's separate pinned CPU environment after its documented acquisition/setup, is:

```bash
python scripts/verify_study.py --run pilot_20261021_v2
```

This command verifies the old CPU study. It does not test Ascend or train the planned large model. Current NPU launch scripts and the model modules described below **still need implementation**; there is no claimed ready-to-run `torchrun` command in this planning PR.

## Week 1: qualify the environment and the measurement path

| Day | Work / owner | Required artifact | Exit condition |
|---|---|---|---|
| 1 | Infrastructure: complete separate B/C inventories; data owner: open original metadata/rights records | Redacted hardware manifest; exact supported software candidates; missing-field list | G0 can be judged; device/board mapping and availability are explicit. |
| 2 | Infrastructure: isolate selected environments and preserve rollback; data: audit aliases and external-age eligibility | Image/package hashes; source registry; donor/control/unit dictionaries | No change to global host packages or the historical CPU environment. |
| 3 | ML: implement device abstraction and a tiny ordinary-operator model; evaluator: freeze CPU fixture/loss/reference | First small implementation PR, fixture hashes and proposed numeric tolerances | Fixture is non-degenerate; no silent CPU fallback accepted as NPU success. |
| 4 | ML/infrastructure: forward/backward/update on one B and one C logical device | Loss, parameter-update, memory and environment reports | Finite values, actual NPU execution and G1/G2 pass independently for each pool. |
| 5 | ML/evaluator: save/terminate/resume; qualify AMP separately | Same-next-batch/LR/update comparison; full checkpoint manifest | Optimizer, RNG, sampler and scheduler are restored; FP32 and AMP verdicts stay distinct. |
| 6 | Data: build deterministic sparse shards from a bounded real-data subset; isolate evaluation studies | Source-to-shard hashes, counts, exclusions, donor/guide overlap audit | No fitting transform sees sealed outcomes; wrong-unit/overlap inputs are rejected. |
| 7 | Evaluator: examine all week-1 evidence and failure cases | PASS/FAIL/UNKNOWN gate review and first cost ledger | Larger allocations remain blocked if lifecycle or provenance is unresolved. |

For single-device real-data canaries, use at most 30 minutes, then admit a longer job only after the gate is reviewed. Every allocation consumes its matrix row cap, including aborted runs.

## Week 2: distributed correctness, baselines and data readiness

| Day | Work | Acceptance |
|---|---|---|
| 8 | Two-rank HCCL fixture separately in B and C pools | Global loss/gradients match the single-process concatenated-batch reference within the frozen tolerance. |
| 9 | Four-rank fixture and one-worker termination injection | All workers terminate coherently; no orphan or duplicate trainer; latest-good state survives. |
| 10 | Eight-rank fixture only on a qualified homogeneous topology | Correct visible-device mapping, unique data accounting and complete checkpoint; otherwise use smaller groups. |
| 11 | Input/collective profiling at fixed global batch and length distribution | Cold and steady-state costs separated; measured memory/throughput and E4/E8 reported; CPU fallback/stalls investigated. |
| 12 | Run small Ridge/PCA/response and count-model baselines where their data/software permit | Same development budget, splits and biological units; CPU alternatives documented if a third-party NPU path is unsupported. |
| 13 | Freeze RNA-A and target/combination/context contracts; audit 100k-cell canary and candidate external cohorts | Sealed test-study exclusion before pretraining; cell/donor/target counts and assay compatibility documented. |
| 14 | Independent gate review and cost forecast | G4/G5 pass; data and all planned primary metrics frozen; proposed duration fits the remaining row cap. |

GSE307377 and GSE241430 provide only nine and three candidate people respectively before exclusions. Treat small external results as descriptive unless the full donor/cohort requirement is actually met. Additional metadata-only cohort search and rights/independence review are part of the data owner's work.

## Week 3: first useful NPU model

| Day | Work | Acceptance |
|---|---|---|
| 15 | RNA-A 100k-cell canary, fixed length 512 then 1024 if measured headroom permits | Healthy updates, expressed/zero loss breakdown, checkpoint and throughput evidence. |
| 16–17 | Conditional 1M-cell RNA-A run; exact exposure from the main plan | Actual steps/cells/tokens and B/C device-hours; no substituted incomplete checkpoint. |
| 18 | Frozen-embedding downstream comparisons and development-only model selection | Strongest simple baseline retained; weak/negative results and cohort/age-band failures visible. |
| 19 | Prepare conditional-response adapter inputs and context-matched-control fixtures; training waits for the P04 gate | No fabricated same-cell before/after pairing; held-out targets and mirrored guides remain excluded. |
| 20 | Seeds and ablations within the aggregate row budgets | All declared seeds reported; no best-seed promotion. |
| 21 | Independent review: RNA-A gain, support and remaining cost | RNA-B stays blocked if RNA-A has no validated incremental value or data diversity is inadequate. |

## Week 4: decide whether scaling is worthwhile

| Day | Work | Acceptance |
|---|---|---|
| 22–24 | Train response adapters after P04; run conditional RNA-B on a disjoint allocation if justified, or fix the smaller system's bottleneck | Separate hypotheses and allocations; exact shapes/parameter counts; unchanged scoring contracts and budgets. |
| 25 | Combination/context transfer and confounding controls | Report each regime and unsupported contexts separately; large unchanged-gene correlation cannot stand in for effect prediction. |
| 26 | Calibrate intervals/abstention on validation, then evaluate the frozen procedure | Risk-coverage and interval-coverage evidence with true independent-unit counts. |
| 27 | Independent scoring and checkpoint/source/data authentication | Corrupt/stale prediction or checkpoint rejected; outputs link to the executed source. |
| 28 | Produce the first-month scorecard and hypothesis cards | All attempted runs, exclusions, costs, negative results and unknown biological endpoints listed. |
| 29 | Verify release files/manifests and reproduce a selected computational result | Tested source matches packaged artifact; historical evidence unchanged. |
| 30 | Go/no-go review and weeks 5–12 resource forecast | Retain a useful smaller model or baseline when scaling fails; new phase gets its own measured budget and contract. |

## Admission checklist for every nontrivial job

- [ ] Unique run ID; immutable code/config/data/split/environment references; no output-directory reuse.
- [ ] Exact allocated logical devices and qualified same-type rank group; no concurrent allocation overlap.
- [ ] Row budget remaining and wall cap explicit; three-seed cost included where required.
- [ ] Data units, donor/guide/target exclusions, input shapes and valid-mask counts checked.
- [ ] Correct device, backend, precision, model/trainable parameters and effective batch echoed in the log.
- [ ] Tiny lifecycle and relevant numeric/collective canaries passed under the same contract.
- [ ] Storage/checkpoint headroom and known-good resume point verified.
- [ ] Independent scoring owner and primary metric/guardrails frozen before final-test access.
- [ ] Stop conditions active; a failing rank or exhausted budget stops the job and preserves diagnostic evidence.

## First-month handoff

The deliverable is an inspectable computational result or an informative failed hypothesis. Supply source/data/environment manifests, final checkpoint and prediction hashes, all candidate/seed results, independent scoring, B/C hours, throughput/HBM/scaling profiles and explicit biological limitations. The 2,048 C + 1,024 B device-hour ceilings are upper bounds across the whole first-month matrix. No new NPU result is claimed by this checklist.

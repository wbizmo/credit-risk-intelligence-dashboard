# CRIX portfolio dependency, GPU and EVT research

CRIX keeps portfolio simulation and tail extrapolation inside the offline Python research boundary. None of the capabilities in this document alter `/api/v3`, the live CRIX-MonoBoost runtime, or the public synchronous batch limit.

All outputs remain **research-only; not regulatory capital or bank validation**.

## Canonical Monte Carlo path

`simulate_portfolio(...)` continues to default to:

- `backend="numpy"`;
- one-factor Gaussian latent-default dependence;
- counter-indexed deterministic random draws;
- bounded chunk memory;
- empirical VaR/ES only when the existing scenario-support policy is satisfied.

The canonical NumPy one-factor Gaussian path preserves seeded `lossDigest` compatibility with the pre-v3.2 implementation. Chunk size does not change an equivalent seeded run.

## Tail attribution

The v3.2 tail-contribution path no longer replays the complete scenario/obligor simulation once per quantile.

After the first pass has produced the scenario loss vector and empirical thresholds, a single bounded-memory replay:

1. regenerates each scenario/default chunk once;
2. forms that chunk's weighted obligor losses once;
3. assigns scenarios to disjoint loss-threshold buckets;
4. accumulates each scenario row directly into one backend-native disjoint bucket;
5. cumulatively combines the bucket totals to recover every supported tail set.

The final v3.2 audit removed the earlier per-bucket mask scan inside each chunk, so the implementation now matches the documented complexity rather than carrying a hidden `O(S*Q)` mask term.

For `S` scenarios, `N` obligors and `Q` supported quantiles, the dominant contribution replay is:

```text
O(S*N + S*log(Q) + Q*N)
```

rather than replaying the full `S*N` simulation `Q` times. Chunk memory remains `O(C*N + S + Q*N)`; no full `S x N` matrix is retained.

The reference replay is retained only as a private benchmark/test oracle. Before the final indexed-bucket optimization, the shared replay measured approximately **2.80×** the reference throughput for three quantiles and **4.63×** for five quantiles in CI; the release branch reruns the benchmark rather than assuming those ratios improved.

## Dependency models

The legacy `rho=` argument remains backward compatible.

Explicit research configuration is available through `dependency_model`:

```python
simulate_portfolio(
    pd,
    lgd,
    ead,
    dependency_model={"name": "gaussian", "rho": 0.20},
)
```

Student-t dependence uses a common scale-mixture construction with matching Student-t marginal default thresholds:

```python
simulate_portfolio(
    pd,
    lgd,
    ead,
    dependency_model={"name": "student-t", "rho": 0.20, "df": 5},
)
```

Low-rank multi-factor dependence accepts an `N x K` loading matrix:

```python
simulate_portfolio(
    pd,
    lgd,
    ead,
    dependency_model={
        "name": "gaussian",
        "loadings": [[0.25, 0.10], [0.15, 0.20]],
    },
)
```

The implementation rejects non-finite parameters, invalid dimensions, unsupported model names, Student-t degrees of freedom at or below 2, more than 32 factors, and any obligor whose sum of squared factor loadings exceeds 1 (within numerical tolerance). Residual idiosyncratic variance is derived only after those checks.

The normal low-rank path has `O(S*N*K)` simulation work and `O(C*(N+K))` chunk memory. CRIX does not require a dense `N x N` correlation matrix for ordinary portfolio research.

## Optional CuPy backend

GPU acceleration is opt-in:

```python
simulate_portfolio(pd, lgd, ead, scenarios=1_000_000, backend="cupy")
```

NumPy remains the default and canonical evidence path. CuPy is lazy-imported only when requested, and the live Node runtime has no Python or CUDA dependency.

CPU environment:

```bash
pip install -r model/requirements.txt
```

Optional CUDA 12 research environment:

```bash
pip install -r model/requirements-gpu-cuda12.txt
```

Use the CuPy package matching the actual CUDA runtime when it is not CUDA 12. Requesting `backend="cupy"` without a usable CuPy/CUDA installation fails with an actionable error instead of silently falling back to CPU.

Every simulation reports backend name, device, dtype, seed, chunk size and reproducibility class. CPU `lossDigest` remains the canonical reproducibility artifact. GPU runs are evaluated for same-backend repeatability and statistical agreement; CPU/GPU digest equality is not claimed.

Invariant thresholds and low-rank loading/residual arrays are transferred/prepared once per simulation and reused across chunks rather than being recreated inside the `O(S/C)` chunk loop.

Run the benchmark harness with:

```bash
python model/benchmark_portfolio.py --quick
python model/benchmark_portfolio.py
```

The output records wall time, peak host RSS, scenarios/portfolio sizes, tail-attribution speedup, and—when a usable GPU exists—device identity, memory-use evidence, transfer overhead and CPU/CuPy speedup. A CPU-only machine reports GPU unavailability explicitly; it does not fabricate acceleration evidence.

### v3.2 GPU evidence boundary

The v3.2.0 release infrastructure available to this repository is CPU-only. Therefore **no CuPy speedup, crossover point, GPU-memory peak or transfer-vs-compute claim is published for this release**. The benchmark harness records all of those fields when executed on documented CUDA hardware, and optional GPU tests cover same-backend repeatability/statistical agreement when a CUDA device exists. NumPy remains the canonical release evidence path.

## Empirical Monte Carlo versus EVT

Empirical Monte Carlo VaR/ES remains unchanged and continues to be withheld where scenario support is inadequate.

`model/evt.py` adds a **separate** Peaks-Over-Threshold Generalized Pareto analysis:

```python
from evt import fit_pot_tail

evt = fit_pot_tail(
    losses,
    threshold_quantile=0.975,
    target_quantiles=(0.99, 0.999),
    bootstrap_samples=100,
)
```

EVT output never overwrites empirical `var` or `expectedShortfall`. It reports:

- threshold quantile/value;
- strict exceedance count/fraction;
- GPD shape `xi` and scale `beta`;
- fitted tail quantiles;
- expected shortfall only when the fitted mean is finite (`xi < 1`);
- threshold-stability diagnostics across candidate thresholds;
- optional bounded bootstrap confidence intervals;
- explicit `insufficient-data` or `unstable-fit` states;
- fit method and research-only disclaimer.

No favorable-estimate selection is performed. A failed or unstable GPD fit remains failed/unstable rather than being replaced by an empirical or hand-picked number.

## Synthetic validation

`python model/validate_tail_models.py` produces machine-readable evidence for:

- independent Gaussian expected-loss behavior;
- stronger tail concentration under stronger common-factor dependence;
- Student-t joint-tail behavior under matched central dependence assumptions;
- GPD recovery for light, moderate and heavy synthetic tails;
- threshold-stability diagnostics.

These are controlled research checks, not Basel/IRB validation or evidence that a synthetic dependence structure is appropriate for a real portfolio.

## CI policy

CPU CI intentionally installs no CuPy/CUDA dependency. It verifies:

- legacy Gaussian digest and signature compatibility;
- shared-vs-reference tail-attribution equivalence;
- one/two/three-plus supported quantiles;
- unsupported tails, zero LGD/EAD and concentrated exposures;
- dependency parameter validation;
- Gaussian/Student-t deterministic chunking;
- marginal expected-loss behavior;
- controlled tail-dependence behavior;
- clean optional-GPU failure when CuPy is absent;
- EVT parameter recovery, degeneracy handling, finite-ES rules and bounded bootstrap behavior.

Wall-clock benchmark results are evidence, not brittle correctness thresholds.

# Randomised Lie-group integrators

Reproducible numerical experiments for randomised manifold-preserving integration of deterministic ODEs with low temporal regularity. This repository is now named **randomisedLieGroup** (formerly randomisedLie).

Implemented methods include randomised Lie-Euler (RLE), two-stage randomised Runge-Kutta-Munthe-Kaas (RRKMK2), two-stage randomised Lie commutator-free (RLCF2), their reflected/paired-node variants, and deterministic Lie-Euler, midpoint and fourth-order RKMK controls. The ODE is deterministic; only the numerical sampling times are random. The suffix 2 labels the two-stage single-node construction, not an unconditional strong order of two.

## Environment

Tested with Python 3.13.5 and the existing pinned dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_v4_YW_01102026.txt
```

Windows users can use the equivalent virtual-environment activation command and omit or set the shell environment variables below as appropriate.

## New: manifold plots and exponential-action diagnostics (v5)

From the repository root:

```bash
python experiments/test_manifold_diagnostics_v5_YW_27102026.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python experiments/manifold_diagnostics_v5_YW_27102026.py \
  --paths 256 --steps 64 --seed 27102026 --out results/manifold_v5

python experiments/plot_manifolds_v5_YW_27102026.py \
  --results results/manifold_v5 --out figures
```

This writes compressed trajectory arrays, a JSON numerical summary, and six separate PDF/PNG figures:

1. Actual trajectories on S^2, with an unprojected Euclidean randomised RK2 control.
2. Sphere-constraint drift over time for all methods.
3. All randomised endpoints in a tangent chart at the fixed exact endpoint.
4. SO(3) through the trajectories of all three rotation-frame axes.
5. Exponential approximation alone on a constant-rotation equation.
6. Common-node paired RLCF2 with Rodrigues versus Cayley at every action.

The illustration always uses path zero, not the best-looking realisation. Numerical states are never projected or renormalised. SO(3) is not identified with S^2, and no intrinsic error is assigned to the off-manifold Euclidean control. The JSON records the seed, parameters, environment, original-core SHA256, and diagnostic residuals. The independent small regression suite contains five tests.

These are additional diagnostics, **not** a rerun of the complete v4 campaign below. They do not establish universal or equal-cost superiority. The checked-in small validation summary records the v5 run; large arrays and generated figures remain untracked.

For the mathematical conventions, sufficient exponential-error budget, scope of the Cayley replacement, and primary references, see [norms and exponentials](docs/norms_and_exponentials_v5_YW_27102026.md).

## Full v4 paper campaign (unchanged numerical core)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python experiments/randomised_lie_verification_v4_YW_01102026.py \
  --mode all --paths 1024 --maxN 1024 --out results

python experiments/independent_checks_v4_YW_01102026.py --out results

python experiments/plot_results_v4_YW_01102026.py \
  --results results --out figures
```

The v4 plotting script has been repaired: the last plot now calls `ax.legend(...)`, not the invalid `ax.gend()`. Its four plots were regenerated from the archived v4 statistics to test this repair. No v4 integrator formula or archived numerical table was changed.

A smaller numerical smoke test is:

```bash
python experiments/randomised_lie_verification_v4_YW_01102026.py \
  --mode main --paths 64 --maxN 128 --out smoke_results
```

The four-panel-set plotting command requires the complete `--mode all` result summary, not smoke-test output. Do not mix smoke and paper results.

## References and numerical scope

The rough SO(3) and state-dependent S^2 problems use fixed infinite-series coefficients with explicit reference trajectories, not time-stepping reference solvers or a mesh-dependent finite cutoff. The smooth controls use refined DOP853 references as numerical diagnostics. Original code and mathematical limitations are documented in the accompanying manuscript.

Closed-form Rodrigues/quaternion formulas preserve the manifold in exact arithmetic; measured floating-point defects are reported rather than called exact machine preservation. Cayley is used only where it is group-valued and to sufficient local order. Retaining a convergence order does not retain every refined asymptotic bias constant.

Generated `results/`, `smoke_results/`, and `figures/` are ignored. The private working manuscript and provisional author discussion are not published in this code repository.

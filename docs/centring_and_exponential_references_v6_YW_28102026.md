# Conditional centring and exponential accuracy

Version: v6_YW_28102026. Explanatory note; no new convergence theorem is claimed.

## Why Lipschitz continuity does not preserve unbiasedness

For a step from fixed x, write V=u(h)+Q+R, with E Q=0,

$$\|Q\|\le Ch^{1+\alpha},\qquad \|R\|\le Ch^{2+\alpha}.$$

Only Q is centred. The full algebra error Q+R need not be centred. Set

$$\delta=H(u(h)+Q+R,x)-H(u(h),x),\qquad H(v,x)=\exp(v)\cdot x.$$

The Lipschitz estimate bounds the L2 magnitude by O(h^{1+alpha}). By itself it gives only the same loose order for the mean, through ||E delta|| <= E||delta||. Taking absolute values before averaging loses the cancellation.

An example already on S1 is decisive. Let x=(1,0), let J be the rotation generator, and let Q=+epsilon or -epsilon with equal probabilities. Then E Q=0, while

$$\mathbb E[e^{QJ}x]-x=(\cos\varepsilon-1,0)=(-\varepsilon^2/2,0)+O(\varepsilon^4).$$

The action q -> (cos q,sin q) is globally 1-Lipschitz for Euclidean chord distance. Both realisations stay on the circle, but their ambient mean lies inside it. This does not contradict pathwise manifold preservation. No intrinsic manifold expectation is used here.

## Conditional Taylor argument

At step n, condition on F_n, the information before the new independent uniform node is drawn. Both Y_n and the exact local endpoint u_n(h) are F_n-measurable. So is D_n=D_u H(u_n(h),Y_n). Taylor's formula on the common small algebra ball gives

$$\delta_n=D_nQ_n+D_nR_n+T_n,\qquad \|T_n\|\le C\|Q_n+R_n\|^2.$$

The essential cancellation is

$$\mathbb E[D_nQ_n\mid F_n]=D_n\mathbb E[Q_n\mid F_n]=0.$$

Unconditional independence of D_n and Q_n is neither claimed nor needed. R_n and T_n may be correlated with Q_n. Expanding around the sampled exponent V_n instead would not justify taking its derivative outside this conditional expectation.

Therefore, for h<=1,

$$\|\mathbb E[\delta_n\mid F_n]\|\le C(h^{2+\alpha}+h^{2+2\alpha})\le Ch^{2+\alpha}.$$

The centred conditional RMS remains O(h^{1+alpha}). The global mean budget is O(h^{1+alpha}) and the martingale fluctuation budget is O(h^{alpha+1/2}). These are upper bounds, not necessarily nonzero leading asymptotic terms.

The derivatives are of the smooth action in its algebra argument, not of the time-Hoelder coefficient. Thus this step does not add time differentiability to the assumptions.

**Correct statement:** nonlinear reconstruction preserves the useful local moment orders, not exact unbiasedness. The quadrature is unbiased; the complete numerical step generally is not.

## Distinguish the exponential questions

1. **Exact reconstruction:** local Lipschitz continuity of H transfers an algebra error magnitude to a manifold metric error at a common base point. Local-to-global stability is still needed. The Taylor argument is separately needed for the mean.
2. **Approximating the exponential action:** a uniform action error epsilon_h, including internal actions, has a worst-case global budget O(epsilon_h/h) under our stability assumptions. Thus epsilon_h=O(h^{p+1}) suffices to retain global order p. Group membership is a separate requirement. This is an established perturbation principle, not a new general principle invented here.
3. **Floating-point evaluation:** scaling/squaring and Pade analysis controls matrix-function error and conditioning. It does not itself supply the randomised ODE convergence theorem or exact group membership in machine arithmetic.

Cayley on real skew matrices is group-valued in exact arithmetic and has O(h^3) local error for O(h) arguments. It retains a second-order bound but may change the leading second-order bias. It is not claimed to work on every matrix group. A Taylor polynomial may have sufficient approximation order without preserving the group.

## Primary literature and exactly what it supports

- Iserles, Munthe-Kaas, Norsett and Zanna (2000), **Lie-group methods**, Acta Numerica 9, 215-365. DOI: https://doi.org/10.1017/S0962492900002154 . Section 8, Computation of exponentials, explicitly requires both adequate approximation order and group membership. Author-hosted text: https://www.damtp.cam.ac.uk/user/na/NA_papers/NA2000_03.pdf . The relevant paragraph is on p. 97 of this preprint. Use the published year, not the inconsistent header date in the preprint.
- Curry and Schmeding (2020), **Convergence of Lie group integrators**, Numerische Mathematik 144, 357-373. DOI: https://doi.org/10.1007/s00211-019-01083-1 . Online publication was in 2019, journal volume year is 2020. Preprint: https://arxiv.org/abs/1807.11829 . Sections 3-4 relate local errors to metric errors and global convergence. Smooth deterministic hypotheses: not a direct theorem for our randomised low-time-regularity setting.
- Celledoni and Iserles (2000), **Approximating the exponential from a Lie algebra to a Lie group**, Mathematics of Computation 69, 1457-1480. DOI: https://doi.org/10.1090/S0025-5718-00-01223-0 . Group-valued exponential approximation.
- Celledoni and Iserles (2001), **Methods for the approximation of the matrix exponential in a Lie-algebraic setting**, IMA Journal of Numerical Analysis 21, 463-488. DOI: https://doi.org/10.1093/imanum/21.2.463 . Preprint: https://arxiv.org/abs/math/9904122 . This arXiv identifier belongs to the 2001 IMA paper, not the preceding 2000 Mathematics of Computation paper.
- Al-Mohy and Higham (2009), **A New Scaling and Squaring Algorithm for the Matrix Exponential**, SIAM Journal on Matrix Analysis and Applications 31(3), 970-989. DOI: https://doi.org/10.1137/09074721X . Author-deposited copy: https://eprints.maths.manchester.ac.uk/1442/ . Backward-error and finite-precision analysis of matrix exponential evaluation.

## Redrawn figures

The v6 plotting script reads the unchanged v4 statistics and v5 saved trajectories. It redraws the ten previous empirical figures using distinct dashes, hollow markers and longer legend samples. One additional sphere_error_vs_time plot shows pointwise empirical RMS ambient errors from the same ensemble. It is not the grid-maximum-over-time statistic in the theorem. No simulation is rerun, no favourable path selected, and no trajectory displaced or projected for visibility. Default colours are retained. All eleven figures are separate PDF/PNG files.

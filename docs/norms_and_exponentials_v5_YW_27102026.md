# Norms, exponential actions, and implementation error

Companion notes for the v5 numerical diagnostics. These describe the norm and implementation conventions; numerical evidence is not a substitute for the manuscript proofs.

## 1. The Lie-algebra norm

The Lie algebra is a finite-dimensional real vector space. Fix an inner product and its norm, independently of the step size. All fixed norms are equivalent, so convergence exponents transfer between them with changed constants. This does not give dimension-independent constants or cover degenerating step-dependent norms. Exact variance and martingale orthogonality identities use the inner product; they should not be asserted unchanged for an arbitrary norm.

The bilinear bracket has a finite bound

$$\|[u,v]\|_{\mathfrak g}\le C_{\mathrm{br}}\|u\|_{\mathfrak g}\|v\|_{\mathfrak g}.$$

No Ad-invariant metric or positive Killing form is assumed. An abstract Lie algebra need not have associative multiplication, so it does not require a submultiplicative matrix norm.

In the code, `hat(w) v = cross(w,v)`. We use the Euclidean norm of `w`, equivalently the Frobenius matrix norm divided by the square root of two:

$$\|\widehat w\|_F=\sqrt{2}|w|_2,\qquad\|\widehat w\|_{2\to2}=|w|_2.$$

Sphere distances use the standard round metric. SO(3) geodesic distance is the rotation angle, corresponding to the normalised Frobenius inner product on its Lie algebra. Matrix-error and orthogonality diagnostics use the unnormalised Frobenius norm.

## 2. Three different meanings of exponential convergence

**Exact-map error transfer.** With `H(u,x)=exp(u).x`, smoothness on a compact local set gives

$$d_M(H(u,x),H(v,x))\le C\|u-v\|_{\mathfrak g}.$$

This preserves an error upper bound when the base point is the same. Propagation from different states additionally needs one-step stability. It is neither an isometry nor an assertion of exponential-in-1/h convergence. Isotropy can collapse different algebra elements to the same state, so a reverse inequality need not hold. The group exponential is also not generally the Riemannian exponential at a manifold point.

For matrices,

$$e^U-e^V=\int_0^1e^{(1-s)U}(U-V)e^{sV}\,ds.$$

For skew U,V this implies `||exp(U)-exp(V)||_F <= ||U-V||_F`. It does not make the expectation of a nonlinear push-forward unbiased. The manuscript separately controls that bias by a Taylor expansion.

**Series convergence.** The matrix-exponential power series converges for every finite matrix. The Bernoulli series for `dexp` inverse is a local construction, not an entire series. The manuscript uses it near zero.

**Numerical evaluation error.** Let the implemented action remain on M and have uniform absolute action error `epsilon_h` on exponents of size O(h). Coefficient and inverse-differential evaluations are assumed exact in this statement. For a fixed finite number of stages, same-state one-step error is O(epsilon_h): an internal-stage perturbation contributes O(h epsilon_h), and the final action contributes O(epsilon_h). With exact-action stability 1+Ch, common-node discrete Gronwall gives

$$\max_n d_M(\widetilde Y_n,Y_n)\le C\epsilon_h/h.$$

An order-p exact-action method therefore has implemented error bounded by

$$C(h^p+\epsilon_h/h).$$

A sufficient action tolerance is O(h^(p+1)); uncentred implementation errors cannot be assumed to accumulate like independent sampling errors. A fixed floating-point tolerance does not by itself establish h-to-zero convergence. This is a sufficient perturbation bound, not a universal backward-error guarantee for an arbitrary library routine.

## 3. Cayley versus polynomial approximations

For real skew matrices,

$$C(U)=(I-U/2)^{-1}(I+U/2)\in SO(d),\qquad C(U)-e^U=U^3/12+O(\|U\|^4).$$

Thus replacing every exponential action, including internal stages, by Cayley gives an O(h^2) global perturbation on the sphere and orthogonal-group examples. This is enough for the low-regularity orders alpha+1/2 (alpha <= 1) and for smooth paired order two. It need not retain the refined second-order bias constant or special order-5/2 cancellations. Cayley is not group-valued for every arbitrary Lie algebra. Here it is an approximation of an existing exponential action, not a new RKMK derivation in Cayley coordinates.

A quadratic Taylor polynomial has sufficient local approximation order for second order, but generally leaves the group. Already `(I+U).T @ (I+U) = I-U^2`, not I. Numerical order and manifold preservation are different requirements. Generic off-manifold implementations fall outside the group-valued-action perturbation result.

## 4. What the new experiments do

`manifold_diagnostics_v5_YW_27102026.py` imports the unchanged v4 core. The sphere illustration uses alpha=0.5, gamma=5, kappa=4, 64 steps, 256 paths and seed 27102026. Path zero is selected before seeing errors. All endpoints are included in the endpoint chart. No trajectory is projected or renormalised.

The off-manifold control is ordinary Euclidean randomised two-stage RK, using the same explicitly extended vector field. Only ambient errors are assigned to it. Its error is not relabelled as a geodesic distance after projection.

The SO(3) visualisation plots all three frame axes R(t)e_i; it does not identify SO(3) with the sphere. The exponential test uses a constant skew coefficient, so exact exponential stepping has no time-discretisation error. A separate common-node comparison uses Rodrigues versus Cayley at every stage of paired RLCF2. The small regression suite compares recorded paths with the original v4 streaming implementation and tests algebra, exponential, and constraint identities.

These are additional diagnostics, not a rerun of the archived full v4 campaign. Finite-sample slopes and constraint residuals are not rigorous enclosures or cost-normalised dominance claims. In the new sphere example, single-node RLCF2 has slightly larger grid-RMS error than deterministic midpoint, while pairing is more accurate at the specified step count and different cost.

## Primary references

- E. Celledoni and A. Iserles, *Methods for the approximation of the matrix exponential in a Lie-algebraic setting*, IMA J. Numer. Anal. 21 (2001), 463-488. [Author preprint](https://arxiv.org/abs/math/9904122). Discusses order and group-valuedness of exponential approximants.
- A. Iserles, H. Z. Munthe-Kaas, S. P. Norsett and A. Zanna, *Lie-group methods*, Acta Numerica 9 (2000), 215-365. [DOI](https://doi.org/10.1017/S0962492900002154). See the computation-of-exponentials discussion.
- C. Curry and A. Schmeding, *Convergence of Lie group integrators*, Numer. Math. 144 (2020), 357-373. [DOI](https://doi.org/10.1007/s00211-019-01083-1). Relates local errors and global metric errors in the deterministic setting; it does not prove the present low-regularity randomised theorem.
- A. H. Al-Mohy and N. J. Higham, *A new scaling and squaring algorithm for the matrix exponential*, SIAM J. Matrix Anal. Appl. 31 (2009), 970-989. [Author repository](https://eprints.maths.manchester.ac.uk/1442/). Numerical evaluation, conditioning and backward-error considerations.

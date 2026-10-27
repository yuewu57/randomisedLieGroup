#!/usr/bin/env python3
"""Reproducible trajectories, constraint checks, and exponential-action diagnostics.

Extends, without changing, the v4 integrators and exact-reference rough equations.
Only auxiliary sampling nodes are random. No trajectory is projected or normalised.
Run this script, then plot_manifolds_v5_YW_27102026.py. See --help.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import expm
from numba import njit
import randomised_lie_verification_v4_YW_01102026 as core

TAG = 'v5_YW_27102026'
NAMES = ['RLCF2', 'RRKMK2', 'paired_RLCF2', 'paired_RRKMK2',
         'CF_midpoint', 'ambient_random_RK2']

@njit(cache=True)
def action(v, y, cayley=False):
    if not cayley:
        return core.rotate(v, y)
    z = core.cross(v, y)
    return y + (z + 0.5*core.cross(v, z))/(1 + 0.25*np.dot(v, v))

@njit(cache=True)
def sphere_paths(nodes, alpha, gamma, kappa):
    """All grid states; first four methods share exactly the same random nodes."""
    P, N = nodes.shape
    h = 1.0/N
    ys = np.empty((P, N+1, 6, 3))
    for p in range(P):
        for m in range(6):
            ys[p, 0, m] = core.Y0
        for n in range(N):
            t = n*h
            w0, q0 = core.data(t, alpha, gamma, False)
            yr0 = core.qapply(q0, core.Y0)
            for m in range(6):
                y = ys[p, n, m].copy()
                r = 0.5 if m == 4 else nodes[p, n]
                wr, qr = core.data((n+r)*h, alpha, gamma, False)
                yrr = core.qapply(qr, core.Y0)
                A = core.sphere_a(w0, yr0, y, kappa)
                if m == 5:
                    # Explicit extension of the same coefficient to ambient R^3.
                    z = y + r*h*core.cross(A, y)
                    B = core.sphere_a(wr, yrr, z, kappa)
                    yn = y + h*core.cross(B, z)
                else:
                    u = r*h*A
                    z = core.rotate(u, y)
                    B = core.sphere_a(wr, yrr, z, kappa)
                    if m == 1 or m == 3:
                        B = core.dinv(u, B)
                    if m == 2 or m == 3:
                        s = 1-r
                        ws, qs = core.data((n+s)*h, alpha, gamma, False)
                        us = s*h*A
                        bs = core.sphere_a(ws, core.qapply(qs, core.Y0),
                                           core.rotate(us, y), kappa)
                        if m == 3:
                            bs = core.dinv(us, bs)
                        B = 0.5*(B + bs)
                    yn = core.rotate(h*B, y)
                ys[p, n+1, m] = yn
    return ys

@njit(cache=True)
def paired_backends(nodes, alpha, gamma, kappa):
    """Coupled paired RLCF2 with Rodrigues vs Cayley in EVERY action."""
    P, N = nodes.shape
    h = 1.0/N
    maximum = np.zeros((P, 2))
    gaps = np.zeros(P)
    for p in range(P):
        ys = np.empty((2, 3))
        ys[0] = core.Y0
        ys[1] = core.Y0
        for n in range(N):
            w0, q0 = core.data(n*h, alpha, gamma, False)
            for k in range(2):
                y = ys[k].copy()
                A = core.sphere_a(w0, core.qapply(q0, core.Y0), y, kappa)
                B = np.zeros(3)
                for r in (nodes[p, n], 1-nodes[p, n]):
                    wr, qr = core.data((n+r)*h, alpha, gamma, False)
                    z = action(r*h*A, y, k == 1)
                    B += 0.5*core.sphere_a(wr, core.qapply(qr, core.Y0), z, kappa)
                ys[k] = action(h*B, y, k == 1)
            _, qref = core.data((n+1)*h, alpha, gamma, False)
            yr = core.qapply(qref, core.Y0)
            for k in range(2):
                maximum[p, k] = max(maximum[p, k], core.sdist(ys[k], yr))
            gaps[p] = max(gaps[p], core.sdist(ys[0], ys[1]))
    return maximum, gaps

@njit(cache=True)
def rotation_paths(nodes, alpha, gamma):
    """One complete SO(3) path per method: midpoint, RLCF2, paired RLCF2."""
    N = len(nodes)
    qs = np.zeros((N+1, 3, 4))
    qs[0, :, 0] = 1.0
    h = 1.0/N
    for n in range(N):
        wm, _ = core.data((n+0.5)*h, alpha, gamma, False)
        wr, _ = core.data((n+nodes[n])*h, alpha, gamma, False)
        ws, _ = core.data((n+1-nodes[n])*h, alpha, gamma, False)
        vs = np.empty((3, 3))
        vs[0], vs[1], vs[2] = h*wm, h*wr, h*0.5*(wr+ws)
        for m in range(3):
            qs[n+1, m] = core.qmul(core.qexp(vs[m]), qs[n, m])
    return qs


def matrices(qs):
    flat = qs.reshape((-1, 4))
    out = np.array([np.column_stack([core.qapply(q, e) for e in np.eye(3)])
                    for q in flat])
    return out.reshape(qs.shape[:-1] + (3, 3))


def exponential_checks():
    """Isolate exponential evaluation error with a constant skew coefficient."""
    A = core.hat(np.array([0.6, -0.4, 0.7]))
    exact = expm(A)
    rows = []
    for N in (8, 16, 32, 64, 128, 256, 512):
        U = A/N
        v = np.array([0.6, -0.4, 0.7])/N
        maps = {
            'Rodrigues': np.column_stack([core.rotate(v, e) for e in np.eye(3)]),
            'Cayley': np.linalg.solve(np.eye(3)-U/2, np.eye(3)+U/2),
            'Taylor1': np.eye(3)+U,
            'Taylor2': np.eye(3)+U+U@U/2,
        }
        for method, step in maps.items():
            R = np.eye(3)
            for _ in range(N):
                R = step @ R
            rows.append(dict(N=N, method=method,
                local_error_F=float(np.linalg.norm(step-expm(U))),
                terminal_error_F=float(np.linalg.norm(R-exact)),
                orthogonality_defect=float(np.linalg.norm(R.T@R-np.eye(3))),
                determinant_defect=float(abs(np.linalg.det(R)-1))))
    rng = np.random.default_rng(2710202601)
    residuals = []
    for scale in (1e-9, 1e-6, 1e-3, 0.1, 0.5):
        for _ in range(20):
            v = scale*rng.normal(size=3)
            R = np.column_stack([core.rotate(v, e) for e in np.eye(3)])
            residuals.append(np.linalg.norm(R-expm(core.hat(v))))
    return rows, float(max(residuals))


def verify_recording():
    """Check recorded paths against independent v4 streaming loops."""
    N = 16
    nodes = np.random.default_rng(123).random((4, N))
    refs = core.refs_for(N, 0.5, 1., True, False)
    ys = sphere_paths(nodes, 0.5, 1., 2.)
    te, me, _, _, _ = core.run_random(N, nodes, 0.5, 1., 2., True, False, refs)
    te2 = np.zeros((4, 4)); me2 = np.zeros((4, 4))
    for p in range(4):
        for m in range(4):
            errs = np.array([core.sdist(y, ref) for y, ref in zip(ys[p, :, m], refs)])
            te2[p, m], me2[p, m] = errs[-1], errs.max()
    _, dm = core.run_det(N, 0.5, 1., 2., True, False, refs)
    det = max(core.sdist(y, ref) for y, ref in zip(ys[0, :, 4], refs))
    residual = max(float(np.max(abs(te2-te[:, :4]))),
                   float(np.max(abs(me2-me[:, :4]))), abs(det-dm[1]))
    assert residual < 1e-12, f'v4 recording mismatch: {residual}'
    v = np.array([0.2, -0.3, 0.1]); U = core.hat(v)
    C = np.linalg.solve(np.eye(3)-U/2, np.eye(3)+U/2)
    assert np.linalg.norm(action(v, core.Y0, True)-C@core.Y0) < 1e-13
    return float(residual)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/manifold_v5'))
    p.add_argument('--paths', type=int, default=256)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--seed', type=int, default=27102026)
    args = p.parse_args()
    if args.paths < 2 or args.steps < 8 or args.seed < 0:
        p.error('require paths >= 2, steps >= 8, seed >= 0')
    args.out.mkdir(parents=True, exist_ok=True)
    validation = verify_recording()
    alpha, gamma, kappa = 0.5, 5., 4.
    nodes = np.random.default_rng(args.seed).random((args.paths, args.steps))
    ys = sphere_paths(nodes, alpha, gamma, kappa)
    times = np.linspace(0., 1., args.steps+1)
    refq = np.array([core.data(t, alpha, gamma, False)[1] for t in times])
    refy = np.array([core.qapply(q, core.Y0) for q in refq])
    dense_t = np.linspace(0., 1., 2049)
    dense_q = np.array([core.data(t, alpha, gamma, False)[1] for t in dense_t])
    dense_y = np.array([core.qapply(q, core.Y0) for q in dense_q])
    qpaths = rotation_paths(nodes[0], alpha, gamma)
    Rpaths, Rrefs = matrices(qpaths), matrices(refq)
    np.savez_compressed(args.out/f'manifold_paths_{TAG}.npz',
        times=times, nodes=nodes, sphere=ys, reference_sphere=refy,
        dense_times=dense_t, dense_sphere=dense_y, rotations=Rpaths,
        reference_rotations=Rrefs, dense_rotations=matrices(dense_q))
    stats = {}
    for m, name in enumerate(NAMES):
        e = np.linalg.norm(ys[:, :, m]-refy[None, :, :], axis=-1)
        radial = np.abs(np.linalg.norm(ys[:, :, m], axis=-1)-1)
        entry = dict(endpoint_ambient_rms=float(np.sqrt(np.mean(e[:, -1]**2))),
            grid_ambient_rms=float(np.sqrt(np.mean(e.max(axis=1)**2))),
            max_radial_defect=float(radial.max()))
        if m < 5:
            ge = np.arctan2(np.linalg.norm(np.cross(ys[:, :, m], refy), axis=-1),
                            np.sum(ys[:, :, m]*refy, axis=-1))
            entry['grid_geodesic_rms'] = float(np.sqrt(np.mean(ge.max(axis=1)**2)))
        stats[name] = entry
    backend_rows = []
    for N in (32, 64, 128, 256):
        rr = np.random.default_rng(np.random.SeedSequence([args.seed, N])).random((args.paths, N))
        errs, gaps = paired_backends(rr, 0.5, 1., 2.)
        backend_rows.append(dict(N=N,
            Rodrigues_grid_rms=float(np.sqrt(np.mean(errs[:, 0]**2))),
            Cayley_grid_rms=float(np.sqrt(np.mean(errs[:, 1]**2))),
            coupled_grid_rms=float(np.sqrt(np.mean(gaps**2)))))
    exprows, expresidual = exponential_checks()
    orth = np.linalg.norm(np.swapaxes(Rpaths, -1, -2)@Rpaths-np.eye(3), axis=(-2, -1))
    summary = dict(version=TAG, seed=args.seed, paths=args.paths, steps=args.steps,
        alpha=alpha, gamma=gamma, kappa=kappa, illustrated_path_index=0,
        norm_conventions=dict(algebra='Euclidean R^3 / Frobenius(skw)/sqrt(2)',
            sphere='unit-sphere geodesic; ambient errors also reported',
            SO3='rotation-angle geodesic; matrix Frobenius diagnostics'),
        v4_recording_residual=validation, Rodrigues_expm_residual=expresidual,
        max_SO3_orthogonality_defect=float(orth.max()),
        max_SO3_determinant_defect=float(np.max(abs(np.linalg.det(Rpaths)-1))),
        sphere_statistics=stats, paired_exponential_backends=backend_rows,
        constant_rotation_exponential_test=exprows,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        notes=['Illustrated path is always index zero, not selected for low error.',
               'An SO(3) frame plot shows three unit-vector images, not an embedding SO(3)=S^2.',
               'Ambient RK2 is not manifold-valued, so no intrinsic error is assigned to it.',
               'No action or state is projected or renormalised.',
               'Finite-sample diagnostics, not rigorous enclosures or universal superiority claims.'])
    (args.out/f'manifold_summary_{TAG}.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('constant_rotation_exponential_test','notes')}, indent=2))

if __name__ == '__main__':
    main()

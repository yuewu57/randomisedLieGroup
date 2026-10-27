#!/usr/bin/env python3
"""Create manifold and exponential-error plots from the v5 recorded arrays.

Each figure is separate. Default matplotlib colours are used. Path zero is fixed
in advance; the endpoint scatter uses the entire ensemble. No plotted numerical
trajectory is projected onto the sphere.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TAG = 'v5_YW_27102026'


def sphere_axes(title):
    fig = plt.figure(figsize=(8, 7.4))
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 29), np.linspace(0, np.pi, 17))
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                      linewidth=0.35, alpha=0.16)
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), zlim=(-1.08, 1.08),
           xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$', title=title)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=35)
    return fig, ax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('results/manifold_v5'))
    p.add_argument('--out', type=Path, default=Path('figures'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.results/f'manifold_paths_{TAG}.npz')
    stats = json.loads((args.results/f'manifold_summary_{TAG}.json').read_text())
    times, paths = data['times'], data['sphere']
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(args.out/f'{name}_{TAG}.{suffix}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    fig, ax = sphere_axes(r'Rough nonlinear $S^2$: actual numerical trajectories')
    exact = data['dense_sphere']
    ax.plot(*exact.T, linewidth=2.2, label='Exact reference')
    for m, label, ls, mark in [(4, 'Deterministic CF midpoint', '--', 's'),
                               (0, 'Single-node RLCF2', '-.', 'o'),
                               (2, 'Paired RLCF2', ':', '^'),
                               (5, 'Ambient random RK2 (no projection)', '--', 'x')]:
        ax.plot(*paths[0, :, m].T, linestyle=ls, marker=mark, markersize=3,
                markevery=max(1, len(times)//10), linewidth=1.4, label=label)
    ax.scatter(*exact[0], marker='o', s=40)
    ax.text(*exact[0], '  start', fontsize=9)
    ax.text(*exact[-1], '  exact end', fontsize=9)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    save(fig, 'sphere_manifold_trajectories')

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for m, label in [(0, 'RLCF2'), (1, 'RRKMK2'), (2, 'Paired RLCF2'),
                      (3, 'Paired RRKMK2'), (4, 'CF midpoint'), (5, 'Ambient random RK2')]:
        drift = np.max(np.abs(np.linalg.norm(paths[:, :, m], axis=-1)-1), axis=0)
        ax.semilogy(times, np.maximum(drift, 1e-18), label=label)
    ax.set(xlabel='Time', ylabel=r'$\max_j\,|\|Y_n^{(j)}\|_2-1|$',
           title='Constraint error; zero values displayed at $10^{-18}$')
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8, ncol=2)
    save(fig, 'sphere_constraint_drift')

    # Intrinsic logarithm at the fixed exact endpoint, only for manifold methods.
    end = data['reference_sphere'][-1]
    basis1 = np.cross(end, np.eye(3)[np.argmin(abs(end))])
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(end, basis1)
    def log_coords(ys):
        dot = ys@end
        tangent = ys-dot[:, None]*end
        norm = np.linalg.norm(tangent, axis=1)
        angle = np.arctan2(norm, dot)
        if np.any(angle >= np.pi-1e-8):
            raise ValueError('Endpoint plot meets the logarithm cut locus.')
        factor = np.divide(angle, norm, out=np.ones_like(norm), where=norm>1e-15)
        return tangent@np.column_stack((basis1, basis2))*factor[:, None]
    fig, ax = plt.subplots(figsize=(7.1, 5.6))
    for m, label, marker in [(0, 'Single-node RLCF2', 'o'), (2, 'Paired RLCF2', '^')]:
        coords = log_coords(paths[:, -1, m])
        ax.scatter(coords[:, 0], coords[:, 1], marker=marker, s=14, alpha=0.4, label=label)
    point = log_coords(paths[:1, -1, 4])[0]
    ax.scatter(*point, marker='s', s=70, label='Deterministic CF midpoint')
    ax.scatter(0, 0, marker='*', s=120, label='Exact endpoint')
    ax.set(xlabel='First tangent coordinate (radians)', ylabel='Second tangent coordinate (radians)',
           title='Endpoint distribution in the exact-endpoint logarithm chart')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    save(fig, 'sphere_endpoint_distribution')

    fig, ax = sphere_axes(r'$SO(3)$ through frame axes: $R(t)e_1,R(t)e_2,R(t)e_3$')
    exactR, computedR = data['dense_rotations'], data['rotations'][:, 2]
    for j in range(3):
        ax.plot(*exactR[:, :, j].T, linewidth=2, label=f'Exact axis {j+1}')
        ax.plot(*computedR[:, :, j].T, linestyle='--', marker='o', markersize=3,
                markevery=max(1, len(times)//8), linewidth=1.2, label=f'Paired axis {j+1}')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    save(fig, 'so3_frame_trajectories')

    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    rows = stats['constant_rotation_exponential_test']
    for method, marker in [('Rodrigues', 'o'), ('Cayley', 's'), ('Taylor1', '^'), ('Taylor2', 'D')]:
        rs = [r for r in rows if r['method'] == method]
        ax.loglog([r['N'] for r in rs], [r['terminal_error_F'] for r in rs],
                  marker=marker, label=method)
    ax.set(xlabel='Number of steps N', ylabel=r'$\|\widetilde R_N-\exp(A)\|_F$',
           title='Constant rotation: exponential-evaluation error only')
    ax.grid(True, alpha=0.25); ax.legend()
    save(fig, 'exponential_global_error')

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    rows = stats['paired_exponential_backends']
    n = np.array([r['N'] for r in rows])
    for key, label, marker in [('Rodrigues_grid_rms', 'Paired RLCF2: Rodrigues', 'o'),
                                ('Cayley_grid_rms', 'Paired RLCF2: Cayley', 's'),
                                ('coupled_grid_rms', 'Common-node trajectory difference', '^')]:
        ax.loglog(n, [r[key] for r in rows], marker=marker, label=label)
    ax.set(xlabel='Number of steps N', ylabel='Grid-maximum geodesic RMS error',
           title=r'Rough nonlinear $S^2$: group-preserving exponential approximation')
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
    save(fig, 'cayley_coupled_comparison')
    print(f'Six PNG/PDF figures saved to {args.out}')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Redraw saved v4/v5 results with distinct dashes, markers and long legend keys.

No simulation is run and no trajectory is projected or displaced for visibility.
Inputs retain their original version tags; all output filenames use the v6 tag.
Each plot is a separate figure. Matplotlib's default colour cycle is unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TAG = 'v6_YW_28102026'
V4, V5 = 'v4_YW_01102026', 'v5_YW_27102026'
# Distinct encodings do not require colour to identify the methods.
STYLE = {
    'CF_midpoint': ('--', 's'),
    'RLCF2': ('-.', 'o'),
    'paired_RLCF2': (':', '^'),
    'RRKMK2': ('-', 'D'),
    'paired_RRKMK2': ((0, (7, 2, 1, 2, 1, 2)), 'v'),
    'ambient_RK2': ((0, (2, 1)), 'x'),
    'RKMK4': ((0, (8, 2)), 'D'),
}
METHODS = [
    (0, 'RLCF2', 'Single-node RLCF2'),
    (1, 'RRKMK2', 'Single-node RRKMK2'),
    (2, 'paired_RLCF2', 'Paired RLCF2'),
    (3, 'paired_RRKMK2', 'Paired RRKMK2'),
    (4, 'CF_midpoint', 'Deterministic CF midpoint'),
    (5, 'ambient_RK2', 'Ambient random RK2 (unprojected)'),
]


def style(key, *, length=None, offset=0):
    ls, marker = STYLE[key]
    opts = dict(linestyle=ls, marker=marker, linewidth=1.75,
                markersize=6, fillstyle='none', markeredgewidth=1.25)
    if length is not None:
        stride = max(1, length // 9)
        opts['markevery'] = (offset % stride, stride)
    return opts


def axes(title, ylabel, xlabel='Number of time steps N'):
    fig, ax = plt.subplots(figsize=(8.1, 5.2))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=.2)
    return fig, ax


def legend(ax, **kwargs):
    opts = dict(fontsize=9, handlelength=4.8, handletextpad=.7, numpoints=2)
    opts.update(kwargs)
    return ax.legend(**opts)


def sphere_axes(title):
    fig = plt.figure(figsize=(8.6, 8))
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 15))
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                      linewidth=.3, alpha=.11)
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), zlim=(-1.08, 1.08),
           xlabel='$x_1$', ylabel='$x_2$', zlabel='$x_3$', title=title)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35)
    return fig, ax


def save(fig, name, out):
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}_{TAG}.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def manifold_plots(folder, out):
    paths_file = folder / f'manifold_paths_{V5}.npz'
    stats_file = folder / f'manifold_summary_{V5}.json'
    with np.load(paths_file, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    stats = json.loads(stats_file.read_text())
    t, paths = data['times'], data['sphere']
    if paths.ndim != 4 or paths.shape[1:] != (len(t), 6, 3):
        raise ValueError('Unexpected sphere array shape; expected (paths, times, 6, 3).')
    if not np.isfinite(paths).all():
        raise ValueError('Non-finite trajectory values.')
    selected = [METHODS[i] for i in (4, 0, 2, 5)]

    fig, ax = sphere_axes(r'Rough nonlinear $S^2$: actual numerical trajectories')
    exact = data['dense_sphere']
    ax.plot(*exact.T, linewidth=2.4, linestyle='-', label='Exact reference')
    for order, (m, key, label) in enumerate(selected):
        ax.plot(*paths[0, :, m].T, label=label,
                **style(key, length=len(t), offset=order))
    ax.text(*exact[0], '  start', fontsize=9)
    ax.text(*exact[-1], '  exact end', fontsize=9)
    legend(ax, loc='lower center', bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
    save(fig, 'sphere_manifold_trajectories', out)

    fig, ax = axes('Constraint error; zeros displayed at $10^{-18}$',
                   r'$\max_j\,|\|Y_n^{(j)}\|_2-1|$', 'Time')
    for m, key, label in METHODS:
        drift = np.max(np.abs(np.linalg.norm(paths[:, :, m], axis=-1)-1), axis=0)
        ax.semilogy(t, np.maximum(drift, 1e-18), label=label,
                    **style(key, length=len(t), offset=m))
    legend(ax, ncol=2, fontsize=8)
    save(fig, 'sphere_constraint_drift', out)

    # Additional diagnostic: near-coincident spatial curves are resolved by their
    # errors, not by moving trajectories apart. Ambient error also covers RK2.
    fig, ax = axes('Same stored trajectories: error resolved in time',
                   r'$(\mathbb{E}\|Y_n-y(t_n)\|_2^2)^{1/2}$', 'Time')
    for order, (m, key, label) in enumerate(selected):
        error = np.linalg.norm(paths[:, :, m]-data['reference_sphere'][None, :, :], axis=-1)
        rms = np.sqrt(np.mean(error**2, axis=0))
        ax.plot(t, rms, label=label, **style(key, length=len(t), offset=order))
    legend(ax, fontsize=8)
    save(fig, 'sphere_error_vs_time', out)

    end = data['reference_sphere'][-1]
    e1 = np.cross(end, np.eye(3)[np.argmin(abs(end))]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(end, e1)
    def log_coords(ys):
        dot = ys@end; tangent = ys-dot[:, None]*end
        norm = np.linalg.norm(tangent, axis=1); angle = np.arctan2(norm, dot)
        if np.any(angle >= np.pi-1e-8):
            raise ValueError('Endpoint logarithm meets the cut locus.')
        factor = np.divide(angle, norm, out=np.ones_like(norm), where=norm>1e-15)
        return tangent@np.column_stack((e1, e2))*factor[:, None]
    fig, ax = axes('Endpoint distribution in the exact-endpoint logarithm chart',
                   'Second tangent coordinate (radians)', 'First tangent coordinate (radians)')
    for m, key, label in (METHODS[0], METHODS[2]):
        coords = log_coords(paths[:, -1, m])
        ax.plot(*coords.T, linestyle='None', marker=STYLE[key][1],
                markersize=4.5, fillstyle='none', alpha=.55, label=label)
    point = log_coords(paths[:1, -1, 4])[0]
    ax.plot(*point, linestyle='None', marker='s', markersize=10,
            fillstyle='none', label='Deterministic CF midpoint')
    ax.plot(0, 0, linestyle='None', marker='*', markersize=13, label='Exact endpoint')
    ax.set_aspect('equal', adjustable='datalim'); legend(ax, numpoints=1, fontsize=8)
    save(fig, 'sphere_endpoint_distribution', out)

    fig, ax = sphere_axes(r'$SO(3)$ frame axes: dashes encode axis, markers encode numerical path')
    exactR, computedR = data['dense_rotations'], data['rotations'][:, 2]
    for j, (ls, mark) in enumerate((('-', 'o'), ('--', 's'), ('-.', '^'))):
        ax.plot(*exactR[:, :, j].T, linewidth=2.2, linestyle=ls, label=f'Exact axis {j+1}')
        ax.plot(*computedR[:, :, j].T, linestyle=ls, marker=mark, fillstyle='none',
                markersize=6, markevery=(j, max(1, len(t)//8)), linewidth=1.2,
                label=f'Paired axis {j+1}')
    legend(ax, loc='lower center', bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
    save(fig, 'so3_frame_trajectories', out)

    fig, ax = axes('Constant rotation: exponential-evaluation error only',
                   r'$\|\widetilde R_N-\exp(A)\|_F$')
    for method, ls, mark in [('Rodrigues', '-', 'o'), ('Cayley', '--', 's'),
                              ('Taylor1', '-.', '^'), ('Taylor2', ':', 'D')]:
        rs = sorted([r for r in stats['constant_rotation_exponential_test']
                     if r['method'] == method], key=lambda r: r['N'])
        ax.loglog([r['N'] for r in rs], [r['terminal_error_F'] for r in rs],
                  linestyle=ls, marker=mark, linewidth=1.8, markersize=6,
                  fillstyle='none', label=method)
    legend(ax); save(fig, 'exponential_global_error', out)

    fig, ax = axes(r'Rough $S^2$: exponential implementation and coupled difference',
                   'Grid-maximum geodesic RMS error')
    rows = stats['paired_exponential_backends']; n = [r['N'] for r in rows]
    # Large hollow squares and small circles distinguish the near-coincident data.
    for key, label, ls, mark, size in [
        ('Rodrigues_grid_rms', 'Paired RLCF2: Rodrigues', '-', 'o', 4.5),
        ('Cayley_grid_rms', 'Paired RLCF2: Cayley', '--', 's', 9),
        ('coupled_grid_rms', 'Common-node trajectory difference', '-.', '^', 6)]:
        ax.loglog(n, [r[key] for r in rows], linestyle=ls, marker=mark,
                  fillstyle='none', markersize=size, linewidth=1.8, label=label)
    legend(ax, fontsize=8); save(fig, 'cayley_coupled_comparison', out)
    return [paths_file, stats_file]


def campaign_plots(folder, out):
    summary = folder / f'experiment_summary_all_{V4}.json'
    checks_path = folder / f'independent_checks_{V4}.json'
    data = json.loads(summary.read_text()); checks = json.loads(checks_path.read_text())
    cases = {c['id']: c for c in data['cases']}
    def rows(cid, method):
        return sorted([r for r in cases[cid]['rows'] if r['method'] == method], key=lambda r: r['N'])
    methods = [('CF_midpoint', 'Deterministic CF midpoint'), ('RKMK4', 'Deterministic RKMK4'),
               ('RLCF2', 'Single-node RLCF2'), ('paired_RLCF2', 'Paired RLCF2')]
    for cid, key, title, name in [
        ('rough_S2_a0.5_g1_k2', 'grid_maximum', r'Rough nonlinear $S^2$: $\alpha=0.5$', 'rough_sphere_convergence'),
        ('smooth_S2_a1_g1_k2', 'endpoint', r'Smooth nonlinear $S^2$ control', 'smooth_sphere_comparison')]:
        ylabel = 'RMS of grid-maximum geodesic error' if key == 'grid_maximum' else 'Terminal geodesic error / RMS error'
        fig, ax = axes(title, ylabel)
        for method, label in methods:
            rs = rows(cid, method); n = [r['N'] for r in rs]
            y = np.array([r[key]['rms'] for r in rs])
            if key == 'grid_maximum':
                lo = np.array([r[key]['ci95_low'] for r in rs]); hi = np.array([r[key]['ci95_high'] for r in rs])
                ax.errorbar(n, y, yerr=[y-lo, hi-y], capsize=2, label=label, **style(method))
            else:
                ax.plot(n, y, label=label, **style(method))
        ax.set_xscale('log', base=2); ax.set_yscale('log'); legend(ax, fontsize=8)
        save(fig, name, out)
    fig, ax = axes(r'Common-node trajectory comparison on $S^2$',
                   'RMS of maximum coupled RLCF2-RRKMK2 distance')
    for alpha, ls, mark in [(.25, '-', 'o'), (.5, '--', 's'), (.75, '-.', '^')]:
        rs = cases[f'rough_S2_a{alpha:g}_g1_k2']['coupled']
        ax.loglog([r['N'] for r in rs], [r['single']['rms'] for r in rs],
                  linestyle=ls, marker=mark, fillstyle='none', linewidth=1.8,
                  label=fr'$\alpha={alpha:g}$')
    legend(ax); save(fig, 'coupled_sphere_difference', out)
    rs = rows('smooth_SO3_a1_g1_k0', 'paired_RLCF2'); n = np.array([r['N'] for r in rs]); h = 1/n
    c = checks['smooth_linear_bias_variance_constants'][-1]
    fig, ax = axes(r'Paired RLCF2 on smooth $SO(3)$: bias and fluctuation', 'Frobenius error')
    for y, label, ls, mark in [
        ([r['matrix_mean_Frobenius'] for r in rs], 'Empirical matrix mean error', '-', 'o'),
        ([np.sqrt(r['matrix_centred_variance']) for r in rs], 'Empirical centred RMS', '-.', 's'),
        (c['B_Frobenius']*h**2, r'$\|\mathcal{B}_1\|_F h^2$', '--', None),
        (np.sqrt(c['V'])*h**2.5, r'$\sqrt{\mathcal{V}_1}\,h^{5/2}$', ':', None)]:
        ax.loglog(n, y, linestyle=ls, marker=mark, fillstyle='none', linewidth=1.8, label=label)
    legend(ax, fontsize=8); save(fig, 'smooth_matrix_bias_fluctuation', out)
    return [summary, checks_path]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=('all', 'manifold', 'campaign'), default='all')
    p.add_argument('--manifold-results', type=Path, default=Path('results/manifold_v5'))
    p.add_argument('--campaign-results', type=Path, default=Path('results'))
    p.add_argument('--out', type=Path, default=Path('figures/v6'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    inputs = []
    try:
        if a.scope in ('all', 'manifold'):
            inputs.extend(manifold_plots(a.manifold_results, a.out))
        if a.scope in ('all', 'campaign'):
            inputs.extend(campaign_plots(a.campaign_results, a.out))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        p.error(str(exc))
    manifest = {
        'version': TAG, 'scope': a.scope,
        'inputs': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        'simulation_rerun': False, 'trajectory_projection': False,
        'trajectory_selection': 'Stored path zero; all stored endpoints in scatter.',
        'extra_figure': 'sphere_error_vs_time uses RMS ambient errors of existing trajectories.',
        'encoding': 'Method-specific dash patterns and hollow markers; default colour cycle.',
    }
    (a.out/f'plot_provenance_{TAG}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Redraw complete: {len(list(a.out.glob("*"+TAG+".png")))} figures in {a.out}')


if __name__ == '__main__':
    main()

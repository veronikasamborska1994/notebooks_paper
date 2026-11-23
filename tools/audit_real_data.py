#!/usr/bin/env python3
"""Temporary audit script: runs the animal-permutation analyses, Fig 4g and Fig 1g
on the real data and prints every significance marker as numbers.

Usage (from the repo root):
    MPLBACKEND=Agg python tools/audit_real_data.py DATA_PATH OUT_DIR [section ...]
Sections: fig1g dec ext5a ext5b ext5cd ext9d diagA diagB (default: all)
"""
import os, sys, time, traceback, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())

DATA = sys.argv[1].rstrip('/') + '/'
OUT = sys.argv[2]
SECTIONS = sys.argv[3:] or ['fig1g', 'dec', 'ext5a', 'ext5b', 'ext5cd', 'ext9d', 'diagA', 'diagB']
os.makedirs(OUT, exist_ok=True)

from scipy import io
from functions import functions_behaviour as fun_beh
from functions import data_import as di_beh
from functions import functions_cells_regress as regs
from functions import functions_cells_rsa as rsas
from functions import functions_cells_decode as dec
from functions import functions_cells_svd as SVD
from functions import functions_cells_policy as pol


def dump_figs(prefix):
    """Save every open figure and print marker-only lines, vlines and error bars as numbers."""
    for k, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        fig.savefig(os.path.join(OUT, '%s_%d.png' % (prefix, k + 1)), dpi=150, bbox_inches='tight')
        for a, ax in enumerate(fig.axes):
            title = ax.get_title()
            for line in ax.get_lines():
                if line.get_marker() in ('.',) and line.get_linestyle() in ('None', 'none', '', ' '):
                    x = np.asarray(line.get_xdata())
                    print('  [%s fig%d ax%d "%s"] markers size=%s color=%s label=%s n=%d x=%s' % (
                        prefix, k + 1, a, title, line.get_markersize(), matplotlib.colors.to_hex(line.get_color()),
                        line.get_label(), len(x), np.array2string(x, max_line_width=1000)))
            for coll in ax.collections:
                if isinstance(coll, matplotlib.collections.LineCollection):
                    segs = coll.get_segments()
                    if 0 < len(segs) <= 12:
                        print('  [%s fig%d ax%d "%s"] line segments label=%s: %s' % (
                            prefix, k + 1, a, title, coll.get_label(),
                            '; '.join(np.array2string(np.round(s, 4), max_line_width=1000).replace('\n', '') for s in segs)))
    plt.close('all')


def section(name, func):
    if name not in SECTIONS:
        return
    print('\n===== %s =====' % name); sys.stdout.flush()
    t0 = time.time()
    try:
        func()
    except Exception:
        traceback.print_exc()
    plt.close('all')
    print('----- %s done in %.0f s' % (name, time.time() - t0)); sys.stdout.flush()


need_cells = any(s in SECTIONS for s in ['dec', 'ext5a', 'ext5b', 'ext5cd', 'ext9d'])
need_beh = any(s in SECTIONS for s in ['fig1g', 'diagA', 'diagB'])
if need_cells:
    HP = io.loadmat(DATA + 'data_recordings/HP.mat'); PFC = io.loadmat(DATA + 'data_recordings/PFC.mat')
if need_beh:
    exp_train = di_beh.Experiment(DATA + 'training/')


def s_fig1g():
    fun_beh.out_of_sequence(exp_train, first_task_only=False)
    fig = plt.figure(plt.get_fignums()[0])
    for cont in fig.axes[0].containers:
        if hasattr(cont, 'lines'):
            data_line, caps, bars = cont.lines
            segs = bars[0].get_segments()
            print('  Fig 1g mean per problem:', np.round(data_line.get_ydata(), 3))
            print('  Fig 1g error bar half-length per problem:', np.round([(s[1, 1] - s[0, 1]) / 2 for s in segs], 3))
    dump_figs('fig1g')


def s_dec():
    dec.plot(HP, PFC); dump_figs('fig4fg')


def s_ext5a():
    p_95, p_99, real_diff = regs.GLM_perm_across_regions(HP, PFC, perm=1, animal_perm=True)
    names = ['Choice', 'Reward', 'Reward x Choice', 'constant']
    print('  exact thresholds (2nd largest of 35 max-over-time values):', np.round(p_95 * 100, 3), ' p_99:', p_99)
    for i, n in enumerate(names[:3]):
        idx = np.where(real_diff[:, i] > p_95[i])[0]
        print('  Ext 5a %s: max real |PFC-CA1| = %.3f%%, threshold = %.3f%%, significant time bins (P = 1/35): %s' % (
            n, real_diff[:, i].max() * 100, p_95[i] * 100, idx))


def s_ext5b():
    rsas.RSA_across_time_plot(HP, PFC, perm_n=True, animal_perm=True, dlc=False); dump_figs('ext5b')


def s_ext5cd():
    pol.plot_correlations_slice(PFC, HP, n=11, c_1=1, perm_n=False, dlc=False, animal_perm=True); dump_figs('ext5d')
    pol.plot_diagonal_sums(HP, PFC, c_1=1, perm_n=True, dlc=False, animal_perm=True); dump_figs('ext5c')


def s_ext9d():
    SVD.plot_supplementary(HP, PFC, inds=np.arange(24, 35), animal_perm=True, n_perms=False); dump_figs('ext9d')


def s_diagA():
    """trials_till_reversal_plot: how often do the cross-session carry-over branches fire?"""
    n_sess = 0; n_mismatch = 0; n_mismatch_multi = 0; n_nocross = 0; n_nocross_forced = 0; n_prt0 = 0
    n_rev_without_cross = 0; diffs = []
    for subject_ID in exp_train.subject_IDs:
        for session in exp_train.get_sessions(subject_IDs=[subject_ID]):
            n_sess += 1
            block = session.trial_data['block']
            prt_raw = session.trial_data['pre-reversal trials']
            prt = (prt_raw > 0).astype(int)
            forced = session.trial_data['forced_trial']; fa = np.where(forced == 1)[0]
            block = np.delete(block, fa); prt = np.delete(prt, fa)
            cross = np.where((prt[1:] - prt[:-1]) == 1)[0]
            rev = np.where((block[1:] - block[:-1]) == 1)[0]
            if len(prt) and prt[0] > 0: n_prt0 += 1
            if len(rev) != len(cross):
                n_mismatch += 1
                diffs.append(len(cross) - len(rev))
                if len(cross) >= 2: n_mismatch_multi += 1
                if len(cross) == 0: n_rev_without_cross += 1
            elif len(cross) == 0:
                n_nocross += 1
                if len(fa) > 0: n_nocross_forced += 1
    print('  sessions: %d' % n_sess)
    print('  sessions with first trial already in pre-reversal countdown (prt[0] > 0): %d' % n_prt0)
    print('  sessions where #reversals != #threshold crossings: %d (of these with >= 2 crossings, where [i-1] != last crossing: %d; with 0 crossings, where [i-1] would fail or reuse a stale i: %d)' % (n_mismatch, n_mismatch_multi, n_rev_without_cross))
    print('  (#crossings - #reversals) values in those sessions:', dict(zip(*np.unique(diffs, return_counts=True))) if diffs else {})
    print('  sessions with no crossing and no reversal: %d (with forced trials, i.e. forced subtracted twice: %d)' % (n_nocross, n_nocross_forced))


def s_diagB():
    """meta_learning_reversals: where does the state change relative to the block-transition index b?"""
    same_b = 0; change_at_b1 = 0; change_at_b = 0; other = 0
    for subject_ID in exp_train.subject_IDs:
        for session in exp_train.get_sessions(subject_ID):
            block = session.trial_data['block']; forced = session.trial_data['forced_trial']
            state = session.trial_data['state']; fa = np.where(forced == 1)[0]
            block = np.delete(block, fa); state = np.delete(state, fa)
            for b in np.where((block[1:] - block[:-1]) == 1)[0]:
                if b < 1 or b + 1 >= len(state): continue
                if state[b - 1] == state[b] and state[b] != state[b + 1]: change_at_b1 += 1   # trial b is the last trial of the old state
                elif state[b - 1] != state[b]: change_at_b += 1                               # trial b is already the new state
                elif state[b] == state[b + 1]: same_b += 1
                else: other += 1
    print('  block transitions where state changes between trial b and b+1 (window index tr = last OLD-state trial): %d' % change_at_b1)
    print('  block transitions where state changes between trial b-1 and b (window index tr = first NEW-state trial): %d' % change_at_b)
    print('  no state change around b: %d, other: %d' % (same_b, other))


section('fig1g', s_fig1g)
section('dec', s_dec)
section('ext5a', s_ext5a)
section('ext5b', s_ext5b)
section('ext5cd', s_ext5cd)
section('ext9d', s_ext9d)
section('diagA', s_diagA)
section('diagB', s_diagB)
print('\nALL DONE')

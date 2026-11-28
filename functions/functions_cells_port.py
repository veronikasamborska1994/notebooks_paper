"""Port versus role decoding over time (post hoc analysis, not part of the published paper).

In the layout 2/3 pair the initiation port of layout 2 (I2) is the B port of layout 3 (B3). A linear decoder of
three states (initiation, A choice, B choice) is trained on one layout and tested on the other, with the A/B window
placed at different times after the choice. Three answers are scored, averaged over the two directions:
- same port:      B3 called I2 and I2 called B3 (same physical port, different role)
- same role:      B3 called B2 and I2 called I3 (same role, different port)
- different port: I3 called B2 and B2 called I3 (same confusion of roles, two different ports; I3 and B2 are
                  neighbouring ports in every session, so this is a neighbouring-port measure, not a control)
Chance is 33.3 %.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.svm import SVC
from functions.helper_functions import task_ind, animal_exp_permute, exact_perm_p

INIT_BINS = [24, 25]
CHOICE_STARTS = np.arange(34, 62, 2)        # start of the two-bin A/B window (choice 35, outcome 42)
WINDOW_REWARDED = np.arange(35, 42)         # choice to just before the outcome
N_TRIALS, N_TRIALS_REWARDED, N_DRAWS, SEED = 20, 12, 20, 0
MEASURES = ('same port', 'same role', 'different port')


def session_states(firing, DM, choice_bins, rewarded_only=False):
    """Mean rate per trial and neuron of initiation (I), A choice (A) and B choice (B) in layouts 2 and 3."""
    layout = task_ind(DM[:, 5], DM[:, 6], DM[:, 7]); firing = np.asarray(firing, dtype=float)
    chosen = DM[:, 2] == 1 if rewarded_only else np.ones(len(DM), dtype=bool)
    out = {}
    for k in (2, 3):
        out['I%d' % k] = firing[layout == k][:, :, INIT_BINS].mean(2)
        for name, choice in (('A', 1), ('B', 0)):
            out['%s%d' % (name, k)] = firing[(layout == k) & (DM[:, 1] == choice) & chosen][:, :, choice_bins].mean(2)
    return out


def decode_once(sessions, train, test, n, rng):
    """3 x 3 matrix (true I, A, B; decoded I, A, B) in percent, neurons of all sessions pooled, n random trials per state."""
    stack = lambda k: np.vstack([np.hstack([s[x + str(k)][rng.choice(len(s[x + str(k)]), n, replace=False)] for s in sessions]) for x in 'IAB'])
    X, Y = stack(train), stack(test); mean, sd = X.mean(0), X.std(0) + 1e-6; labels = np.repeat([0, 1, 2], n)
    pred = SVC(kernel='linear').fit((X - mean) / sd, labels).predict((Y - mean) / sd)
    return np.array([[np.mean(pred[labels == i] == j) for j in range(3)] for i in range(3)]) * 100


def animal_scores(dms, firing, choice_bins, n, rewarded_only=False, draws=N_DRAWS, seed=SEED):
    """(same port, same role, different port) of one animal and the number of sessions with >= n trials of every state."""
    sessions = [session_states(f, d, choice_bins, rewarded_only) for d, f in zip(dms, firing)]
    sessions = [s for s in sessions if min(len(v) for v in s.values()) >= n]
    rng = np.random.default_rng(seed)
    a, b = np.mean([[decode_once(sessions, 2, 3, n, rng), decode_once(sessions, 3, 2, n, rng)] for _ in range(draws)], 0)
    return np.array([(a[2, 0] + b[0, 2]) / 2, (a[2, 2] + b[0, 0]) / 2, (a[0, 2] + b[2, 0]) / 2]), len(sessions)


def region_difference_p(scores):
    """Exact one-sided P (CA1 > PFC, 35 splits of 7 animals). scores: (bins, 7), CA1 animals first.

    Returns P per bin and the P of the maximum over bins (max-statistic).
    """
    scores = np.atleast_2d(scores)
    diff = np.array([scores[:, list(c)].mean(1) - scores[:, [i for i in range(7) if i not in c]].mean(1) for c in combinations(range(7), 3)])
    per_bin = np.array([exact_perm_p(diff[0, k], diff[:, k])[0] for k in range(diff.shape[1])])
    return per_bin, exact_perm_p(diff[0].max(), diff.max(1))[0]


def port_over_time(HP, PFC, draws=N_DRAWS):
    """Scores (measure, bin, animal) over time and for the choice-to-outcome window of rewarded trials."""
    dms, firing = animal_exp_permute(HP, PFC)
    time = np.array([[animal_scores(dms[a], firing[a], [t, t + 1], N_TRIALS, draws=draws)[0] for a in range(7)] for t in CHOICE_STARTS])
    window = [animal_scores(dms[a], firing[a], WINDOW_REWARDED, N_TRIALS_REWARDED, rewarded_only=True, draws=draws) for a in range(7)]
    n_all = [animal_scores(dms[a], firing[a], [34, 35], N_TRIALS, draws=1)[1] for a in range(7)]
    return np.transpose(time, [2, 0, 1]), np.array([w[0] for w in window]).T, n_all, [w[1] for w in window]


def plot_port_over_time(HP, PFC, draws=N_DRAWS):
    """Figure and printed statistics of port_over_time (post hoc, not in the published paper)."""
    time, window, n_all, n_rew = port_over_time(HP, PFC, draws)
    print('Sessions used: all trials %s (n = %d trials per state), rewarded %s (n = %d)' % (n_all, N_TRIALS, n_rew, N_TRIALS_REWARDED))
    colours = sns.color_palette('Set2'); fig, axs = plt.subplots(1, 4, figsize=(21, 4.8), gridspec_kw={'width_ratios': [1, 1, 1, .9]})
    for k, (ax, name) in enumerate(zip(axs, MEASURES)):
        for who, rows, c in (('PFC', slice(3, 7), colours[0]), ('CA1', slice(0, 3), colours[1])):
            v = time[k][:, rows]; m = v.mean(1); e = v.std(1) / np.sqrt(v.shape[1])
            ax.plot(CHOICE_STARTS, m, color=c, label=who); ax.fill_between(CHOICE_STARTS, m - e, m + e, color=c, alpha=.25)
        p, pmax = region_difference_p(time[k])
        for j, t in enumerate(CHOICE_STARTS):
            print('%s, bins %d-%d: CA1 %.1f PFC %.1f, CA1 > PFC P = %.3f' % (name, t, t + 1, time[k][j, :3].mean(), time[k][j, 3:].mean(), p[j]))
        print('%s: max-statistic over bins, CA1 > PFC P = %.3f' % (name, pmax))
        ax.axhline(100 / 3, color='grey', ls='--', lw=.8); ax.set_ylim(0, 100); ax.set_title(name if k < 2 else 'neighbouring port')
        ax.set_xticks([35, 42, 49, 60]); ax.set_xticklabels(['Ch', 'R', '+0.5', '']); ax.set_xlabel('Time of the A/B window')
    print('same role: max-statistic over bins, PFC > CA1 P = %.3f' % region_difference_p(-time[1])[1])
    diff = time[0] - time[2]; p, pmax = region_difference_p(diff)
    print('same port minus different port: max-statistic over bins, CA1 > PFC P = %.3f; mean over bins CA1 %.1f PFC %.1f' % (pmax, diff[:, :3].mean(), diff[:, 3:].mean()))
    axs[0].set_ylabel('Decoding (%)'); axs[0].legend(frameon=False)
    ax = axs[3]
    for k, name in enumerate(MEASURES):
        v = window[k]; pw = region_difference_p(v[None])[0][0]
        print('rewarded, bins %d-%d, %s: CA1 %.1f %s PFC %.1f %s, CA1 > PFC P = %.3f' % (WINDOW_REWARDED[0], WINDOW_REWARDED[-1], name, v[:3].mean(), np.round(v[:3]).astype(int).tolist(), v[3:].mean(), np.round(v[3:]).astype(int).tolist(), pw))
        for j, (rows, c) in enumerate(((slice(3, 7), colours[0]), (slice(0, 3), colours[1]))):
            x = k + (j - .5) * .36; ax.bar(x, v[rows].mean(), .34, color=c, alpha=.5); ax.plot(np.full(len(v[rows]), x), v[rows], 'o', color=c, mec='k', mew=.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(['same\nport', 'same\nrole', 'different\nport']); ax.axhline(100 / 3, color='grey', ls='--', lw=.8); ax.set_ylim(0, 100)
    ax.set_title('rewarded, choice to outcome'); sns.despine(); plt.tight_layout()
    return fig

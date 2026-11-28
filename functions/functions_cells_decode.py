"""SVM decoding of trial states across task layouts (Figure 4F-G).

Trains an SVM (RBF kernel) to classify trial position (initiation, A/B choice,
A/B reward, A/B no-reward) using neural population activity from one task
layout, then tests cross-task generalisation on a different layout.
Animal-shuffle permutation tests compare HP vs PFC.

Key functions:
- decoder_trial_states: train/test SVM decoder for one task pair
- permute_subjects: run all animal permutations
- decoding_statistics: exact animal-permutation P values for the Fig 4G bars
- plot: main plotting function
"""

import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations 
import pandas as pd
import palettable 
from scipy import io
from functions.helper_functions import task_ind, animal_exp_permute, exact_perm_p, HP_SESSIONS_PER_ANIMAL, PFC_SESSIONS_PER_ANIMAL

def import_cell_data(path):
    
    'Import neural data.'
    HP = io.loadmat(path +'/data_recordings/HP.mat')
    PFC = io.loadmat(path + '/data_recordings/PFC.mat')
    
    'Import neural data after accounting for physical movements.'
    HP_dlc = io.loadmat(path +'/data_recordings/after PCA /HP_dlc_pca.mat')
    PFC_dlc = io.loadmat(path + '/data_recordings/after PCA /PFC_dlc_pca.mat')
    
    return HP, PFC, HP_dlc, PFC_dlc


STATES = ['Initiation', 'A choice', 'A reward', 'A no-reward', 'B choice', 'B reward', 'B no-reward']
TASK_PAIRS = {1: (1, 2), 2: (1, 3), 3: (2, 3)}   # task_test -> (task x, task y)

# Confusion-matrix cells (true row, predicted column) averaged for the Port bar, for training on
# task 2 / testing on task 3 and for training on task 3 / testing on task 2.
# In all 81 recorded sessions the initiation port of task 2 (DM column 8) is the B port of task 3,
# and the initiation port of task 3 is never the B port of task 2. The cells in which the true and
# the predicted state share a physical port are therefore B choice -> Initiation when trained on
# task 2 and tested on task 3, and Initiation -> B choice for the reverse.
PORT_CELLS_SAME_PORT = ((4, 0), (0, 4))
# Cells used for the printed figure: they compare two different physical ports.
PORT_CELLS_PUBLISHED = ((0, 4), (4, 0))

MIN_TRIALS = 10         # a session is used if every choice x outcome x task has more trials than this
N_TRIALS = MIN_TRIALS + 1   # trials per state: the number every session that is used is guaranteed to have
N_DRAWS, SEED = 100, 0      # random draws of N_TRIALS trials per state; confusion matrices are averaged over draws
BIN_INIT, BIN_CHOICE, BIN_REWARD = 24, 35, 42


def _split_by_animal(data, splits):
    """Split a {'DM', 'Data'} dict into per-animal lists (last animal is the remainder)."""
    bounds = [0] + list(np.cumsum(splits)) + [None]
    return ([data['DM'][0][i:j] for i, j in zip(bounds[:-1], bounds[1:])],
            [data['Data'][0][i:j] for i, j in zip(bounds[:-1], bounds[1:])])


def _session_patterns(firing, DM, task_test, rng=None):
    """Return (task x, task y) matrices (samples x neurons, ordered as STATES) or None if too few trials.

    Each state contributes N_TRIALS trials drawn at random without replacement from all its trials in the
    layout (``rng``), so the result does not depend on which part of the layout the trials come from.
    The published code took the last MIN_TRIALS // 2 + 1 = 6 trials of each state.
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    taskid = task_ind(DM[:, 5], DM[:, 6], DM[:, 7])
    bins_init = [BIN_INIT - 1, BIN_INIT]
    bins_choice = [BIN_CHOICE - 1, BIN_CHOICE]
    bins_reward = [BIN_REWARD, BIN_REWARD + 1]

    def stack(trials, bins):
        trials = np.sort(rng.choice(np.asarray(trials), N_TRIALS, replace=False))
        fr = firing[trials][:, :, bins]
        return np.concatenate(np.transpose(fr, [0, 2, 1]), 0)

    patterns = []
    for task in TASK_PAIRS[task_test]:
        in_task = taskid == task
        blocks = [stack(np.where(in_task)[0], bins_init)]
        for choice in (1, 0):                               # A then B
            chosen = in_task & (DM[:, 1] == choice)
            rewarded = np.where(chosen & (DM[:, 2] == 1))[0]
            unrewarded = np.where(chosen & (DM[:, 2] == 0))[0]
            if len(rewarded) <= MIN_TRIALS or len(unrewarded) <= MIN_TRIALS:
                return None
            blocks += [stack(np.where(chosen)[0], bins_choice),
                       stack(rewarded, bins_reward), stack(unrewarded, bins_reward)]
        patterns.append(np.vstack(blocks))
    return patterns


def _fit_and_score(train, test, labels):
    """Row-normalised confusion matrix in percent (rows true, columns predicted)."""
    # RBF kernel with gamma='scale': the scikit-learn default, as used for the published figure
    pred = svm.SVC(kernel='rbf', gamma='scale').fit(train, labels).predict(test)
    cnf = metrics.confusion_matrix(labels, pred, labels=np.arange(len(STATES)))
    return cnf / cnf.sum(1, keepdims=True) * 100


def decoder_trial_states(data, task_test=0, H=0, perm=True, decoder_full=True, rng=None):
    """Decode position in the trial across two tasks (Fig 4F, G), for one random draw of trials.

    Returns two lists (one 7 x 7 matrix per animal): train on task x / test on y, and the reverse.
    ``decoder_full`` is kept for backward compatibility and is unused.
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    if perm:
        dms, firing = data
    else:
        dms, firing = _split_by_animal(data, HP_SESSIONS_PER_ANIMAL if H == 0 else PFC_SESSIONS_PER_ANIMAL)

    xy, yx = [], []
    for animal_dm, animal_fr in zip(dms, firing):
        sessions = [_session_patterns(f, d, task_test, rng) for f, d in zip(animal_fr, animal_dm)]
        sessions = [p for p in sessions if p is not None]
        task_x = np.concatenate([p[0] for p in sessions], 1)
        task_y = np.concatenate([p[1] for p in sessions], 1)
        labels = np.repeat(np.arange(len(STATES)), task_x.shape[0] // len(STATES))
        xy.append(_fit_and_score(task_x, task_y, labels))
        yx.append(_fit_and_score(task_y, task_x, labels))
    return xy, yx


def permute_subjects(HP, PFC, draws=N_DRAWS, seed=SEED):
    """Enumerate all 35 assignments of 7 animals to PFC (4) and CA1 (3) (Fig 4F-G).

    Decoding is done within animal and does not depend on the grouping, so each animal is
    decoded once and only the group averages are permuted. Each animal's confusion matrices are
    averaged over ``draws`` random draws of N_TRIALS trials per state.

    Returns
    -------
    perm_all_xy, perm_all_yx : ndarray (6, 35, 7, 7)
        First three rows PFC, last three CA1, task pairs (1,2), (1,3), (2,3).
    subj_hp_xy, subj_hp_yx, subj_pfc_xy, subj_pfc_yx : ndarray (3 pairs, animals, 7, 7)
    """
    data = animal_exp_permute(HP, PFC)                      # animals 0-2 CA1, 3-6 PFC
    rng = np.random.default_rng(seed)
    both = np.mean([[decoder_trial_states(data, task_test=c, perm=True, rng=rng) for c in TASK_PAIRS]
                    for _ in range(draws)], 0)
    xy, yx = both[:, 0], both[:, 1]                         # (pair, animal, 7, 7)
    perm_xy, perm_yx = [], []
    for m, out in ((xy, perm_xy), (yx, perm_yx)):
        groups = [(list(pfc), [i for i in range(7) if i not in pfc]) for pfc in combinations(range(7), 4)]
        for which in (0, 1):                                # PFC, then CA1
            for pair in range(3):
                out.append([m[pair, g[which]].mean(0) for g in groups])
    return (np.asarray(perm_xy), np.asarray(perm_yx),
            xy[:, :3], yx[:, :3], xy[:, 3:], yx[:, 3:])


def decoding_scores(m1, m2, same_port=True):
    """Per-animal Correct, Position and Port scores. m1, m2: (pair, animal, 7, 7)."""
    m1 = np.asarray(m1); m2 = np.asarray(m2)
    av = np.mean([m1, m2], 0).mean(0)
    (r1, c1), (r2, c2) = PORT_CELLS_SAME_PORT if same_port else PORT_CELLS_PUBLISHED
    return {'Correct': np.mean([av[:, r, c] for r, c in CORRECT_CELLS], 0),
            'Position in Trial Sequence': np.mean([av[:, r, c] for r, c in POSITION_CELLS], 0),
            'Port': np.mean([m1[2][:, r1, c1], m2[2][:, r2, c2]], 0)}


def animal_permutation_null(hp, pfc, direction):
    """Real difference and the 35-split null. direction: +1 PFC > CA1, -1 CA1 > PFC."""
    both = np.concatenate([hp, pfc]); n = len(both); k = len(hp)

    def diff(idx):
        rest = [i for i in range(n) if i not in idx]
        return direction * (both[rest].mean() - both[list(idx)].mean())

    return diff(tuple(range(k))), np.array([diff(c) for c in combinations(range(n), k)])


def decoding_statistics(hp1, hp2, pfc1, pfc2, same_port=True):
    """Scores, differences and exact P values for the three bars of Fig 4G.

    The null is computed from the same per-animal scores as the real difference.
    """
    hp = decoding_scores(hp1, hp2, same_port); pfc = decoding_scores(pfc1, pfc2, same_port)
    out = {}
    for name, sign in (('Correct', 1), ('Position in Trial Sequence', 1), ('Port', -1)):
        diff, null = animal_permutation_null(hp[name], pfc[name], sign)
        p, rank = exact_perm_p(diff, null)
        out[name] = {'hp': hp[name], 'pfc': pfc[name], 'diff': diff, 'p': p, 'rank': rank, 'null': null}
    return out


# Cells (true row, predicted column) of the averaged matrix used for the other two bars of Fig 4G
CORRECT_CELLS = ((4, 4), (5, 5), (6, 6))
POSITION_CELLS = ((4, 1), (5, 2), (6, 3))


def plot_matrices(hp1, hp2, pfc1, pfc2):
    """Confusion matrices that the bars of Fig 4G are computed from (Fig 4F).

    One row per region. Left: average over all layout pairs and both directions, with the cells of the
    Correct (blue) and Position (red) bars. Middle and right: the layout 2/3 pair in each direction, with the
    cell in which the true and the decoded state share a physical port (green); the Port bar is the mean of
    the two green cells. Rows are the true states, columns the decoded states, in percent.
    """
    cmap = palettable.scientific.sequential.Acton_3.mpl_colormap
    names = ['I%s', 'A%s choice', 'A%s reward', 'A%s no-reward', 'B%s choice', 'B%s reward', 'B%s no-reward']
    fig, axs = plt.subplots(2, 3, figsize=(19, 13), constrained_layout=True)
    for row, (region, m1, m2) in enumerate((('CA1', hp1, hp2), ('PFC', pfc1, pfc2))):
        m1 = np.asarray(m1); m2 = np.asarray(m2)
        panels = ((region + ': all layout pairs, both directions', np.mean([m1, m2], 0).mean((0, 1)), '', '',
                   [(c, 'deepskyblue') for c in CORRECT_CELLS] + [(c, 'red') for c in POSITION_CELLS]),
                  (region + ': trained on layout 2, tested on layout 3', m1[2].mean(0), 2, 3, [(PORT_CELLS_SAME_PORT[0], 'limegreen')]),
                  (region + ': trained on layout 3, tested on layout 2', m2[2].mean(0), 3, 2, [(PORT_CELLS_SAME_PORT[1], 'limegreen')]))
        for ax, (title, conf, train, test, boxes) in zip(axs[row], panels):
            img = ax.imshow(conf, cmap=cmap, vmin=0, vmax=100)
            for i in range(7):
                for j in range(7):
                    ax.text(j, i, '%.0f' % conf[i, j], ha='center', va='center', fontsize=9, color='black' if conf[i, j] > 60 else 'white')
            for (r, c), colour in boxes:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=colour, linewidth=3.5))
            ax.set_title(title)
            ax.set_xlabel('Predicted' + (' (layout %s states)' % train if train else ''))
            ax.set_ylabel('Actual' + (' (layout %s states)' % test if test else ''))
            ax.set_xticks(range(7)); ax.set_xticklabels([n % train for n in names], rotation='vertical')
            ax.set_yticks(range(7)); ax.set_yticklabels([n % test for n in names])
    fig.colorbar(img, ax=axs, label='Decoding (%)', shrink=0.6)
    return fig


def plot(HP, PFC, same_port=True, draws=N_DRAWS):
    """Plot cross-task decoding results (Fig 4F-G).

    Runs all permutations and plots:
    - The confusion matrices the bars are computed from (Fig 4F)
    - Bar plots comparing correct, sequential, and port decoding patterns
    - Statistical significance from animal-permutation tests

    Parameters
    ----------
    HP, PFC : dict
        Neural data dicts with keys 'DM' and 'Data'.
    same_port : bool
        True (default): Port cells that share a physical port, as defined in the paper.
        False: the cells used for the printed figure.
    """

    perm_all, perm_all_2, matrix_hp_1, matrix_hp_2, matrix_pfc_1, matrix_pfc_2 = permute_subjects(HP, PFC, draws=draws)
    stats = decoding_statistics(matrix_hp_1, matrix_hp_2, matrix_pfc_1, matrix_pfc_2, same_port)
    plot_matrices(matrix_hp_1, matrix_hp_2, matrix_pfc_1, matrix_pfc_2)

    subj_hp_gen, subj_hp_seq, subj_hp_b = [stats[k]['hp'] for k in stats]
    subj_pfc_gen, subj_pfc_seq, subj_pfc_b = [stats[k]['pfc'] for k in stats]
    sem = lambda v: np.std(v)/np.sqrt(len(v))
    gen_hp_std, sequence_hp_std, std_hp_b = map(sem, (subj_hp_gen, subj_hp_seq, subj_hp_b))
    gen_pfc_std, sequence_pfc_std, std_pfc_b = map(sem, (subj_pfc_gen, subj_pfc_seq, subj_pfc_b))
    labels = list(stats)
    for name in labels:
        st = stats[name]
        print('%s decoding %s (exact p = %d/%d = %.3f)' % (name, 'p < 0.05' if st['p'] <= 0.05 else 'NS',
                                                           st['rank'], len(st['null']), st['p']))

    # plot Correct, Position in Sequence and Port decoding patterns 
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    cl = sns.color_palette("Set2")
    fig, ax = plt.subplots()
    ax.bar(x - width/2, [np.mean(subj_pfc_gen), np.mean(subj_pfc_seq),np.mean(subj_pfc_b)], width, label = 'PFC', color = cl[0], alpha = 0.5,\
                    yerr = [gen_pfc_std, sequence_pfc_std, std_pfc_b])
    
    ax.bar(x + width/2, [np.mean(subj_hp_gen), np.mean(subj_hp_seq), np.mean(subj_hp_b)], width, label = 'CA1', color = cl[1], alpha = 0.5,\
                    yerr = [gen_hp_std, sequence_hp_std, std_hp_b])

    ax.set_xticks(x);  ax.set_xticklabels(labels); ax.legend()
    
    # plot subjects
    hp_str = np.hstack([np.tile('Correct',4), np.tile('Position \n in Trial Sequence',4),np.tile('Port',4), np.tile('Correct',3), np.tile('Position \n in Trial Sequence',3), np.tile('Port',3),])
    area = np.hstack([np.tile('PFC',12),np.tile('CA1',9)]) 
    decode = np.hstack([subj_pfc_gen, subj_pfc_seq, subj_pfc_b,subj_hp_gen, subj_hp_seq, subj_hp_b])
    data_ = {'Decoding': decode, 'Label': hp_str, 'Area':area}
    df = pd.DataFrame(data = data_)
    sns.swarmplot(x = "Label", y = "Decoding", hue = 'Area', palette="Set2", dodge = True, data = df)
    # one legend (bars: PFC, CA1), outside the axes so it covers no data points (the swarm legend repeated it as "HP")
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[:2], names[:2], title = 'Area', frameon = False, loc = 'upper left', bbox_to_anchor = (1, 1))
    sns.despine()

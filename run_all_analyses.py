#!/usr/bin/env python3
"""Run all analyses and save figures as SVG + PNG.

Usage:
    python run_all_analyses.py [--perms N] [--data-path PATH] [--output-dir DIR] [--seed N]

Defaults: 5000 permutations, random seed 0, figures saved to ./figures/
The random number generator is reset to the seed before every step, so each figure is reproduced exactly.
"""

import argparse
import gc
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Run all paper analyses')
parser.add_argument('--perms', type=int, default=5000, help='Number of permutations (default: 5000)')
parser.add_argument('--data-path', type=str,
                    default='/Users/veronikasamborska/Desktop/paper_nn_code_data/data_share/',
                    help='Path to data folder')
parser.add_argument('--output-dir', type=str, default='figures', help='Output directory for figures')
parser.add_argument('--seed', type=int, default=0, help='Random seed, reset before every step (default: 0)')
args = parser.parse_args()

PERMS = args.perms
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir
SEED = args.seed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: save all new figures created by a function call
# ---------------------------------------------------------------------------
def save_figures(prefix, figs_before):
    """Save any new figures that appeared since figs_before snapshot."""
    new_figs = sorted(set(plt.get_fignums()) - figs_before)
    saved = []
    for i, fig_num in enumerate(new_figs):
        fig = plt.figure(fig_num)
        suffix = f"_{i+1}" if len(new_figs) > 1 else ""
        base = os.path.join(OUTPUT_DIR, f"{prefix}{suffix}")
        fig.savefig(f"{base}.svg", format='svg', bbox_inches='tight')
        fig.savefig(f"{base}.png", format='png', dpi=300, bbox_inches='tight')
        saved.append(base)
    plt.close('all')
    return saved


def run_step(description, prefix, func, *a, **kw):
    """Run an analysis step, save figures, print timing."""
    figs_before = set(plt.get_fignums())
    np.random.seed(SEED)  # same seed at the start of every step: each figure can be reproduced on its own
    t0 = time.time()
    func(*a, **kw)
    elapsed = time.time() - t0
    saved = save_figures(prefix, figs_before)
    n = len(saved)
    tqdm.write(f"  -> {n} figure(s) saved as {prefix}*.svg/png  [{elapsed:.1f}s]")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
print("Importing analysis modules...")
from functions import functions_behaviour as fun_beh
from functions import functions_behaviour_rec as fun_beh_rec
from functions import data_import_ephys as dimp
from functions import data_import as di_beh
from functions import functions_cells_regress as regs
from functions import functions_cells_rsa as rsas
from functions import functions_cells_decode as dec
from functions import functions_cells_port as port
from functions import functions_cells_svd as SVD
from functions import functions_cells_policy as pol
from functions import functions_cells_ext as ext
from functions import plot_neurons as plot_n
from scipy import io

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Loading data from {DATA_PATH} ...")

# Behavioural data
exp_train = di_beh.Experiment(DATA_PATH + 'training/')
animals = ['/m484', '/m479', '/m483', '/m478', '/m486', '/m480', '/m481']
exp_rec = []
for an in animals:
    exp_rec.append(dimp.Experiment(DATA_PATH + 'data_recordings/behaviour' + an))

# Neural data
HP = io.loadmat(DATA_PATH + '/data_recordings/HP.mat')
PFC = io.loadmat(DATA_PATH + '/data_recordings/PFC.mat')
HP_dlc = io.loadmat(DATA_PATH + '/data_recordings/after PCA /HP_dlc_pca.mat')
PFC_dlc = io.loadmat(DATA_PATH + '/data_recordings/after PCA /PFC_dlc_pca.mat')

print("Data loaded.\n")

# ---------------------------------------------------------------------------
# Define all analysis steps
# ---------------------------------------------------------------------------
steps = [
    # ---- BEHAVIOUR (Main Figures) ----
    ("Fig 1D  | Last training session",
     "fig1d_session_plot",
     lambda: fun_beh.session_plot_moving_average(exp_train.sessions[-1])),

    ("Fig 1E  | Trials to reversal",
     "fig1e_trials_reversal",
     lambda: fun_beh.trials_till_reversal_plot(exp_train)),

    ("Fig 1F  | Meta-learning reversals",
     "fig1f_meta_learning",
     lambda: fun_beh.meta_learning_reversals(exp_train)),

    ("Fig 1G-H | Out of sequence pokes",
     "fig1gh_out_of_sequence",
     lambda: fun_beh.out_of_sequence(exp_train, first_task_only=False)),

    ("Fig 1I-J | Policy during training",
     "fig1ij_policy_training",
     lambda: fun_beh.plot_policy_during_training(exp_train)),

    # ---- BEHAVIOUR (Extended Data) ----
    ("Ext Fig 2A-C | Reversals during recording",
     "ext2ac_reversals_rec",
     lambda: fun_beh_rec.trials_till_reversal_plot_recordings(exp_rec)),

    ("Ext Fig 2D-F | Out of sequence (A/B pokes only)",
     "ext2df_oos_rec",
     lambda: fun_beh_rec.out_of_sequence_recordings(exp_rec, only_A_B=True)),

    ("Ext Fig 2D-G | Out of sequence (all pokes)",
     "ext2dg_oos_all",
     lambda: fun_beh_rec.out_of_sequence_recordings(exp_rec, only_A_B=False)),

    ("Ext Fig 2I-J | Forced choice trials",
     "ext2ij_forced",
     lambda: fun_beh_rec.plot_forced(exp_rec)),

    # ---- NEURAL: Main Figures ----
    ("Fig 2    | Example neuron",
     "fig2_neuron",
     lambda: plot_n.plot_neuron(PFC, session_id=0, neuron_ID=0)),

    ("Fig 4B   | GLM CPDs (session perms)",
     "fig4b_glm",
     lambda: regs.plot_simple_GLM(HP, PFC, n_perm=PERMS, animal_perm=False)),

    ("Fig 4C-E | RSA across time",
     "fig4ce_rsa",
     lambda: rsas.RSA_across_time_plot(HP, PFC, perm_n=PERMS, dlc=False)),

    ("Fig 4F-G | Decoding",
     "fig4fg_decode",
     lambda: dec.plot(HP, PFC)),

    ("Post hoc  | Port vs role decoding over time (not in the paper)",
     "posthoc_port_over_time",
     lambda: port.plot_port_over_time(HP, PFC)),

    ("Fig 5A-C | SVD example patterns",
     "fig5ac_svd_examples",
     lambda: SVD.plot_example_patterns(PFC)),

    ("Fig 5D-E | SVD between brains",
     "fig5de_svd_brains",
     lambda: SVD.svd_between_brains(HP_dlc, PFC_dlc)),

    ("Fig 5D-H | SVD main figure (session perms)",
     "fig5dh_svd_main",
     lambda: SVD.plot_main_figure(HP, PFC, HP_dlc, PFC_dlc, n_perms=PERMS)),

    ("Fig 6A   | Policy during recordings",
     "fig6a_policy_rec",
     lambda: pol.plot_policy_during_recordings(PFC, HP, n=12)),

    ("Fig 6B   | Policy GLM",
     "fig6b_policy_glm",
     lambda: pol.plot_policy_simple_GLM(HP, PFC, n_perm=PERMS, n=11, dlc=False)),

    ("Fig 6C   | Policy correlations",
     "fig6c_policy_corr",
     lambda: pol.plot_correlations_slice(PFC, HP, n=11, c_1=1, perm_n=PERMS, dlc=False, animal_perm=False)),

    ("Fig 6D   | Policy diagonal sums",
     "fig6d_policy_diag",
     lambda: pol.plot_diagonal_sums(HP, PFC, perm_n=PERMS, dlc=False)),

    # ---- NEURAL: Extended Data ----
    ("Ext Fig 4 | Init-B remapping",
     "ext4_remapping",
     lambda: ext.plot_Init_B_remapping(PFC, HP)),

    ("Ext Fig 5A | GLM CPDs (animal perms)",
     "ext5a_glm_animal",
     lambda: regs.plot_simple_GLM(HP, PFC, n_perm=PERMS, animal_perm=True)),

    ("Ext Fig 5B | RSA (animal perms)",
     "ext5b_rsa_animal",
     lambda: rsas.RSA_across_time_plot(HP, PFC, perm_n=PERMS, animal_perm=True, dlc=False)),

    ("Ext Fig 5C-D | Policy (animal perms)",
     "ext5cd_policy_animal",
     lambda: (pol.plot_correlations_slice(PFC, HP, n=11, c_1=1, perm_n=PERMS, dlc=False, animal_perm=True),
              pol.plot_diagonal_sums(HP, PFC, c_1=1, perm_n=PERMS, dlc=False, animal_perm=True))),

    ("Ext Fig 7C-E | RSA (DLC residuals)",
     "ext7_dlc_rsa",
     lambda: rsas.RSA_across_time_plot(HP_dlc, PFC_dlc, perm_n=PERMS, dlc=True)),

    ("Ext Fig 7 | Policy GLM (DLC residuals)",
     "ext7_dlc_policy_glm",
     lambda: pol.plot_policy_simple_GLM(HP_dlc, PFC_dlc, n_perm=PERMS, n=11, dlc=True)),

    ("Ext Fig 7 | Policy correlations (DLC residuals)",
     "ext7_dlc_policy_corr",
     lambda: pol.plot_correlations_slice(PFC_dlc, HP_dlc, n=11, c_1=1, perm_n=PERMS, dlc=True, animal_perm=False)),

    ("Ext Fig 7 | Policy diagonal sums (DLC residuals)",
     "ext7_dlc_diagonal",
     lambda: pol.plot_diagonal_sums(HP_dlc, PFC_dlc, perm_n=PERMS, dlc=True)),

    ("Ext Fig 8  | Surprise plot",
     "ext8_surprise",
     lambda: ext.surprise_plot(HP, PFC)),

    ("Ext Fig 9A | SVD all timepoints",
     "ext9a_svd_all",
     lambda: SVD.plot_supplementary(HP, PFC, inds=np.arange(63), animal_perm=False, n_perms=PERMS)),

    ("Ext Fig 9B | SVD init-choice",
     "ext9b_svd_init_choice",
     lambda: SVD.plot_supplementary(HP, PFC, inds=np.arange(24, 35), animal_perm=False, n_perms=PERMS)),

    ("Ext Fig 9C | SVD init-choice DLC",
     "ext9c_svd_dlc",
     lambda: SVD.plot_supplementary(HP_dlc, PFC_dlc, inds=np.arange(24, 39), animal_perm=False, n_perms=PERMS)),

    ("Ext Fig 9D | SVD animal perms",
     "ext9d_svd_animal",
     lambda: SVD.plot_supplementary(HP, PFC, inds=np.arange(24, 35), animal_perm=True, n_perms=PERMS)),
]

# ---------------------------------------------------------------------------
# Run all steps with progress bar
# ---------------------------------------------------------------------------
print(f"Running {len(steps)} analyses with {PERMS} permutations...\n")

failed = []
pbar = tqdm(steps, desc="Analyses", unit="step", ncols=100)
for desc, prefix, func in pbar:
    pbar.set_description(desc)
    try:
        run_step(desc, prefix, func)
    except Exception as e:
        tqdm.write(f"  !! ERROR in {desc}: {e}")
        failed.append(desc)
        plt.close('all')
    gc.collect()

if failed:
    print(f"\n{len(failed)} of {len(steps)} steps failed:")
    for desc in failed:
        print(f"  {desc}")
    sys.exit(1)
print(f"\nDone! All figures saved to {OUTPUT_DIR}/")

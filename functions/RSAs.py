"""Model matrices for the RSA analysis (named RDMs in the code, as in the paper). They are similarity matrices:
True means the two conditions are in the same group. The CPDs do not depend on the naming.

Each function returns a boolean matrix of shape (15, 15) encoding which
trial-type pairs should be grouped together under a given hypothesis.

The 15 trial types are ordered as:
  [A_rew_T1, A_nrew_T1, A_rew_T2, A_nrew_T2, A_rew_T3, A_nrew_T3,
   Init_T1, Init_T3, Init_T2,
   B_rew_T3, B_nrew_T3, B_rew_T2, B_nrew_T2, B_rew_T1, B_nrew_T1]
"""

import numpy as np


def _zeros_15x15(n_rows):
    """Create an (n_rows, 15) boolean array initialised to False."""
    return np.zeros((n_rows, 15), dtype=bool)


def RSA_physical_rdm():
    """Physical grouping RDM: groups by port identity.

    Groups: A choices (6), Initiations 1&3 (2), Initiation 2 & B task 3 (3),
    B task 2 (2), B task 1 (2).
    """
    a = _zeros_15x15(6)
    i1_i3 = _zeros_15x15(2)
    i2_b3 = _zeros_15x15(3)
    b2 = _zeros_15x15(2)
    b1 = _zeros_15x15(2)
    a[:, 0:6] = True
    i1_i3[:, 6:8] = True
    i2_b3[:, 8:11] = True
    b2[:, 11:13] = True
    b1[:, 13:15] = True
    return np.vstack([a, i1_i3, i2_b3, b2, b1])


def RSA_a_b_rdm():
    """Choice identity RDM: groups A vs B choices (ignoring initiations)."""
    a = _zeros_15x15(6)
    i1_i3_i2 = _zeros_15x15(3)
    b = _zeros_15x15(6)
    a[:, 0:6] = True
    b[:, 9:15] = True
    return np.vstack([a, i1_i3_i2, b])


def reward_rdm():
    """Reward outcome RDM: groups rewarded vs non-rewarded trials."""
    a = _zeros_15x15(6)
    for i in [0, 2, 4]:
        a[i, [0, 2, 4, 9, 11, 13]] = True
    for i in [1, 3, 5]:
        a[i, [1, 3, 5, 10, 12, 14]] = True
    i1_i3_i2 = _zeros_15x15(3)
    return np.vstack([a, i1_i3_i2, a])


def reward_choice_space():
    """Reward x Choice interaction RDM."""
    return reward_rdm() & RSA_a_b_rdm()


def choice_vs_initiation():
    """Choice vs Initiation RDM: groups all choices together, initiations separate."""
    a = _zeros_15x15(6)
    i1_i3_i2 = _zeros_15x15(3)
    b = _zeros_15x15(6)
    a[:, 0:6] = True
    a[:, 9:15] = True
    b[:, 0:6] = True
    b[:, 9:15] = True
    i1_i3_i2[:, 6:9] = True
    return np.vstack([a, i1_i3_i2, b])


def a_bs_task_specific():
    """Task-specific A/B RDM: groups A and B choices within each task separately."""
    a = _zeros_15x15(6)
    i1_i3_i2 = _zeros_15x15(3)
    for i in range(len(a)):
        if i % 2 == 0:
            a[i, i:i + 2] = True
        else:
            a[i, i - 1:i + 1] = True
    b = np.flip(a)
    return np.vstack([a, i1_i3_i2, b])

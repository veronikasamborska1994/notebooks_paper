"""Validation tests for analysis functions.

These tests verify that:
1. Helper functions produce correct outputs on known inputs
2. RDM matrices have expected shapes and properties
3. Data import and extraction functions work correctly
4. Key analysis functions produce consistent outputs
5. No NumPy/SciPy deprecation warnings occur

Run with: python -m pytest tests/test_validation.py -v
Or standalone: python tests/test_validation.py
"""

import sys
import os
import warnings
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Test helper_functions.py
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for functions in helper_functions.py."""

    def test_cpd_known_input(self):
        """CPD should be ~1 for a perfectly predictive variable."""
        from functions.helper_functions import _cpd
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 3.0 * x1 + 0.0 * x2 + np.random.randn(n) * 0.01
        X = np.column_stack([x1, x2, np.ones(n)])
        cpd = _cpd(X, y)
        # x1 should have high CPD, x2 should have near-zero CPD
        assert cpd[0] > 0.9, f"CPD for strong predictor should be > 0.9, got {cpd[0]}"
        assert cpd[1] < 0.1, f"CPD for null predictor should be < 0.1, got {cpd[1]}"

    def test_CPD_multi_output(self):
        """Multi-output CPD should return correct shape."""
        from functions.helper_functions import _CPD
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.random.randn(n), np.random.randn(n), np.ones(n)])
        y = np.random.randn(n, 5)
        cpd = _CPD(X, y)
        assert cpd.shape == (5, 3), f"Expected shape (5, 3), got {cpd.shape}"
        assert np.all(np.isfinite(cpd)), "CPD should not contain inf values"

    def test_exp_mov_ave_converges(self):
        """Exponential moving average of constant should converge to that constant."""
        from functions.helper_functions import exp_mov_ave
        data = np.ones(200) * 0.8
        result = exp_mov_ave(data, tau=8, initValue=0.5)
        assert len(result) == len(data), "Output length should match input"
        assert abs(result[-1] - 0.8) < 0.01, f"EMA should converge to 0.8, got {result[-1]}"
        assert result[0] < result[-1], "EMA should increase from 0.5 toward 0.8"

    def test_exp_mov_ave_alpha(self):
        """EMA with alpha=1 should equal the input data."""
        from functions.helper_functions import exp_mov_ave
        data = np.array([0.1, 0.9, 0.3, 0.7])
        result = exp_mov_ave(data, alpha=1.0)
        np.testing.assert_array_almost_equal(result, data)

    def test_task_ind_values(self):
        """task_ind should only return values in {0, 1, 2, 3}."""
        from functions.helper_functions import task_ind
        task = np.ones(10)
        a_pokes = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        b_pokes = np.array([5, 2, 3, 7, 8, 1, 9, 4, 6, 5])
        result = task_ind(task, a_pokes, b_pokes)
        assert len(result) == 10
        assert set(np.unique(result)).issubset({0, 1, 2, 3})
        # b_poke=5 means 10-5=5=a_poke, so task 1
        assert result[0] == 1
        # b_poke=2 is diagonal, so task 2
        assert result[1] == 2
        # b_poke=1 is top/bottom, so task 3
        assert result[5] == 3

    def test_exponential(self):
        """Exponential function should produce correct output."""
        from functions.helper_functions import exponential
        assert exponential(0, 1, 1, 0) == 1.0  # a*exp(0) + 0 = a
        assert abs(exponential(1, 1, 0, 2) - 3.0) < 1e-10  # 1*exp(0) + 2

    def test_animal_exp_permute_structure(self):
        """animal_exp_permute should return 7 groups (3 HP + 4 PFC)."""
        from functions.helper_functions import animal_exp_permute
        # Create mock data with correct session counts
        HP_n = 15 + 7 + 10  # 3 HP animals
        PFC_n = 9 + 16 + 13 + 5  # 4 PFC animals
        mock_HP = {
            'DM': np.array([[np.zeros((50, 10)) for _ in range(HP_n)]], dtype=object),
            'Data': np.array([[np.zeros((50, 5, 63)) for _ in range(HP_n)]], dtype=object)
        }
        mock_PFC = {
            'DM': np.array([[np.zeros((50, 10)) for _ in range(PFC_n)]], dtype=object),
            'Data': np.array([[np.zeros((50, 5, 63)) for _ in range(PFC_n)]], dtype=object)
        }
        dm, fr = animal_exp_permute(mock_HP, mock_PFC)
        assert len(dm) == 7, f"Expected 7 animal groups, got {len(dm)}"
        assert len(fr) == 7, f"Expected 7 animal groups, got {len(fr)}"
        # Check session counts per animal
        assert len(dm[0]) == 15  # HP animal 1
        assert len(dm[1]) == 7   # HP animal 2
        assert len(dm[3]) == 9   # PFC animal 1
        assert len(dm[4]) == 16  # PFC animal 2
        assert len(dm[5]) == 13  # PFC animal 3


# ---------------------------------------------------------------------------
# Test RSAs.py
# ---------------------------------------------------------------------------

class TestRDMs:
    """Tests for RDM construction functions."""

    def test_rdm_shapes(self):
        """All RDMs should be 15x15."""
        from functions.RSAs import (RSA_physical_rdm, RSA_a_b_rdm, reward_rdm,
                                     reward_choice_space, choice_vs_initiation,
                                     a_bs_task_specific)
        for name, func in [('physical', RSA_physical_rdm),
                           ('a_b', RSA_a_b_rdm),
                           ('reward', reward_rdm),
                           ('reward_choice', reward_choice_space),
                           ('choice_init', choice_vs_initiation),
                           ('task_specific', a_bs_task_specific)]:
            rdm = func()
            assert rdm.shape == (15, 15), f"{name} RDM should be (15,15), got {rdm.shape}"
            assert rdm.dtype == bool, f"{name} RDM should be boolean"

    def test_rdm_not_all_same(self):
        """Each RDM should have both True and False values."""
        from functions.RSAs import (RSA_physical_rdm, RSA_a_b_rdm, reward_rdm,
                                     reward_choice_space, choice_vs_initiation,
                                     a_bs_task_specific)
        for func in [RSA_physical_rdm, RSA_a_b_rdm, reward_rdm,
                     reward_choice_space, choice_vs_initiation, a_bs_task_specific]:
            rdm = func()
            assert rdm.any(), "RDM should have at least one True"
            assert not rdm.all(), "RDM should have at least one False"

    def test_choice_ab_groups_correct_rows(self):
        """A/B RDM: rows 0-5 (A) should mark columns 0-5, rows 9-14 (B) mark 9-14."""
        from functions.RSAs import RSA_a_b_rdm
        rdm = RSA_a_b_rdm()
        # A rows should only have True in A columns
        assert np.all(rdm[:6, :6] == True)
        assert np.all(rdm[:6, 6:] == False)
        # B rows should only have True in B columns
        assert np.all(rdm[9:, 9:] == True)
        assert np.all(rdm[9:, :9] == False)

    def test_reward_choice_is_intersection(self):
        """reward_choice_space should be reward AND choice."""
        from functions.RSAs import reward_rdm, RSA_a_b_rdm, reward_choice_space
        expected = reward_rdm() & RSA_a_b_rdm()
        actual = reward_choice_space()
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Test for deprecation warnings
# ---------------------------------------------------------------------------

class TestNoDeprecations:
    """Ensure no NumPy/SciPy deprecation warnings in imports."""

    def test_no_numpy_deprecations(self):
        """Importing all modules should not trigger NumPy deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Re-import modules to check for deprecation warnings
            import importlib
            modules_to_check = [
                'functions.helper_functions',
                'functions.RSAs',
            ]
            for mod_name in modules_to_check:
                if mod_name in sys.modules:
                    importlib.reload(sys.modules[mod_name])
                else:
                    importlib.import_module(mod_name)

            numpy_warnings = [x for x in w if 'numpy' in str(x.message).lower()
                              and issubclass(x.category, DeprecationWarning)]
            assert len(numpy_warnings) == 0, \
                f"Got NumPy deprecation warnings: {[str(x.message) for x in numpy_warnings]}"


# ---------------------------------------------------------------------------
# Test batch_corrcoef
# ---------------------------------------------------------------------------

class TestBatchCorrcoef:
    """Tests for the vectorized Pearson correlation function."""

    def test_perfect_correlation(self):
        """Perfectly correlated data should give r=1."""
        from functions.functions_cells_policy import _batch_corrcoef
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = np.column_stack([y, y * 2, y + 10])  # all perfectly correlated with y
        r = _batch_corrcoef(X, y)
        np.testing.assert_array_almost_equal(r, [1.0, 1.0, 1.0])

    def test_zero_correlation(self):
        """Orthogonal data should give r~0."""
        from functions.functions_cells_policy import _batch_corrcoef
        y = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        x = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
        r = _batch_corrcoef(x.reshape(-1, 1), y)
        assert abs(r[0]) < 0.01, f"Expected ~0 correlation, got {r[0]}"

    def test_matches_numpy_corrcoef(self):
        """Should match np.corrcoef for each column."""
        from functions.functions_cells_policy import _batch_corrcoef
        np.random.seed(42)
        n, p = 50, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        r_batch = _batch_corrcoef(X, y)
        for i in range(p):
            r_numpy = np.corrcoef(X[:, i], y)[0, 1]
            assert abs(r_batch[i] - r_numpy) < 1e-10, \
                f"Column {i}: batch={r_batch[i]}, numpy={r_numpy}"


# ---------------------------------------------------------------------------
# Run as standalone script
# ---------------------------------------------------------------------------

class TestExactPermThreshold:
    """Tests for the exact threshold used with the 35 animal splits."""

    def test_35_splits(self):
        from functions.helper_functions import exact_perm_threshold
        null = np.random.RandomState(0).randn(35)
        s = np.sort(null)
        thr = exact_perm_threshold(null, 0.05)
        assert thr == s[-2], "threshold should be the 2nd largest of 35"
        assert s[-1] > thr, "largest split (P = 1/35) should be significant at .05"
        assert not s[-2] > thr, "2nd largest split (P = 2/35) should not be significant"
        assert np.isinf(exact_perm_threshold(null, 0.01)), "P < .01 cannot be reached with 35 splits"

    def test_axis_and_large_n(self):
        from functions.helper_functions import exact_perm_threshold
        null = np.random.RandomState(1).randn(35, 4)
        assert exact_perm_threshold(null, 0.05, axis=0).shape == (4,)
        big = np.random.RandomState(2).randn(5000)
        assert abs(exact_perm_threshold(big, 0.05) - np.percentile(big, 95)) < 0.01


if __name__ == '__main__':
    print("Running validation tests...\n")
    n_pass = 0
    n_fail = 0
    n_total = 0

    for cls_name, cls in [(n, c) for n, c in list(globals().items()) if isinstance(c, type) and n.startswith('Test')]:
        print(f"\n{cls_name}:")
        instance = cls()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                n_total += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {method_name}")
                    n_pass += 1
                except Exception as e:
                    print(f"  FAIL: {method_name} - {e}")
                    n_fail += 1

    print(f"\n{'='*50}")
    print(f"Results: {n_pass}/{n_total} passed, {n_fail} failed")
    if n_fail == 0:
        print("All tests passed!")
    else:
        sys.exit(1)

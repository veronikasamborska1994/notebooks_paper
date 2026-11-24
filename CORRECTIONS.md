# Corrections to the analysis code

The code in this repository differs from the version used for the published paper
(Samborska et al., Nat Neurosci 25, 1314-1326, 2022). The bugs below were found and fixed
in November 2026. Every fix has a test in `tests/`, and all affected figures in `figures/`
were regenerated on the full data set (5,000 session permutations; all 35 animal splits).

| # | Bug | Panels affected |
|---|---|---|
| 1 | Multiple-comparison correction over time took the percentile first and the maximum second | Fig 4b, 4e, 6b, 6e; Ext 5a, 5b, 5d; Ext 7 |
| 2 | The 99th percentile was labelled P < 0.001; it is P < 0.01 | Fig 4b, 4e, 6b; Ext 5a, 5b; Ext 7 |
| 3 | Policy permutation test kept only the last shuffle | Fig 6e; Ext 5d; Ext 7 |
| 4 | Policy null distribution was taken one time bin later than the data | Fig 6e; Ext 5d; Ext 7 |
| 5 | Decoding returned one PFC matrix twice | Fig 4g |
| 6 | Cross-region SVD used problem-1 modes for problem-2 data | Fig 5e |
| 7 | Training-policy coefficients were not reset between problems | Fig 1i-l |
| 8 | Wrong subject labels in the repeated-measures ANOVA | Fig 1k, l |
| 9 | Wrong upper error band on the choice-history weights | Fig 6a |
| 10 | Layout-3 port correction skipped in the forced-trial analysis | Ext 2i, j |
| 11 | Misplaced logic in the all-pokes out-of-sequence count | Ext 2e-g |
| 12 | Misplaced bracket in the selection of non-rewarded trials | Ext 4a-f |
| 13 | Decoding null averaged one direction with itself | Fig 4g |
| 14 | Decoding "Correct" significance used the wrong cell for B no-reward | Fig 4g |
| 15 | Out-of-sequence count in the first problem of a session never counted A pokes after a B choice | Ext 2d-h |
| 16 | Animal-permutation thresholds were interpolated percentiles of 35 splits; now exact rank-based (smallest possible P = 1/35) | Ext 5a-d; Ext 9d |
| 17 | Decoding null was not built from the same quantity as the observed difference | Fig 4g |
| 18 | The same error bar was drawn on every problem | Fig 1g |
| 19 | The Port bar averaged the confusion-matrix cells that compare different physical ports | Fig 4g |
| 20 | The reversal window started one trial early; sessions with a single reversal were left out, although every reversal with 10 trials on each side qualifies | Fig 1f (final: t(8) = -8.61, p < .001) |
| 21 | Trials to threshold were carried over incorrectly across sessions; now every trial since the last reversal is counted | Fig 1e; Ext 1a |
| 22 | Out-of-sequence poke counts were stored one trial late and blocks started on the last trial of the old block | Fig 1g, h; Ext 1b; Ext 2d-h |
| 23 | The first block of each recording session was counted one trial short | Ext 2a-c |
| 24 | The rewarded-streak counter selecting forced trials lagged by a trial and did not reset at an error | Ext 2i, j |
| 25 | The policy regressor was built with choices coded differently from the logistic fit | Fig 6b-e; Ext 5c, d; Ext 7 |
| 26 | The "B - A, PFC vs CA1" test was computed as A - B | Fig 6e; Ext 5d; Ext 7 |

## Results that change

- Fig 4g, Port: with the cells that share a physical port (Initiation of layout 2 and B choice of layout 3, both
  directions), CA1 44.4% and PFC 51.0% (P = 23/35 = 0.657); the decoder does not show a stronger port code in CA1.
  The original code gave CA1 68.1% (two states at different ports) and PFC 27.1% (one same-port and one
  different-port cell, because of bug 5). `dec.plot(HP, PFC, same_port=False)` uses the original cells with bug 5 fixed (PFC 13.5%).
  The result depends on the definition: also counting B reward at that port, averaged over cells, gives CA1 38.9%
  and PFC 25.5% (P = 3/35 = 0.086). The decoder uses the last six trials of each type and the two bins around each poke.
- Fig 4f: the Port bar uses the layout 2/3 pair only, so `dec.plot` also draws those matrices with the Port cells boxed
  (`figures/fig4fg_decode_3.png`).
- Fig 4f, PFC matrix: changes slightly because of bug 5 (B choice decoded as Initiation 9% to 16%, as B choice 54% to 47%).
- Fig 6e: PFC > CA1 holds for B choices at choice and outcome time. For A choices only a few
  bins remain, and the A versus B difference is a single bin.
- Extended Data Fig 5 (animals as the unit): outcome (5a), outcome and problem-specific A
  choice (5b), policy on A (5c, P = 2/35) and A and A versus B (5d) do not reach P < 0.05.
- Large significance markers are P < 0.01, not P < 0.001.
- Fig 1k, history of choices: F(9,72) = 2.71, P = 0.009 (printed as P = 0.773).

All other conclusions are unchanged; several statistics change in value.

## Choices kept as in the original code

- The decoder uses scikit-learn's default RBF kernel, as for the published figure.
- The policy regressor is the linear score (log-odds) of the logistic model.

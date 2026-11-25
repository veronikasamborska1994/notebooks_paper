# Corrections to the analysis code

The code in this repository differs from the version used for the published paper
(Samborska et al., Nat Neurosci 25, 1314-1326, 2022). The bugs below were found and fixed
in November 2026. Every fix has a test in `tests/`, and all figures in `figures/` were
regenerated on the full data set (5,000 session permutations, random seed 0; all 35 animal splits).

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
| 11 | Misplaced logic in the all-pokes out-of-sequence count | Ext 2d-g |
| 12 | Misplaced bracket in the selection of non-rewarded trials | Ext 4a-f |
| 13 | Decoding null averaged one direction with itself | Fig 4g |
| 14 | Decoding "Correct" significance used the wrong cell for B no-reward | Fig 4g |
| 15 | Out-of-sequence count in the first problem of a session never counted A pokes after a B choice | Ext 2d-h |
| 16 | Animal-permutation thresholds were interpolated percentiles of 35 splits; now exact rank-based (smallest possible P = 1/35) | Ext 5a-d; Ext 9d |
| 17 | Decoding null was not built from the same quantity as the observed difference | Fig 4g |
| 18 | The same error bar was drawn on every problem | Fig 1g |
| 19 | The Port bar averaged the confusion-matrix cells that compare different physical ports | Fig 4g |
| 20 | The reversal window started one trial early; sessions with a single reversal were left out | Fig 1f (final: t(8) = -8.61, p < .001) |
| 21 | Trials to threshold were carried over incorrectly across sessions (stale index, forced trials subtracted twice, counts overwritten, trials after a session's last reversal dropped) | Fig 1e; Ext 1a |
| 22 | Out-of-sequence poke counts were stored one trial late and blocks started on the last trial of the old block | Fig 1g, h; Ext 1b; Ext 2d-h |
| 23 | The first block of each recording session was counted one trial short | Ext 2a-c |
| 24 | The rewarded-streak counter selecting forced trials lagged by a trial and did not reset at an error | Ext 2i, j |
| 25 | The policy regressor was built with choices coded differently from the logistic fit | Fig 6b-e; Ext 5c, d; Ext 7 |
| 26 | The animal-split null distributions of the policy tests used a policy over 3 past trials instead of 11 (a variable was overwritten) | Ext 5c, d |

## Results that change

- Fig 4g, Port: with the cells that share a physical port (Initiation of layout 2 and B choice of layout 3, both
  directions), CA1 44.4% and PFC 51.0% (P = 23/35 = 0.657); the decoder does not show a stronger port code in CA1.
  The original code gave CA1 68.1% (two states at different ports) and PFC 27.1% (one same-port and one
  different-port cell, because of bug 5). `dec.plot(HP, PFC, same_port=False)` uses the original cells with bug 5 fixed (PFC 13.5%).
  The result depends on the definition: also counting B reward at that port, averaged over cells, gives CA1 38.9%
  and PFC 25.5% (P = 3/35 = 0.086). The decoder uses the last six trials of each type and the two bins around each poke.
- Fig 4f is redrawn (`figures/fig4fg_decode_1.png`) so that every bar of Fig 4g can be read off it. The printed panel showed
  the average over all layout pairs with green boxes, but the Port bar was computed from the layout 2/3 pair only. Now the
  averaged matrix carries the Correct (blue) and Position (red) cells, and the two layout 2/3 matrices carry the Port cells (green),
  with every state labelled with its layout.
- Fig 4f, PFC matrix: changes slightly because of bug 5 (B choice decoded as Initiation 9% to 16%, as B choice 54% to 47%).
- Fig 6e: PFC > CA1 holds for B choices at choice and outcome time. For A choices only a few
  bins remain, none at outcome time. The A versus B comparison (grey markers) is removed from Fig 6e and
  Extended Data Fig 5d and 7g: at most one bin reached P < 0.05, whichever way the difference was taken, so the
  larger A versus B difference in CA1 at outcome time reported in the paper is not supported.
- Extended Data Fig 5 (animals as the unit): outcome (5a), outcome and problem-specific A
  choice (5b), policy on A (5c, P = 2/35), and A (5d) do not reach P < 0.05.
- Extended Data Fig 7f: policy on A choices now generalizes more strongly in PFC than CA1 (P < 0.05); the caption says it does not.
- Extended Data Fig 9d, all time points, cellular and temporal vectors (animals as the unit): P = 2/35 = 0.057; printed above the threshold.
- Large significance markers are P < 0.01, not P < 0.001.
- Fig 1k, history of choices: F(9,72) = 2.71, P = 0.009 (printed as P = 0.773).
- Extended Data Fig 2h (A/B pokes only): the first reversal block differs significantly only from the third
  (P = 0.088, 0.039, 0.123); the caption's statement holds for all pokes (2g: P = 0.023, 0.012, 0.012).

All other conclusions are unchanged; several statistics change in value.

Forced-choice trials (a quarter of all trials) are included in all neural analyses, as in the original code. In a quick check,
excluding them changes several results (Fig 4g Correct P = 2/35 to 10/35; Extended Data Fig 5c B 1/35 to 3/35).

## For discussion: the Fig 4g Port bar

In layouts 2 and 3 the initiation port of layout 2 (I2) is the B port of layout 3 (B3). The Port bar should count I2 and B3 being
confused. The original code counted I3 and B2 instead, which are different ports (for PFC one of the two cells, because of bug 5).

| Port bar (%) | CA1 | PFC | CA1 > PFC |
|---|---|---|---|
| Printed | 68.1 | 27.1 | P < 0.05 |
| Same port (I2 and B3), final | 44.4 | 51.0 | 23/35 |

So Fig 4g does not support "CA1 predominantly decoded to the same physical port". Fig 4e and Extended Data Fig 4 do not use these cells.

A new post hoc analysis (`figures/posthoc_port_over_time.png`, linear decoder, all 81 sessions) follows the port over time.
CA1 / PFC in %, chance 33; P of CA1 > PFC over the 35 animal splits (smallest possible 0.029):

| A/B window | Same port | Same role | Different ports (control) |
|---|---|---|---|
| Choice poke | 72.9 / 63.0 (P = 0.200) | 18.1 / 20.8 | 64.2 / 47.6 |
| Rewarded, choice to outcome | 78.2 / 44.1 (P = 0.029) | 17.9 / 34.0 | 66.8 / 24.6 (P = 0.029) |

At the poke both regions represent the port; afterwards CA1 keeps it and PFC moves toward the role. This does not isolate the shared
port: CA1 is also above PFC for two different ports, and the difference between the two is not larger in CA1 (P = 0.286). The window
and decoder were chosen after seeing the Fig 4g result, and the Fig 4g decoder itself shows the same direction without significance.

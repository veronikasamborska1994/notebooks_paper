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
| 27 | The SVD between-problem projections used the modes of problem 1 for both problem 2 and problem 3 (3 half-steps away for problem 3), the second half of problem 3 was never used, and the within-problem pair for problem 3 was missing; within pairs took modes from the later half and between pairs from the earlier. Now every pair is one half-step apart (within 1a/1b, 2a/2b, 3a/3b; between 1b/2a, 2b/3a) and each pair is used in both directions, with the two curves averaged | Fig 5d-j; Ext 9a-d |
| 28 | The within-region roll null of the single-cell GLM rolled problems 1 and 2 by shifts drawn up to the trial count of problem 3 (a variable left over from a loop), so many shifts wrapped round and some gave back the real data; shifts are now drawn within each problem's own trials, at least the shortest block away from the identity in both directions | Fig 4b; Ext 5a (within-region markers) |
| 29 | The roll null of the policy GLM drew shifts up to the session's trial count, but the regression drops the first n + 1 trials, so about 5% of shifts gave back the real data or moved it by at most n trials; shifts are now drawn within the rows regressed, at least n away from the identity | Fig 6b; Ext 7 policy GLM |
| 30 | The time control of the surprise analysis kept sessions in which the second layout came right before the first, but compared the end of the first layout with the start of the second (not adjacent in those sessions); for T1-T3 both orders were kept. Sessions in which the two layouts are adjacent in either order are now kept, and the end of the earlier layout is compared with the start of the later one | Ext 8 |
| 31 | In the recordings out-of-sequence count the last free trial before a layout switch was always scored zero (the trial was checked against the next layout's ports, with an off-by-one trial index, and a reset ran on every poke). It is now excluded, because the pokes after it belong to the next layout's free-reward and forced trials. The first free trial of each layout is scored normally | Ext 2d-h |
| 32 | A block that crossed the performance threshold just before its session ended took two reversal slots when the next session went on with the same block (13 of 17 such cases), shifting every later reversal of that problem by one | Fig 1e; Ext 1a |
| 33 | Problems were told apart by the initiation port only, which merged two layouts that share it (615 and 653 in mice 478, 480 and 481). Problems are now told apart by the whole port layout; the three short 653 runs (one session, 2-4 reversals; every other problem has at least 9) are left out as interrupted problems | Fig 1e-l |
| 34 | The A-rewarded error band of the example neurons used the s.e.m. of layout 1 in every layout | Fig 2 |
| 35 | The two SVD panels with a shuffle had their titles swapped (the "temporal shuffle" panel shuffled cells and vice versa) | Fig 5g; Ext 9 |
| 36 | Plotting only: the first-10-blocks band of Fig 1f was almost white (grey at alpha 0.1) and the other band had no fill colour in the SVG; RSA ticks for +0.5 s and +1 s were one to four bins off and the P < .01 legend was not drawn; the last time tick (+1 s) was one bin past the data; "Previous outcome x choice" in Fig 1i-l had the choice colour; the Fig 4g legend named CA1 "HP" and covered a data point | Fig 1f, 1i-l, 4b, 4e, 4g, 6b, 6e; Ext 5, 7 |
| 37 | The finite-value mask of the policy correlations ran over time bins instead of neurons (no effect: the data have no non-finite values) | Fig 6c-e; Ext 5c, d; Ext 7f, g |
| 38 | The decoder used only the last six trials of each state (`MIN_TRIALS // 2 + 1`), although every session it uses has at least 11 of each; this is not in the Methods. It now uses 11 trials per state drawn at random from the whole layout, with the confusion matrices averaged over 100 draws (seed 0) | Fig 4f, g |

## Results that change

- Fig 4g, Port: with the cells that share a physical port (Initiation of layout 2 and B choice of layout 3, both
  directions), CA1 68.6% and PFC 59.7% (P = 8/35 = 0.229); the decoder does not show a significantly stronger port
  code in CA1 at the choice poke. Correct (CA1 34.5%, PFC 58.0%, P = 2/35) and Position (2.1%, 25.3%, P = 1/35) hold.
  The original code gave CA1 68.1% (I3 and B2, two neighbouring but different ports) and PFC 27.1% (one same-port and one
  different-port cell, because of bug 5). `dec.plot(HP, PFC, same_port=False)` uses the original cells with bug 5 fixed.
  With the last six trials per state (bug 38) the same-port values were CA1 44.4% and PFC 51.0% (P = 23/35).
  The decoder uses the two bins up to each poke.
- Fig 4f is redrawn (`figures/fig4fg_decode_1.png`) so that every bar of Fig 4g can be read off it. The printed panel showed
  the average over all layout pairs with green boxes, but the Port bar was computed from the layout 2/3 pair only. Now the
  averaged matrix carries the Correct (blue) and Position (red) cells, and the two layout 2/3 matrices carry the Port cells (green),
  with every state labelled with its layout.
- Fig 4f: all matrices change slightly because of bugs 5 and 38.
- Fig 6e: PFC > CA1 holds for B choices at choice and outcome time. For A choices only a few
  bins remain, none at outcome time. The A versus B comparison (grey markers) is removed from Fig 6e and
  Extended Data Fig 5d and 7g: at most one bin reached P < 0.05, whichever way the difference was taken, so the
  larger A versus B difference in CA1 at outcome time reported in the paper is not supported.
- Extended Data Fig 5 (animals as the unit): outcome (5a), outcome and problem-specific A
  choice (5b), policy on A (5c, P = 2/35), and A (5d) do not reach P < 0.05.
- Extended Data Fig 7f: policy on A choices now generalizes more strongly in PFC than CA1 (P < 0.05); the caption says it does not.
- Fig 5h-j (CA1 vs PFC; published P in brackets): temporal 5h P = 0.328 (0.128), cellular 5i P = 0.041 (0.018),
  cellular and temporal 5j P = 0.0024 (0.006). 5h stays not significant, 5i is borderline, 5j holds. Ext 9a-c (session permutations):
  5h P = 0.310, 0.801, 0.351; 5i P = 0.0002, 0.0002, 0.116 (9c no longer significant); 5j P = 0.0002, 0.0002, 0.018.
  Ext 9d (animals as the unit): all time points 5h P = 17/35, 5i and 5j P = 1/35 = 0.029 (5j was printed as 2/35, above the threshold);
  initiation to choice 5h P = 28/35, 5i P = 1/35, 5j P = 3/35.
- Fig 5g and 5j use cumulative absolute diagonal weights (as coded); the Results text says squared. Fig 5h and 5i use squared sums (variance per mode).
- Large significance markers are P < 0.01, not P < 0.001.
- Fig 1e and Extended Data Fig 1a (bugs 32, 33): problem F(9,72) = 6.99, P < 0.001 (Fig 1e; printed 3.91); reversal within
  problem F(9,72) = 4.46, P < 0.001 (Extended Data Fig 1a; printed 3.52, P = 0.001). Before bugs 32 and 33: 6.65 and 3.49. Fig 1g, h, Extended Data Fig 1b and Fig 1f do not change with bugs 32 and 33.
- Fig 1i-l (bug 33): previous choice F(9,72) = 13.61, history of choices F(9,72) = 2.57, P = 0.013 (printed as P = 0.773),
  previous outcome x choice F(9,72) = 6.58, history of outcomes x choices F(9,72) = 4.56 (all P < 0.05).
- Extended Data Fig 2d-h (bug 31): the conclusions hold. With all pokes (2g) the first reversal block differs from the second,
  third and fourth (P = 0.023, 0.012, 0.014; printed P = 0.015, 0.001, < 0.001 with t(167)); with A/B pokes only (2h) it
  differs from the third only (P = 0.088, 0.039, 0.140), so the caption's "compared to every following reversal block" holds
  for 2g but not for 2h. Extended Data Fig 2d: reversal F(3,18) = 13.46, P < 0.001; problem F(23,138) = 1.00, P = 0.472
  (printed 11.12 and 0.10, P = 0.474). Extended Data Fig 2f: t(5) = 1.49, P = 0.197 (printed 1.39, P = 0.220).
  A/B pokes only: reversal F(3,18) = 4.92, P = 0.011.

- Fig 6b and the Extended Data Fig 7 policy GLM (bug 29): the null no longer contains near-unshuffled data, so more bins pass.
  Policy at P < 0.01: PFC bins 16-39 (before bug 29, 21-38), CA1 bins 36-40 (37-40); DLC residuals (Ext 7): PFC 28-38,
  CA1 39-43. Choice, previous choice and policy x choice now also pass in some bins before initiation. The conclusions
  (policy and policy x choice coded in both regions) hold.
- Fig 4b and Extended Data Fig 5a, within-region markers (bug 28): within each region the markers start a few bins earlier (Fig 4b at P < 0.01: choice
  PFC from bin 15, CA1 from bin 26; outcome PFC from bin 27, CA1 from bin 32; outcome x choice PFC from bin 37, CA1 from bin 41;
  a few isolated bins before initiation also pass), as the null is no longer inflated by unshuffled data. The between-region markers
  do not change (Fig 4b at P < 0.01: choice bins 35-45 and 47-50, outcome 40-62, outcome x choice 49-58; Ext 5a, animals as the unit: choice bins 36-41, outcome x choice 54-55, outcome none).
- Extended Data Fig 8 (bug 30): with sessions in which the two layouts are adjacent in either order (CA1 23, 28 and 21
  sessions for T1-T2, T2-T3 and T1-T3; PFC 30, 34 and 26), CA1 surprise after a layout switch is clearer than before,
  mostly at initiation, and PFC surprise is weaker. The conclusion (CA1 more surprised by a new layout than PFC) holds.

All other conclusions are unchanged; several statistics change in value.

Forced-choice trials (a quarter of all trials) are included in all neural analyses, as in the original code. In a quick check,
excluding them changes several results (Fig 4g Correct P = 2/35 to 10/35, checked with six trials per state; Extended Data Fig 5c B 1/35 to 3/35).

## For discussion: the Fig 4g Port bar

In layouts 2 and 3 the initiation port of layout 2 (I2) is the B port of layout 3 (B3). The Port bar should count I2 and B3 being
confused. The original code counted I3 and B2 instead (for PFC one of the two cells, because of bug 5). I3 and B2 are different
ports, but they are diagonal neighbours in all 81 sessions, so that measure is a neighbouring-port confusion, not a control.

| Port bar (%) | CA1 | PFC | CA1 > PFC |
|---|---|---|---|
| Printed | 68.1 | 27.1 | P < 0.05 |
| Same port (I2 and B3), final | 68.6 | 59.7 | 8/35 |
| Neighbouring ports (I3 and B2), final | 68.7 | 38.1 | 2/35 |

So at the choice poke Fig 4g does not show a significantly stronger code for the same physical port in CA1.

Post hoc, with the same decoder and the A/B choice window moved later (exploratory; the windows were chosen after seeing the
result; CA1 / PFC in %, P of CA1 > PFC over the 35 animal splits, smallest possible 1/35):

| A/B window (bins) | Same port | Neighbouring ports |
|---|---|---|
| 34-35 (Fig 4g) | 68.6 / 59.7 (8/35) | 68.7 / 38.1 (2/35) |
| 35-36 | 73.4 / 60.1 (8/35) | 64.2 / 38.6 (3/35) |
| 38-39 | 51.6 / 26.4 (2/35) | 38.3 / 9.4 (2/35) |
| 35-41, rewarded trials | 66.7 / 39.5 (1/35) | 48.0 / 17.7 (2/35) |

After the choice poke CA1 confuses initiation and B choice more than PFC, for the same port and for the neighbouring port alike;
the difference between the two is never larger in CA1 (P = 21/35 to 30/35 in these windows). This fits a location signal in CA1 at the scale of
neighbouring ports, not a code for the exact port. The layouts contain no distant port with a comparable role to test against.
`figures/posthoc_port_over_time.png` shows a related three-state linear decoder (20 random trials per state, all 81 sessions).

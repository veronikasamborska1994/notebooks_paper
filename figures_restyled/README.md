# Figures in the style of the published paper

The SVGs in `figures_svg/` redrawn in the style of the published figures (fonts, colours, line widths, panel sizes).
The data in each panel are read from the SVG and are not changed. File names match `figures/` and `figures_svg/`;
`fig1d_session_plot` has no restyled version.

To rebuild after regenerating `figures_svg/`:

```bash
cd figures_restyled/code
python restyle_all.py            # all figures
python restyle_all.py fig4b      # figures whose name contains "fig4b"
```

Differences from the code's own figures, all in presentation:

- Ext 5a shows only the markers for a difference between CA1 and PFC (as in the printed panel).
- Ext 5c permutation panels are labelled with the rank of the data among the 35 animal splits.
- The "Prev Choice" predictor is not drawn in Fig 6b and Ext 7 (as in the printed panels).
- Fig 4g carries the printed boxes and the CA1 versus PFC test above each pair of bars (animal permutation test:
  Correct 2/35, Position 1/35, Port 8/35); these labels are written in `restyle_all.py`, not read from the SVG.
- Fig 6e, Ext 5d and Ext 7g carry a small copy of a policy matrix with a line at the time of the slice.
- Reversal panels (Ext 1a, 1b, Ext 2a, 2d) are labelled by problem on the x axis.

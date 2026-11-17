# Windowed spike PCA and shape orientation

Load the analysis module in the repository's Julia environment. It does not run
or advance the model:

```julia
include("/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/RotatingShape_Drift.jl")
using .RotatingShapeDrift
```

For an interactive run, pause and obtain a snapshot first:

```julia
dashboard.running[] = false
result = interactive_result(dashboard)
```

For batch runs, use the existing `result`. It must have completed at least the
last requested iteration. The window includes both endpoints (5000:6000 contains
1001 spike vectors).

```julia
p = plot_orientation_pc1(result;
    window=5000:6000,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/PC1_orientation_5000_6000.png")
display(p.fig)
```

Alternatively, analyze a saved run directly, including older runs:

```julia
p = plot_orientation_pc1(
    "/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Data";
    window=5000:6000)
display(p.fig)
```

Replace `Data` with the particular `Data/Interactive/run_XXXX` folder when needed.
The saved spike and orientation rows must come from the same run.

By default there is one panel for subnetwork A (first half of the neurons) and
one for B (second half). Use `groups=(:all,)` for a single whole-network PCA.
The horizontal axis is input orientation wrapped to 0–360 degrees. The vertical
axis is PC1 score. Consecutive observations are joined in time order, with a
shared iteration color scale, except across the orientation wrap. Dots include
every observation even where no segment is drawn.

Each PCA is fitted only to the requested window and group. Observations are
spike-vector rows; neuron columns are centered but not standardized. PC1 is the
largest-variance direction obtained by SVD. The title reports the percentage of
within-window variance it explains. Windows with no spike variance raise a clear
error because PCA would be undefined.

Inspect the extracted data and fit:

```julia
a = p.analyses[1]          # A; analyses[2] is B
 a.spikes                 # selected iterations × neurons
 a.iterations
 a.orientation_degrees
 a.scores                 # PC1 at each selected iteration
 a.loadings               # PC1 weight for each selected neuron
 a.center                 # fitted mean spike value for each neuron
 a.explained_variance      # fraction, from 0 to 1
```

This plot explores orientation encoding and changes across successive rotations;
it does not itself establish representational drift or synchronization. Within
a window the PCA basis is fixed. Different windows and subnetworks use independent
fits: signs and directions can differ. A deterministic sign convention makes the
largest absolute loading positive, but does not align independent fits. For later
cross-window drift comparisons, fit a reference PCA once and project subsequent
spike vectors with its mean and loadings. Comparing the two scalar PC1 curves
alone cannot establish a constant relationship between the full population codes.

## Full-network PC1 versus PC2

Fit a single PCA across all neurons, using the same iteration window:

```julia
p2 = plot_spike_pc12(result;
    window=5000:6000,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/PC1_PC2_5000_6000.png")
display(p2.fig)
```

The function also accepts a saved-run directory as its first argument. PC1 is
horizontal and PC2 vertical, with equal units on both axes. The line connects
successive model iterations, with color indicating iteration. Axes show each
PC's explained variance. The PCA includes both subnetworks jointly; each neuron's
spikes are centered within the selected window without variance scaling.

```julia
p2.analysis.component_scores       # iterations × 2
p2.analysis.component_loadings     # all neurons × 2
p2.analysis.explained_variances    # two fractions of total spike variance
```

The existing orientation plot remains available. As with it, comparing separate
windows requires a shared basis if the coordinates are to be directly comparable.

## Separate subnetwork PC1 versus PC2 panels

```julia
p3 = plot_subnetwork_pc12(result;
    window=5000:6000,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/PC1_PC2_subnetworks_5000_6000.png")
display(p3.fig)
```

The first argument can also be a saved-run directory. PCA is fitted independently
to subnetwork A and B, using the same window and centering convention. Both
panels join successive iterations and share a time color scale. Each panel uses
equal units on its two axes, and labels show that subnetwork's explained variance.

```julia
p3.analyses[1].component_scores   # A, iterations × 2
p3.analyses[2].component_scores   # B, iterations × 2
```

These independent PCA bases are not aligned; differences in axis direction or
sign between the panels do not by themselves indicate different drift directions.

## 3D paired subnetwork PC1 with time

```julia
p4 = plot_subnetwork_pc1_time(result;
    window=5000:6000,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/PC1_subnetworks_time_5000_6000.png")
display(p4.fig)
```

A saved-run directory is also accepted as the first argument. The x-axis is
model iteration, the y-axis is subnetwork A's PC1 score, and the z-axis is B's
PC1 score. The default viewing angle places time horizontally from left to right. Points are paired by iteration and joined chronologically. The shared
iteration color scale runs from purple (early) to yellow (late). Each subnetwork
gets its own PCA fitted to the requested window with centered, unscaled spikes.

`p4.coordinates` contains three columns: iteration, PC1_A, PC1_B. The two fits
are in `p4.analyses[1]` and `p4.analyses[2]`. Optional `azimuth` and `elevation`
keywords (radians) let you choose the viewing angle for the static figure.
Separate PCA axes have arbitrary signs and capture only part of population
variance; this trajectory is exploratory rather than a test of synchronized drift.

## Rotate and explore the 3D trajectory

```julia
p4 = plot_subnetwork_pc1_time(result;window=5000:6000,interactive=true)
```

This opens a native GLMakie window directly; no additional display call is
needed. Drag with the left mouse button inside the axes to rotate. Use the Zoom
slider to inspect details and Reset view to restore the starting view and zoom.
The installed Makie version uses these explicit zoom controls. The default
`interactive=false` still produces the original static figure. The interactive
option also accepts a saved-run directory and does not advance the simulation.

Run from your normal graphical Julia session. If using this from a standalone
terminal script, call `wait(p4.screen)` to keep the process alive until the window
closes. This is unnecessary in the REPL. Native graphics require desktop access;
offscreen construction for tests uses `interactive=true, backend=:cairo, show=false`.

## PCA restricted to the horizontal L pose

```julia
p5 = plot_horizontal_subnetwork_pc12(result;
    window=5000:6000,
    target_angle_degrees=90.,
    tolerance_degrees=1.,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/Horizontal_PC1_PC2.png")
display(p5.fig)
```

The first argument can also be a saved-run directory. For the current L vertices,
90° counterclockwise from the initial upright shape places its long arm along
the bottom and its short arm pointing up. Use 270° for the opposite horizontal
pose, or another target if the shape geometry changes. Circular angular distance
handles wraparound; tolerance boundaries are inclusive with a tiny numerical
allowance. The default ±1° selects one frame per 120-iteration rotation at the
current speed. A longer window is often useful (e.g. `window=5000:40000`);
`window=nothing` selects matching orientations from the entire completed run.

PCA is fitted separately for A and B **after filtering** to this orientation.
It centers each selected neuron's spike values without scaling. Color indicates
the original iteration; lines join successive selected frames across rotations.
The fitted axes describe variability at this pose, rather than the axes fitted
to all orientations by the preceding plots. At least three matched observations
and nonzero spike variance are required. Wider tolerances can select several
frames per rotation and include additional orientation-related variability.

```julia
p5.selected_iterations
p5.analyses[1].spikes              # selected rows × A nodes
p5.analyses[2].spikes              # selected rows × B nodes
p5.analyses[1].component_scores    # selected rows × 2 PCs
```

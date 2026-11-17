# Interactive rotating-shape reservoir

## Launch

In a fresh local Julia session with the repository environment active:

```julia
include("Julia/RotatingShape/RotatingShape.jl")
```

The main script now defaults to `interactive=true` and opens a native GLMakie
window. It starts paused at iteration zero. The model parameters are the same as
the batch version; no reservoir equations or sensory projections were changed.
The default limit remains 40,000 recorded iterations.

Four synchronized panels form a 2×2 grid: the network and stimulus on the top
row, and the mean-absolute-error time series and target histogram on the bottom row. They show:

- The recurrent network: blue silent nodes in A, green silent nodes in B, and
  yellow nodes that spike in the current iteration. Gray arrows are within-half
  connections and orange arrows cross between the halves. Topology is fixed.
- The visual stimulus that was actually presented to produce those spikes. It is
  rendered as black/white pixels with an orange left/right divider. The divider
  is an overlay, not an input pixel. The reservoir uses black=1 and white=0.

- A time series of `mean(abs.(model.acts .- model.targets))`, averaged across
  reservoir nodes after each learning update. The horizontal axis is model
  iteration. Every completed iteration is recorded, including multi-step advances.
  Reset clears the curve. The axes rescale as the run progresses.

- A histogram of current target values across all reservoir nodes. It refreshes
  on each step and reset, with axes scaled to the current distribution.

The header reports iteration count, input-frame time, orientation and spike
count. At iteration zero the frame is a preview and the reservoir is still in
its initial state. After Step/Run, the displayed frame and spikes belong to the
same update, even though the underlying environment has already advanced to
its next timestep internally.

## Controls

| Control | Behavior |
|---|---|
| Run / Pause | Start or pause continuous stepping |
| Step | Pause and advance exactly one model iteration |
| Reset | Save any unsaved steps, then restore the original weights, angle and state |
| Save spikes | Pause and save all completed steps of the current run |
| Playback slider | Set the requested display/stepping rate from 1 to 60 iterations per second |

The 40,000-step limit stops playback and saves automatically. Closing the window
also saves completed unsaved steps. If saving fails, the status line reports the
error. Reset does not throw away the previous run; each run has its own numbered
folder under `Data/Interactive/run_0001/`, `run_0002/`, and so on. This keeps the
previous fast-batch `Data/spikes.csv` untouched. Saving repeatedly within the
same run refreshes that run's files. Empty runs create no spike CSV.

Playback controls wall-clock pace, not model dt. Each displayed update runs one
actual reservoir/learning step; iterations are not skipped. At dt=1/30 and
30 playback iterations/second the requested pace matches simulation real time.
A slow machine can render below the requested rate; this changes wall-clock
speed only, not the simulated time increment or stimulus angle sequence.

## Main-script options

```julia
interactive = true
playback_fps = 30
start_running = false
```

Set `interactive=false` to restore the original fast batch run and its automatic
CSV export to `Data/`. Batch previews and animation options still apply in that
mode. The interactive script does not first run all 40,000 steps in the background.
For a terminal-launched Julia script, the main file waits for the window to close;
when included from the REPL it returns while the window and timer stay active.

The new file `RotatingShape_Interactive.jl` defines the
`RotatingShapeInteractive` module. The visual and reservoir modules must be loaded
first. It uses the same observable-driven GLMakie approach as ObjectTracking,
Pong and WallAvoidance, with controls for the existing non-Agents visual model.
It does not introduce artificial spatial agents or change model dynamics.

## Access current data

Pause first, then take a copied snapshot:

```julia
result = interactive_result(dashboard)
result.ts.spikes
result.ts.time
result.ts.angle
```

This result has the same layout as batch results and includes only completed
rows. `model` in the main script is the live model; its identity is retained
through Reset. `dashboard.state.model` is also available. Programmatic controls:

```julia
dashboard.running[] = true   # Run
dashboard.running[] = false  # Pause
dashboard.advance(1)        # advance exactly one step
dashboard.save()            # save current run
dashboard.reset()           # save unfinished run and restore initial state
dashboard.stop()            # stop timer and save; does not close the window
```

The controls use the same stepping and recording helpers as batch mode. Current
spike colors and image data are exposed as `dashboard.node_colors[]` and
`dashboard.image_data[]` for inspection. The current per-node absolute errors are available as
`dashboard.absolute_errors[]`, the plotted curve as `dashboard.error_history[]`,
and target histogram values as
`dashboard.target_values[]`.

## Validation

Offscreen tests exercise control callbacks, model/state reset, timer playback,
spike recording, cap/completion saving, stop/close saving behavior, frame/spike
alignment and exact agreement with batch stepping. An independently rendered
preview was visually checked. Batch results after factoring out shared recording
helpers exactly matched the preceding implementation over 120 steps.
GLMakie is an existing project dependency; a local graphical Julia session is
required for native windows. `backend=:cairo, show=false, start_timer=false`
provides offscreen construction for tests without opening a window.

The 63 offscreen interactive checks and 34 existing reservoir checks passed.
Native-window verification could not complete inside the automation host: macOS
desktop services rejected the graphics initialization. Launch the main script
from your normal local Julia session, where your other GLMakie models run.

The full precision mean-absolute-error history is `result.ts.mean_abs_error` and is
also saved as `mean_absolute_error` in `reservoir_summary.csv`.

## Subnetwork comparisons

The error panel plots mean absolute error separately for A (blue) and B (green),
using all nodes within the respective half. Histories are available as
`result.ts.mean_abs_error_A` and `result.ts.mean_abs_error_B`, and saved in
`reservoir_summary.csv` as `mean_absolute_error_A` and `mean_absolute_error_B`.
The whole-network mean remains available as `result.ts.mean_abs_error`.
The plotted curves are `dashboard.error_history_A[]` and `dashboard.error_history_B[]`.
Target distributions overlay A and B with 50% opacity and the same 20 bin
intervals, recomputed from the combined target range each update. Both plots
refresh after stepping and clear or restore their initial values on Reset.

# Rotating-shape reservoir: 40,000 visual presentations

## Run from the repository root

```julia
include("Julia/RotatingShape/RotatingShape.jl")
```

The main script now defaults to an interactive window; see `Interactive.md`.
Set `interactive=false` in the main script to initialize the visual environment,
optionally show a preview, then run all 40,000 iterations and write the complete
spike history to `Data/` beside the script. One iteration is one frame followed by one reservoir update,
not a four-token sentence. With dt=1/30, the first input is at time 0 and the last
of 40,000 inputs is at time 1333.3 seconds. The object advances by
`angular_speed*dt` per iteration (3 degrees with the defaults).

For explicit control without running the main script:

```julia
include("Julia/RotatingShape/RotatingShape_Functions.jl")
using .RotatingShapeEnvironment
include("Julia/RotatingShape/RotatingShape_Reservoir.jl")
using .RotatingShapeReservoir

env = initialize_environment(;N=64,angular_speed=pi/2,dt=1/30)
result = run_rotating_model(env;
    iterations=40000,
    nnodes=100,
    subnetworks=1,
    n_subnet_edges=10,
    p_link=.1,
    sensory_fraction=.1,
    input_p_link=.1,
    input_amp=5.,
    lrate_wmat=1.,
    lrate_targ=.01,
    seed=1,
    input_seed=2)
model = result.model
```

To avoid saving, use `save=false`. To save to another directory, pass
`output_dir="/absolute/path/to/run"`. Reusing an output directory replaces its
CSV files. Progress prints every 1,000 frames; change `progress_every` or use zero
to disable it. The input environment is copied for a run; a preview or an earlier
run does not advance it. Its current `env.step` sets the run's starting time.

## Binary sensory encoding and hemisphere-specific edges

The input layer contains exactly N² nodes. Each takes the occupancy value of one
pixel: BLACK=1, WHITE=0. The earlier `frame.pixels` intensity convention remains
black=0, white=1 for plotting; reservoir input explicitly uses `frame.object_mask`.

The input vector is column-major:

```text
pixel_index = row + (column-1)*N
```

Thus all pixels in columns 1:N÷2 occupy input indices 1:N²÷2 and belong to the
left visual field. The remaining columns belong to the right visual field.

- Left pixels connect only to reservoir A, nodes 1:nnodes÷2.
- Right pixels connect only to reservoir B, nodes nnodes÷2+1:nnodes.
- Randomly select `round(Int, sensory_fraction * nnodes/2)` distinct sensory
  nodes in each half, once at initialization. The default fraction is .1.
  For small halves rounding can select zero nodes. Other nodes receive no direct input.
- Each allowed pixel-to-sensory-node pair is connected independently with probability .1
  by default (`input_p_link`). This is not a requirement for exactly 10% per node.
- Forbidden cross-hemisphere input edges are always zero.
- Each existing input edge has weight `input_amp`. The raw pixel value stays 0/1.
- Input weights are fixed; the language model's learning adjusts recurrent
  weights and targets. There is no injected sensory noise.

`model.input_wmat` has shape `(N²,nnodes)` and is stored sparsely. At N=64 there
are 4,096 pixel inputs. With 100 reservoir nodes and the defaults, five nodes
in each half are sensory nodes, giving 20,480 eligible pixel-to-node pairs and
about 2,048 expected edges. The graph and selected nodes are fixed for the run.
`model.params.sensory_nodes` records the selected indices; `sensory_nodes.csv`
saves these even if a selected node happens to receive no edges.

The projection sums contributions, with no division by the number of active
pixels. As a result, pixel resolution, object size and `input_amp` all affect
reservoir input magnitude. Defaults preserve the language model's amplitude and
learning settings from the current main script; tune `input_amp` explicitly if
a different total drive is desired.

## Recurrent reservoir and learning

The model reuses the actual language model initialization, activation and learning
functions by including `../Language/Language_Functions.jl` in a nested module.
It does not edit or replace a language module loaded in Main. Keep this sibling
folder in place when moving the visual model.

With the same parameters and seed, the initial recurrent graph and weights
exactly match the language model: two equal halves, internal probability
`p_link`, no self-connections, and exactly `n_subnet_edges` distinct directed
cross-half edges in total (A->B plus B->A). Recurrent weights start at N(0,1).
Pixel projections are sampled separately using `input_seed`, so they do not
change recurrent initialization. The default is `subnetworks=1` and 10 cross
edges. Setting `subnetworks=0` preserves the language homogeneous graph option;
left/right input projection restrictions still apply.

Activation retention, twice-target spike thresholds, one threshold subtraction
per step, negative-activation clipping, error-based local weight updates, and
target adaptation all use the language functions. Cross-hemisphere input edges
are absent, but recurrent cross-subnetwork connections can transmit activity
between hemispheres later.

## Access and saved data

```julia
result.ts.spikes[t, :]    # all reservoir spikes after input frame t
result.ts.time[t]         # time of that frame
result.ts.angle[t]        # its unwrapped orientation in radians
result.ts.black_left[t]   # object-pixel count in left field
result.ts.black_right[t]  # object-pixel count in right field
reshape(model.inputs,N,N) # most recently presented binary occupancy image
```

Spike history is an `iterations × nnodes` UInt8 matrix: every entry is 0 or 1.
All 40,000 frames are saved; no warmup or downsampling is applied. First-half
columns are A and second-half columns are B. The full history is only 4 MB at
40,000×100 in memory. Full per-frame images and per-node activation/target
histories are not stored; frame time, angle, visual parameters and polygon
vertices make the visual stimulus reconstructible.

| File | Contents |
|---|---|
| `spikes.csv` | 40,000 data rows × 100 reservoir-node columns |
| `timesteps.csv` | Matching iteration, input time, angle and left/right pixel counts |
| `reservoir_summary.csv` | Mean activation, signed error, target and spike count per step |
| `input_edges.csv` | Pixel index/row/column, destination reservoir node and fixed weight |
| `link_mat.csv` | Recurrent structural adjacency, rows=source and columns=destination |
| `start_wmat.csv`, `end_wmat.csv` | Initial and trained recurrent weights |
| `final_targets.csv` | Targets after the final update |
| `shape_vertices.csv` | Centered/scaled polygon actually used in the run |
| `run_parameters.txt` | Parameters, seeds, timing and encoding conventions |

CSV files have headers and no extra row-index column. Row t in spikes and
metadata refers to the same frame. `save_run(result,dir)` exports an existing run
without rerunning training. The included data is a verified 40,000-frame run
with the current L shape (upper vertices at .80 before centering/scaling).

## Validation

The 28 visual-environment checks and 34 reservoir/projection checks all passed.
These cover forbidden input edges, allowed-edge sampling, flattening/encoding,
exact shared recurrent initialization, activation update agreement, reproducible
runs, time/angle pairing, target floors, sparse storage, fixed input weights,
immutable caller environment, CSV export and invalid parameters.

The full 40,000-frame / 100-node / 64×64-input run was executed and verified:
spikes are binary, all 40,000 rows are present, weights and summary values stay
finite, no nonexistent recurrent edges acquire weights, and forbidden input
blocks remain zero. Validation used Julia 1.9.3 with the repository environment.

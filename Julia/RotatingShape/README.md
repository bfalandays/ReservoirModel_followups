# Rotating asymmetric object: pixel-based visual environment

The visual environment contains a filled, asymmetric L with unequal arms. Its
area centroid sits at the square center and rotates at constant angular speed.
The main script now also runs a two-subnetwork allostatic reservoir with
hemisphere-restricted pixel inputs; see `Reservoir.md` for full model details.

## Run

From the repository root, with its existing Julia environment active:

```julia
include("Julia/RotatingShape/RotatingShape.jl")
```

The main script now defaults to an interactive GLMakie window with a live
network/spike panel, rotating-stimulus panel, and Run/Pause/Step/Reset/Save controls.
It starts paused and records up to 40,000 steps. See `Interactive.md` for details.
Set `interactive=false` for the original fast batch run and CSV export to `Data/`. `RotatingShape.jl` contains the parameters;
`RotatingShape_Functions.jl` defines the visual environment and
`RotatingShape_Reservoir.jl` defines reservoir integration.
The existing CairoMakie dependency is used. Validation used Julia 1.9.3 with the
repository's current manifest; no additional packages were installed.

To load just the functions and create your own environment:

```julia
include("Julia/RotatingShape/RotatingShape_Functions.jl")
using .RotatingShapeEnvironment

env = initialize_environment(;
    N=64,
    side_length=2.0,
    object_radius=.65,
    angular_speed=pi/2,
    dt=1/30,
    theta0=0.)

fig = plot_frame(env)
display(fig)
record_rotation(env;duration=4.,framerate=30)
```

## Parameters and rotation

| Parameter | Meaning |
|---|---|
| `N` | Even number of pixels per side; the image contains N² pixels |
| `side_length` | Physical side length of the square, centered at (0,0) |
| `object_radius` | Distance from rotation center to furthest object vertex |
| `angular_speed` | Radians per second; positive is counterclockwise |
| `dt` | Seconds per simulation step |
| `theta0` | Initial angle, in radians |
| `vertices` | Optional M×2 simple polygon replacing the default L |

At time t, orientation is `theta0 + angular_speed*t`. The default `pi/2` is
90 degrees per second: one revolution takes four seconds. Negative speed rotates
clockwise; zero speed gives a stationary object. At each simulation step the
angle changes by `angular_speed*dt`. Time is computed from an integer step count,
rather than repeatedly adding dt, to avoid accumulated time drift.

The polygon is translated to its area centroid and scaled to `object_radius`.
Requiring `object_radius < side_length/2` keeps it inside the square at all angles.
Thus its center stays fixed, without orbital translation. Custom polygons should
be simple (non-self-intersecting) and have nonzero area. Vertex coordinates set
shape proportions; centering/scaling are then applied automatically.

## Pixel values and visual halves

```julia
frame = render_frame(env)
frame.pixels       # N×N Float64: 0.0 = black object, 1.0 = white background
frame.object_mask  # N×N Bool: true = object, false = background
frame.time
frame.angle        # unwrapped orientation in radians
```

Matrices use `[row,column]`: the top-left image pixel is `[1,1]`, rows run down,
and columns run right. World coordinates use x rightward and y upward. Pixel
centers are sampled: a pixel is black if its center lies in/on the rotated
polygon. The polygon is rasterized independently at every angle, so no gray
interpolation pixels appear. At finite resolution the rasterized area and shape
boundary can vary slightly with orientation; the underlying polygon does not.

```julia
halves = visual_hemispheres(frame)
halves.left   # N×(N÷2): columns 1:N÷2
halves.right  # N×(N÷2): columns N÷2+1:N
```

N must be even so the two halves are equal. No pixel center lies on x=0. The
orange dashed line marks the division in plots only; it is absent from input
arrays. The optional display grid is also an overlay. Hemisphere arrays are
views: changing their contents changes `frame.pixels` but not the separate
occupancy mask. Treat frame data as read-only; render another frame to reset it.
Both intensity and occupancy are exposed so the later reservoir projection can
choose explicitly whether black object occupancy or light intensity drives it.

## Advancing and plotting

```julia
frame = step_environment!(env)           # advance one dt
frame = step_environment!(env;nsteps=10) # advance ten more dt
fig = plot_frame(env)
display(fig)
```

Or sample any nonnegative time without changing env:

```julia
frame = frame_at_time(env,1.5)
fig = plot_frame(env;time=1.5,show_split=true,show_grid=true)
sequence = plot_rotation_sequence(env;times=collect(0:.5:3.5))
```

Static plots are snapshots; call the plotting function again after advancing.
Animation samples `t=0` through the requested duration at `1/framerate` intervals
and does not modify env. Frame rate controls playback sampling and is independent
of simulation dt; both use the same angular-speed formula. To save MP4 instead:

```julia
record_rotation(env;duration=4.,framerate=30,
    output_path="/absolute/path/RotatingShape.mp4")
```

Use `show_split=false` for a purely black-and-white visualization. Plot functions
accept `output_path=nothing` to avoid saving; animation requires an output path.
The default output directory is `Figures/` beside the source files.

## Validation

All 28 checks passed: binary intensities and occupancy, frame size, one-turn
periodicity, distinct orientations, clockwise and stationary cases, constant
angular increments, simulation-time indexing, centroid and radius, containment,
hemisphere indexing, and invalid parameters. Static plots, eight snapshots, and
a full-revolution GIF were rendered and inspected. Animation generation was also
checked to leave simulation state unchanged.

Interactive runs save in separate numbered directories beneath
`Data/Interactive/`. Existing batch data is preserved. `Interactive.md` explains
the new controls and how to take a result snapshot for analysis.

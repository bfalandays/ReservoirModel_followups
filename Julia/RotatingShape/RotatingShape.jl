include(joinpath(@__DIR__,"./Julia/RotatingShape/RotatingShape_Functions.jl"))
using .RotatingShapeEnvironment
include(joinpath(@__DIR__,"./Julia/RotatingShape/RotatingShape_Reservoir.jl"))
using .RotatingShapeReservoir

## Visual environment
N = 64
side_length = 2.0
object_radius = .65
angular_speed = pi/2      # radians/second; positive = counterclockwise
dt = 1/30                # one model iteration presents one frame at this interval
theta0 = 0.0

## Reservoir: same two-subnetwork construction as the language model
iterations = 20000
nnodes = 250
subnetworks = 1
n_subnet_edges = 10       # total directed A->B and B->A connections
p_link = .1              # recurrent connection probability within each half
sensory_fraction = .1    # fraction of each half selected as sensory nodes
input_p_link = .1        # pixel->same-side sensory node connection probability
input_amp = 1.0         # weight of each existing pixel->reservoir connection
leak = .25
lrate_wmat = 1.0
lrate_targ = .01
targ_min = 1.0
acts_neg = 0
learn_on = true
seed = 2
input_seed = seed+1

## Display mode: interactive GLMakie window, or the original fast batch run.
interactive = false#true
playback_fps = 30
start_running = false  # press Run to start; Step advances exactly one iteration

## Optional batch previews
make_preview = true
make_animation = false
animation_duration = 4.0
animation_framerate = 30

env = initialize_environment(;N,side_length,object_radius,angular_speed,dt,theta0)
if !interactive && make_preview
    fig = plot_frame(env)
    display(fig)
end
if !interactive && make_animation
    record_rotation(env;duration=animation_duration,framerate=animation_framerate)
end

## Run interactively by default, recording every step as before.
if interactive
    include(joinpath(@__DIR__,"RotatingShape_Interactive.jl"))
    using .RotatingShapeInteractive
    dashboard = launch_rotating_dashboard(env;iterations,nnodes,subnetworks,n_subnet_edges,
        p_link,sensory_fraction,input_p_link,input_amp,leak,lrate_wmat,lrate_targ,targ_min,
        acts_neg,learn_on,seed,input_seed,playback_fps,autoplay=start_running,
        output_dir=joinpath(@__DIR__,"Data","Interactive"))
    fig = dashboard.fig
    model = dashboard.state.model
    # Keep a directly launched terminal script alive until the window closes.
    if abspath(PROGRAM_FILE)==abspath(@__FILE__)
        wait(dashboard.screen)
    end
else
    result = run_rotating_model(env;iterations,nnodes,subnetworks,n_subnet_edges,
        p_link,sensory_fraction,input_p_link,input_amp,leak,lrate_wmat,lrate_targ,targ_min,
        acts_neg,learn_on,seed,input_seed,
        output_dir=joinpath(@__DIR__,"Data"),progress_every=1000)
    model = result.model
    println("Saved full spike history and run metadata to: $(result.output_dir)")
end

# While paused: result = interactive_result(dashboard)
# result.ts.spikes[t,:]  # spike vector after input frame t
# result.ts.time[t]     # time of that input frame
# result.ts.angle[t]    # its orientation in radians

include("/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/RotatingShape_Drift.jl")
using .RotatingShapeDrift

##
using CairoMakie

iterations = eachindex(result.ts.mean_abs_error_A)

fig_errors = Figure(resolution=(900, 500))
ax = Axis(fig_errors[1, 1];
    xlabel="Model iteration",
    ylabel="Mean absolute error",
    title="Mean absolute error over time")

lines!(ax, iterations, result.ts.mean_abs_error_A;
    color=:royalblue, linewidth=2, label="Subnetwork A")

lines!(ax, iterations, result.ts.mean_abs_error_B;
    color=:seagreen, linewidth=2, label="Subnetwork B")

axislegend(ax; position=:rt)
display(fig_errors)

save(
    "/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/Mean_absolute_error.png",
    fig_errors
)

## PC1 by orientation, separated by subnetwork
p = plot_orientation_pc1(
    "/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Data";
    window=5000:6000)

display(p.fig)

## PC1 vs PC2 for full network
p2 = plot_spike_pc12(result;
    window=5000:6000,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/PC1_PC2_5000_6000.png")

display(p2.fig)

## PC1 vs PC2 for subnetworks
p3 = plot_subnetwork_pc12(result;
    window=5000:6000,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/PC1_PC2_subnetworks_5000_6000.png")

display(p3.fig)

## 3d Plot: PC1-A vs PC1-B vs time
p4 = plot_subnetwork_pc1_time(result;
    window=5000:6000,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/PC1_subnetworks_time_5000_6000.png")

display(p4.fig)

## Interactive 3d Plot: PC1-A vs PC1-B vs time
p4 = plot_subnetwork_pc1_time(result;
    window=5000:6000,
    interactive=true)

## PC1 vs PC2 separated by subnet, selecting spikes only when shape is at a particular orientation
p5 = plot_horizontal_subnetwork_pc12(result;
    window=5000:6000,
    target_angle_degrees=90.,
    tolerance_degrees=1.,
    savepath="/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShape/Figures/Horizontal_PC1_PC2.png")

display(p5.fig)

###


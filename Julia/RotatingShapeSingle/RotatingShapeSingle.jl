# Launch this version in a fresh Julia session using the existing repository environment.

# Note: for robustness check, should run a second time with different seed and check cross-orientation cossim

include(joinpath(@__DIR__,"RotatingShapeSingle_Functions.jl"))
using .RotatingShapeSingleModel

batch_dir = "/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShapeSingle/Data/Ensemble/ensemble_100_seeds"

## Visual environment
N = 64
side_length = 2.0
object_radius = .65
dt = 1/30
frames_per_rotation = 120  # 0°, 45°, ..., 315°, then repeat; nothing restores 3°/frame
angular_speed = isnothing(frames_per_rotation) ? pi/2 : 2pi/(frames_per_rotation*dt)
theta0 = 0.0
reverse_rotation = false # true: alternate direction after each block of full turns
rotations_per_direction = 5 # number of full turns in each direction (not image resolution N)

## One homogeneous reservoir; all pixels share the same sensory-node pool
iterations = 48000
nnodes = 250
p_link = .1
sensory_fraction = .1     # 25 randomly selected sensory nodes at nnodes=250
input_p_link = .1         # independent pixel -> selected sensory node probability
input_amp = 1.0
leak = .25
lrate_wmat = 1.0
lrate_targ = .01
targ_min = 1.0
acts_neg = 0
learn_on = true
seed = 2
input_seed = seed+1

## Display and previews
interactive = false
playback_fps = 30
start_running = false
make_preview = true
make_animation = false
animation_duration = 4.0
animation_framerate = 30

env = initialize_environment(;N,side_length,object_radius,angular_speed,dt,theta0,
    reverse_rotation,rotations_per_direction)
if !interactive && make_preview
    display(plot_frame(env))
end
if !interactive && make_animation
    record_rotation(env;duration=animation_duration,framerate=animation_framerate)
end

if interactive
    dashboard = launch_rotating_dashboard(env;iterations,nnodes,p_link,
        sensory_fraction,input_p_link,input_amp,leak,lrate_wmat,lrate_targ,targ_min,
        acts_neg,learn_on,seed,input_seed,playback_fps,autoplay=start_running,
        output_dir=joinpath(@__DIR__,"Data","Interactive"))
    fig = dashboard.fig
    model = dashboard.state.model
    if abspath(PROGRAM_FILE)==abspath(@__FILE__)
        wait(dashboard.screen)
    end
else
    result = run_rotating_model(env;iterations,nnodes,p_link,
        sensory_fraction,input_p_link,input_amp,leak,lrate_wmat,lrate_targ,targ_min,
        acts_neg,learn_on,seed,input_seed,
        output_dir=joinpath(@__DIR__,"Data","Batch"),progress_every=1000)
    model = result.model
    println("Saved spike history, orientations and rotation directions to: $(result.output_dir)")
end

### Running 100 independent sims



# Run these lines to create or resume the eight-frame ensemble:
# new_batch_dir = joinpath(dirname(batch_dir), "ensemble_100_seeds_8_frames")
# env_settings = ensemble_environment(; frames_per_rotation=8)
# run_seed_ensemble(new_batch_dir; environment=env_settings, seeds=1:100, iterations=48000)
# batch_dir = new_batch_dir

run_seed_ensemble(batch_dir;
    environment=ensemble_environment(; frames_per_rotation=120),
    seeds=1:100,
    iterations=48000)

status = ensemble_status(batch_dir)
length(status.completed)

result = load_ensemble_run(batch_dir, 1)
result.model
result.ts.spikes

###


angles = 0:45:315
out = joinpath(batch_dir, "Analysis", "Orientations_All8")

p_lag = plot_ensemble_correlation_lag(batch_dir;
    orientations=angles, output_dir=joinpath(out, "CorrelationLag"))

p_robustness = plot_ensemble_correlation_robustness(batch_dir;
    orientations=angles, output_dir=joinpath(out, "AcrossSeedRobustness"))

p_pairs = plot_ensemble_random_pair_control(batch_dir;
    orientations=angles, pairing_seed=1, output_dir=joinpath(out, "RandomPairControl"))

display(p_lag.fig)
display(p_robustness.fig)
display(p_pairs.fig)

###

using CairoMakie

a = p_pairs.analysis
pair_number = 2                 # change this to any number from 1 to 50
seed_a, seed_b = a.seed_pairs[pair_number]

fig_pair = Figure(resolution=(950, 550))
ax = Axis(fig_pair[1, 1];
    title="Pair $pair_number: seeds $seed_a and $seed_b",
    xlabel="Lag (bins)",
    ylabel="Pearson correlation")

colors = (:royalblue, :purple, :seagreen)
for (k, name) in enumerate(a.names)
    lines!(ax, a.within.lags, a.pair_values[pair_number, :, k];
        color=colors[k], linewidth=2.5, label=name)
end

hlines!(ax, [0.0]; color=:gray, linestyle=:dot)
axislegend(ax; position=:rt)
display(fig_pair)


# Export the currently selected batch; do not reset batch_dir here.
output_path = joinpath(batch_dir, "Exports", "ensemble_spikes.csv")
if isfile(output_path)
    println("Using existing spike export: $output_path")
else
    export_ensemble_spikes(batch_dir, output_path)
end

##

run_id = 1
angles = (45, 225)   # first angle on x, second angle on y

run_result = load_ensemble_run(batch_dir, run_id)

p_run = RotatingShapeSingleModel.plot_orientation_correlation_heatmap(
    run_result;
    orientations=angles,
    run_label=run_id,
    warmup=2400,
    bin_size=1200,
    save_outputs=false
)

display(p_run.fig)

#

# Reuse run_result from the previous plot, or load another run:
run_result = load_ensemble_run(batch_dir, run_id)

p_auto = RotatingShapeSingleModel.plot_binned_orientation_correlations(
    run_result;
    orientations=angles,
    warmup=2400,
    bin_size=1200,
    save_outputs=false
)

display(p_auto.fig)

using CairoMakie

# Cross-angle heatmap
p_run.axis.xgridvisible[] = false
p_run.axis.ygridvisible[] = false

# Both same-angle heatmaps
for ax in p_auto.fig.content
    ax isa Axis || continue
    ax.xgridvisible[] = false
    ax.ygridvisible[] = false
end

display(p_run.fig)
display(p_auto.fig)

save_dir = "/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/RotatingShapeSingle/Figures"

save(joinpath(save_dir, "cross_angle_heatmap.png"), p_run.fig; px_per_unit=2)
save(joinpath(save_dir, "same_angle_heatmaps.png"), p_auto.fig; px_per_unit=2)
# Combined figure using the run and two angles selected above:
p_heatmaps = RotatingShapeSingleModel.plot_orientation_correlation_triptych(
    p_auto.analysis; orientations=angles, fontsize=22)
display(p_heatmaps.fig)
# save(joinpath(save_dir, "orientation_correlation_heatmaps.pdf"), p_heatmaps.fig)

##
p_heatmaps = RotatingShapeSingleModel.plot_orientation_correlation_triptych(
    p_auto.analysis;
    orientations = (45, 225),
    fontsize = 22,
)
display(p_heatmaps.fig)

# Optional PDF export
save(joinpath(save_dir, "orientation_correlation_heatmaps.pdf"), p_heatmaps.fig)
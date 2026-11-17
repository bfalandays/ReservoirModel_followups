include(joinpath(@__DIR__, "./Julia/Language/Language_Functions.jl"))
include("Julia/Language/Language_Plotting.jl")

using .AllostaticLanguage

## Parameters: notebook values, with leak expressed as fraction LOST.
iterations = 40000                       # four tokens per sentence
nnodes = 100
p_link = .1
subnetworks = 1                         # 1: two equal subnetworks; 0: homogeneous reservoir
n_subnet_edges = 10                      # total directed connections across the two halves
leak = .25                              # Python leak=.75 retains 75%
lrate_wmat = 1.0
lrate_targ = .01
targ_min = 1.
input_amp = 5.
acts_neg = 0                            # notebook clips negative activation
learn_on = true
seed = 1

model = initialize_model(;
    nnodes=nnodes,
    subnetworks=subnetworks,
    n_subnet_edges=n_subnet_edges)

fig = plot_network_structure(model)
display(fig)

## Run model and save the initial/final sentence windows.
result = run_model(; iterations, nnodes, p_link, leak, lrate_wmat,
    lrate_targ, targ_min, input_amp, acts_neg, learn_on, seed, subnetworks, n_subnet_edges)
model = result.model

# sum(result.ts.spikes, dims=2)

# ## Completion probes: learning stays ON, as in the notebook.
# # carry_targets=true preserves the notebook's order-dependent target carry-over.
# # For independent probes use carry_targets=false; for frozen probes also set
# # learn_on=false. Every probe starts with the trained weights, acts and spikes.
# probes = run_probes(result; learn_on=true, carry_targets=true)

# ## Fading memory: one dog cue, then zeros, stopping at first silent time step.
# # Targets come from the final probe, as in the notebook. Previous spikes are
# # refreshed on every step (fixes the stale prev_spikes variable in Python).
# fading = fading_memory(result; targets=probes[end].model.targets,
#     learn_on=true, max_steps=10000)
# println("Fading probe: $(fading.steps) steps; first silence reached: $(fading.stopped)")

# ## CSV output (no external packages required).
# data_dir = joinpath(@__DIR__, "Data")
# save_data(result, probes, fading, data_dir)

# ## Optional figures, using the CairoMakie dependency in your repository.
# # include(joinpath(@__DIR__, "Language_Plotting.jl"))
# # plot_results(result, probes; output_dir=joinpath(@__DIR__, "Figures"))

warmup_instances = 500                 # discard this many initial instances of EACH input key

spike_distances = input_hamming_distances(result)
fig = plot_hamming_heatmaps(spike_distances; skip_instances=warmup_instances)
display(fig)

nback = last_row_nback(spike_distances; skip_instances=warmup_instances)
fig = plot_nback_distances(spike_distances; skip_instances=warmup_instances)
display(fig)

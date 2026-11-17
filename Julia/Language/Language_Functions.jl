module AllostaticLanguage

using Random, LinearAlgebra, Statistics, DelimitedFiles
export initialize_model, get_acts, learning, choose_next_input, model_step!,
       run_model, run_probes, fading_memory, save_data, correlation_matrix, input_hamming_distances, plot_hamming_heatmaps, last_row_nback, plot_nback_distances

const input_keys = ["man_s", "dog_s", "walks_v", "bites_v", "dog_o", "man_o", "_"]
const input_dict = Dict(
    "man_s" => [1.,0,0,0,0], "dog_s" => [0.,1,0,0,0],
    "walks_v" => [0.,0,1,0,0], "bites_v" => [0.,0,0,1,0],
    "dog_o" => [0.,1,0,0,0], "man_o" => [1.,0,0,0,0],
    "_" => [0.,0,0,0,1])

function grammar_matrix(; deterministic=false)
    grammarmat = zeros(7, 7)
    p = deterministic ? 1.0 : .75
    grammarmat[1,3:4] = [p, 1-p]
    grammarmat[2,3:4] = [1-p, p]
    grammarmat[3,5:6] = [p, 1-p]
    grammarmat[4,5:6] = [1-p, p]
    grammarmat[5:6,7] .= 1
    grammarmat[7,1:2] .= .5
    return grammarmat
end

# Rows are presynaptic/source nodes; columns are postsynaptic/destination nodes.
mutable struct LanguageModel{R<:AbstractRNG,P}
    rng::R
    params::P
    link_mat::Matrix{Float64}
    input_wmat::Matrix{Float64}
    wmat::Matrix{Float64}
    acts::Vector{Float64}
    spikes::Vector{Float64}
    targets::Vector{Float64}
    errors::Vector{Float64}
    prev_spikes::Vector{Float64}
    inputs::Vector{Float64}
    input_key::String
    n::Int
    mean_act::Float64
    mean_err::Float64
    mean_targ::Float64
end

function initialize_model(; nnodes=100, p_link=.1, leak=.25, lrate_wmat=.1,
                          lrate_targ=.01, targ_min=1., input_amp=5.,
                          learn_on=true, acts_neg=0, seed=1,
                          subnetworks=0, n_subnet_edges=0)
    nnodes > 0 || throw(ArgumentError("nnodes must be positive"))
    0 <= p_link <= 1 || throw(ArgumentError("p_link must lie in [0,1]"))
    0 <= leak <= 1 || throw(ArgumentError("leak is the fraction lost, in [0,1]"))
    targ_min > 0 || throw(ArgumentError("targ_min must be positive"))
    lrate_wmat >= 0 && lrate_targ >= 0 || throw(ArgumentError("learning rates must be nonnegative"))
    subnetworks in (0,1) || throw(ArgumentError("subnetworks must be 0 or 1"))
    (n_subnet_edges isa Integer && n_subnet_edges >= 0) ||
        throw(ArgumentError("n_subnet_edges must be a nonnegative integer"))
    if subnetworks == 1
        iseven(nnodes) || throw(ArgumentError("subnetworks=1 requires even nnodes"))
        n_subnet_edges <= 2*(nnodes ÷ 2)^2 ||
            throw(ArgumentError("n_subnet_edges exceeds the number of possible directed cross-subnetwork edges"))
    end
    rng = MersenneTwister(seed)
    params = (; nnodes, p_link, leak, lrate_wmat, lrate_targ,
              targ_min, input_amp, learn_on, acts_neg, seed, subnetworks, n_subnet_edges)
    input_wmat = input_amp .* Float64.(rand(rng, 5, nnodes) .< p_link)
    link_mat = Float64.(rand(rng, nnodes, nnodes) .< p_link)
    link_mat[diagind(link_mat)] .= 0
    if subnetworks == 1
        half = nnodes ÷ 2
        # A = 1:half; B = half+1:nnodes. p_link applies only within each half.
        link_mat[1:half,half+1:nnodes] .= 0
        link_mat[half+1:nnodes,1:half] .= 0
        # Sample distinct directed edges uniformly from both off-diagonal blocks.
        # Each selected index represents one A->B or B->A connection; reciprocal
        # pairs are not forced, and count as two edges when both are selected.
        candidates = randperm(rng,2*half^2)
        for edge in @view candidates[1:n_subnet_edges]
            offset = mod(edge-1,half^2)
            source = mod(offset,half)+1
            destination = div(offset,half)+1
            if edge <= half^2
                link_mat[source,half+destination] = 1
            else
                link_mat[half+source,destination] = 1
            end
        end
    end
    # Preserve notebook N(0,1) initialization, rather than the sensorimotor variants.
    wmat = link_mat .* randn(rng, nnodes, nnodes)
    return LanguageModel(rng, params, link_mat, input_wmat, wmat,
        zeros(nnodes), zeros(nnodes), fill(Float64(targ_min), nnodes),
        zeros(nnodes), zeros(nnodes), zeros(5), "_", 0, 0., 0., Float64(targ_min))
end

function choose_next_input(rng, input_key, grammarmat)
    row = findfirst(==(input_key), input_keys)
    isnothing(row) && throw(ArgumentError("Unknown token: $input_key"))
    probs = @view grammarmat[row,:]
    isapprox(sum(probs), 1.) && all(probs .>= 0) ||
        throw(ArgumentError("Grammar row must be a probability distribution"))
    index = min(searchsortedlast(cumsum(probs), rand(rng)) + 1, length(input_keys))
    key = input_keys[index]
    return copy(input_dict[key]), key, probs[index]
end

function get_acts(acts, leak, spikes, wmat, input, input_wmat, targets; acts_neg=0)
    prev_spikes = copy(spikes)
    acts = acts .* (1-leak) .+ transpose(input_wmat)*input .+ transpose(wmat)*prev_spikes
    thresholds = targets .* 2
    spikes = Float64.(acts .>= thresholds)
    acts .-= spikes .* thresholds
    # Python clips after the spike/reset step. With positive targets, clipping
    # before thresholding (as in the other Julia models) gives the same result.
    acts_neg == 0 && (acts = max.(acts, 0.))
    errors = acts .- targets
    return acts, spikes, errors, prev_spikes
end

function learning(learn_on, link_mat, spikes, prev_spikes, errors, wmat, targets;
                  lrate_wmat=.1, lrate_targ=.01, targ_min=1.)
    if learn_on == 1
        active_neighbors = abs.(link_mat) .* (prev_spikes .> 0)
        counts = sum(active_neighbors; dims=1)
        # Avoid 0/0 for destinations with no active incoming connections.
        d_wmat = active_neighbors .* transpose(errors) ./ max.(counts, 1.)
        wmat .-= lrate_wmat .* d_wmat
        targets .+= lrate_targ .* errors
        targets .= max.(targets, targ_min)
    end
    return wmat, targets
end

function model_step!(model; input=model.inputs, learn_on=model.params.learn_on)
    p = model.params
    model.inputs .= input
    model.acts, model.spikes, model.errors, model.prev_spikes =
        get_acts(model.acts, p.leak, model.spikes, model.wmat,
                 model.inputs, model.input_wmat, model.targets; acts_neg=p.acts_neg)
    learning(learn_on, model.link_mat, model.spikes, model.prev_spikes,
             model.errors, model.wmat, model.targets;
             lrate_wmat=p.lrate_wmat, lrate_targ=p.lrate_targ, targ_min=p.targ_min)
    model.mean_act = mean(model.acts)
    model.mean_err = mean(model.errors)
    model.mean_targ = mean(model.targets)
    model.n += 1
    return model
end

function history(nsteps, nnodes)
    return (; spikes=zeros(nsteps,nnodes), acts=zeros(nsteps,nnodes),
        targets=zeros(nsteps,nnodes), errors=zeros(nsteps,nnodes),
        stream=fill("",nsteps), tprobs=fill(NaN,nsteps),
        iteration=zeros(Int,nsteps), time=zeros(Int,nsteps),
        forced=falses(nsteps), input_present=falses(nsteps))
end

function record!(ts, row, model; iteration=0, time=row, tprob=NaN,
                 forced=false, input_present=true, label=model.input_key)
    for field in (:spikes, :acts, :targets, :errors)
        getproperty(ts,field)[row,:] .= getproperty(model,field)
    end
    ts.stream[row] = label
    ts.tprobs[row] = tprob
    ts.iteration[row] = iteration
    ts.time[row] = time
    ts.forced[row] = forced
    ts.input_present[row] = input_present
end

function slice_history(ts, rows)
    return (; (name => (value isa AbstractMatrix ? value[rows,:] : value[rows])
               for (name,value) in pairs(ts))...)
end

function run_model(; iterations=1000, verbose=true, alternation=:notebook, kwargs...)
    # The original late alternation assumes that the final 100 sentences have
    # already begun when i >= iterations-8. Require enough sentences for that.
    iterations >= 10 || throw(ArgumentError("Use at least 10 sentence iterations"))
    alternation in (:notebook, :previous_sentence) ||
        throw(ArgumentError("alternation must be :notebook or :previous_sentence"))
    model = initialize_model(; kwargs...)
    start_wmat = copy(model.wmat)
    grammarmat = grammar_matrix()
    ts = history(4*iterations, model.params.nnodes)
    for i in 1:iterations, t in 1:4
        previous_key = model.input_key
        input, input_key, tprob = choose_next_input(model.rng, previous_key, grammarmat)
        forced = false
        if t == 1 && (2 <= i <= 8 || i >= iterations-8)
            # Python start_stream stops growing after sentence 4, so its
            # early forcing uses sentence 4 repeatedly for sentences 5:8.
            subject_iteration = (alternation == :notebook && 2 <= i <= 8) ? min(i-1,4) : i-1
            previous_subject = ts.stream[4*(subject_iteration-1)+1]
            input_key = previous_subject == "man_s" ? "dog_s" : "man_s"
            input = copy(input_dict[input_key])
            # Record probability for the delivered token, rather than the
            # discarded random subject in the notebook (both are .5 here).
            tprob = grammarmat[findfirst(==(previous_key), input_keys),
                                  findfirst(==(input_key), input_keys)]
            forced = true
        end
        model.input_key = input_key
        model_step!(model; input)
        record!(ts,4*(i-1)+t,model; iteration=i,time=t,tprob,forced)
        if verbose && (i == 1 || i % 1000 == 0)
            println("iteration: $i ; time: $t ; ME: $(model.mean_err)")
        end
    end
    return (; model, iterations, alternation, start_wmat, end_wmat=copy(model.wmat), ts,
        start=slice_history(ts,1:16),
        finish=slice_history(ts,max(1,4*(iterations-100)+1):4*iterations))
end

function run_probes(result; learn_on=true, carry_targets=true)
    # Notebook order: man / one cue, man / two cues, dog / one cue, dog / two cues.
    # Weights, activation and spikes reset per probe; targets optionally carry.
    probes = NamedTuple[]
    targets = copy(result.model.targets)
    grammarmat = grammar_matrix(; deterministic=true)
    for first_input in 1:2, input_num in 1:2
        model = deepcopy(result.model)
        model.targets .= carry_targets ? targets : result.model.targets
        ts = history(4, model.params.nnodes)
        key = input_keys[first_input]
        for t in 1:4
            if t > 1
                _, key, _ = choose_next_input(model.rng, key, grammarmat)
            end
            model.input_key = key
            present = t <= input_num
            input = present ? copy(input_dict[key]) : zeros(5)
            model_step!(model; input, learn_on)
            record!(ts,t,model; input_present=present, label=present ? key : "NA")
        end
        targets .= model.targets
        push!(probes, (; first_input, input_num, learn_on, carry_targets, ts, model))
    end
    return probes
end

function fading_memory(result; targets=result.model.targets, learn_on=true, max_steps=10000)
    max_steps >= 1 || throw(ArgumentError("max_steps must be positive"))
    model = deepcopy(result.model)
    model.targets .= targets
    ts = history(max_steps, model.params.nnodes)
    for t in 1:max_steps
        input = t == 1 ? copy(input_dict["dog_s"]) : zeros(5)
        model.input_key = t == 1 ? "dog_s" : "NA"
        model_step!(model; input, learn_on)
        record!(ts,t,model; input_present=t==1)
        if all(iszero, model.spikes)
            return (; ts=slice_history(ts,1:t), model, learn_on, max_steps, stopped=true, steps=t)
        end
    end
    return (; ts, model, learn_on, max_steps, stopped=false, steps=max_steps)
end

function correlation_matrix(spikes)
    n = size(spikes,1)
    cor_matrix = fill(NaN,n,n)
    for i in 1:n, j in 1:i
        x, y = @view(spikes[i,:]), @view(spikes[j,:])
        # Pearson correlation is undefined for a constant spike vector.
        if !all(==(x[1]),x) && !all(==(y[1]),y)
            cor_matrix[i,j] = cor(x,y)
        end
    end
    return cor_matrix
end

function write_csv(path, columns, data)
    open(path,"w") do io
        writedlm(io,permutedims(columns),',')
        writedlm(io,data,',')
    end
end

function save_history(dir, prefix, ts)
    names = ["node_$j" for j in axes(ts.spikes,2)]
    for field in (:spikes,:acts,:targets,:errors)
        # Keep notebook's *_errors.csv as mean signed error; also save all nodes.
        suffix = field == :errors ? "node_errors" : string(field)
        write_csv(joinpath(dir,"$(prefix)_$suffix.csv"),names,getproperty(ts,field))
    end
    write_csv(joinpath(dir,"$(prefix)_errors.csv"),["mean_error"],mean(ts.errors;dims=2))
    write_csv(joinpath(dir,"$(prefix)_spikes_sum.csv"),["spikes_sum"],sum(ts.spikes;dims=2))
    write_csv(joinpath(dir,"$(prefix)_stream.csv"),["token"],ts.stream)
    write_csv(joinpath(dir,"$(prefix)_tprobs.csv"),["tprob"],ts.tprobs)
    write_csv(joinpath(dir,"$(prefix)_metadata.csv"),
        ["iteration","time","forced","input_present"],
        hcat(ts.iteration,ts.time,ts.forced,ts.input_present))
end

function save_data(result, probes, fading, dir)
    mkpath(dir)
    open(joinpath(dir,"run_parameters.txt"),"w") do io
        println(io, "julia_version = ", VERSION)
        println(io, "iterations = ", result.iterations)
        println(io, "alternation = ", result.alternation)
        for (key,value) in pairs(result.model.params)
            println(io, key, " = ", value)
        end
        println(io, "probe_learning = ", isempty(probes) ? "none" : probes[1].learn_on)
        println(io, "probe_carry_targets = ", isempty(probes) ? "none" : probes[1].carry_targets)
        println(io, "fading_learning = ", fading.learn_on)
        println(io, "fading_max_steps = ", fading.max_steps)
        println(io, "fading_steps = ", fading.steps)
        println(io, "fading_first_silence_reached = ", fading.stopped)
    end
    save_history(dir,"start",result.start)
    save_history(dir,"end",result.finish)
    write_csv(joinpath(dir,"start_wmat.csv"),["node_$j" for j in axes(result.start_wmat,2)],result.start_wmat)
    write_csv(joinpath(dir,"end_wmat.csv"),["node_$j" for j in axes(result.end_wmat,2)],result.end_wmat)
    for probe in probes
        save_history(dir,"probe_$(input_keys[probe.first_input])_$(probe.input_num)",probe.ts)
    end
    save_history(dir,"fading",fading.ts)
    return dir
end

"""
    group_spikes_by_input(ts; keys=["man_s", "dog_s", "walks_v", "bites_v", "dog_o", "man_o", "_"])

Group a timestep-by-node spike matrix using the corresponding input keys.
Rows within each group retain their original chronological order. `timesteps`
records which original rows were selected. Missing keys produce zero-row matrices.
"""
function group_spikes_by_input(ts; keys=["man_s", "dog_s", "walks_v", "bites_v", "dog_o", "man_o", "_"])
    size(ts.spikes,1) == length(ts.stream) ||
        throw(DimensionMismatch("There must be one input key per spike-vector row"))
    length(unique(keys)) == length(keys) || throw(ArgumentError("keys must be unique"))
    timesteps = Dict(key => findall(==(key), ts.stream) for key in keys)
    spikes = Dict(key => ts.spikes[timesteps[key],:] for key in keys)
    return (; keys=collect(keys), spikes, timesteps, nnodes=size(ts.spikes,2))
end

"""
    hamming_distance_matrix(spikes; normalize=false)

Pairwise Hamming distances between ROWS of a binary spike matrix. By default,
distance counts nodes with different spike values (0:nnodes). With
`normalize=true`, divide by nnodes to return the fraction differing (0:1).
Counts use UInt8 through 255 nodes, widening automatically for larger networks.
Normalized distances retain Float64 storage.
"""
function hamming_distance_matrix(spikes; normalize=false)
    all(x -> x == 0 || x == 1, spikes) ||
        throw(ArgumentError("Hamming spike analysis requires binary 0/1 values"))
    n, nnodes = size(spikes)
    nnodes > 0 || throw(ArgumentError("Spike vectors must contain at least one node"))
    # Counts need only cover 0:nnodes. Use the smallest safe unsigned type.
    count_type = nnodes <= typemax(UInt8) ? UInt8 :
                 nnodes <= typemax(UInt16) ? UInt16 :
                 nnodes <= typemax(UInt32) ? UInt32 : UInt64
    distances = zeros(normalize ? Float64 : count_type, n, n)
    for i in 1:n, j in 1:i-1
        distance = 0
        @inbounds for node in 1:nnodes
            distance += spikes[i,node] != spikes[j,node]
        end
        value = normalize ? distance / nnodes : distance
        distances[i,j] = value
        distances[j,i] = value
    end
    return distances
end

"""
    input_hamming_distances(result; normalize=false)

Return `spikes[key]`, `distances[key]`, and original `timesteps[key]` for each
word input key, using ALL training timesteps in `result.ts`.
The separator "_" is excluded before computing any distance matrices.
"""
function input_hamming_distances(result; normalize=false)
    groups = group_spikes_by_input(result.ts; keys=filter(!=("_"),input_keys))
    distances = Dict(key => hamming_distance_matrix(groups.spikes[key]; normalize)
                     for key in groups.keys)
    iterations = Dict(key => (hasproperty(result.ts,:iteration) ?
        result.ts.iteration[groups.timesteps[key]] : cld.(groups.timesteps[key],4))
        for key in groups.keys)
    return (; groups..., distances, iterations, normalized=normalize)
end

# Load this file in the repository environment (CairoMakie is already a dependency).
using CairoMakie

# Keep cutoff selection and color scaling tied to exactly the displayed pairs.
function _hamming_plot_data(analysis, key; skip_instances=500)
    (skip_instances isa Integer && skip_instances >= 0) ||
        throw(ArgumentError("skip_instances must be a nonnegative integer"))
    ninstances = size(analysis.distances[key],1)
    keep = collect(min(skip_instances,ninstances)+1:ninstances)
    values = Float64.(analysis.distances[key][keep,keep])
    n = size(values,1)
    for i in 1:n, j in i:n
        values[i,j] = NaN  # identity and upper triangle
    end
    displayed = filter(isfinite,vec(values))
    observed_min, observed_max = isempty(displayed) ? (0.,0.) : extrema(displayed)
    return (; values, observed_min, observed_max, keep)
end

"""
    plot_hamming_heatmaps(analysis; skip_instances=500, output_path=..., ncols=3)

Plot each within-input distance matrix with its own color scale and colorbar.
Only occurrences after the first `skip_instances` of each key and strict lower-triangle pairs are
shown. Each scale spans the minimum to the maximum of those displayed pairs. Axes index
chronological occurrences of that input, rather than original training timesteps.
Minimum observed distance is brightest; maximum observed distance is darkest. The identity diagonal
is masked in the plot without changing the underlying distance matrices.
Saves a PNG and returns the Makie Figure. Use output_path=nothing to skip saving.
"""
function plot_hamming_heatmaps(analysis; skip_instances=500,
        output_path=joinpath(@__DIR__,"Figures","Hamming_by_input.png"), ncols=3)
    (skip_instances isa Integer && skip_instances >= 0) ||
        throw(ArgumentError("skip_instances must be a nonnegative integer"))
    ncols >= 1 || throw(ArgumentError("ncols must be positive"))
    plot_keys = filter(!=("_"),analysis.keys)
    nrows = cld(length(plot_keys),ncols)
    fig = Figure(resolution=(530*ncols, 400*nrows))
    label = analysis.normalized ? "Fraction of nodes differing" : "Nodes differing"
    for (index,key) in enumerate(plot_keys)
        row, col = fldmod(index-1,ncols)
        plot_data = _hamming_plot_data(analysis,key; skip_instances)
        plot_distances = plot_data.values
        n = size(plot_distances,1)
        title = key == "_" ? "_ (separator)" : key
        panel = GridLayout()
        fig[row+1,col+1] = panel
        ax = Axis(panel[1,1]; title="$title (n=$n)",
            xlabel="Retained occurrence", ylabel="Retained occurrence",
            aspect=DataAspect(), yreversed=true)
        if n < 2
            text!(ax,.5,.5;text=n == 0 ? "No instances" : "No pairs",align=(:center,:center))
            limits!(ax,0,1,0,1)
        else
            # Explicit cell edges also handle a group containing a single row.
            edges = .5:1:(n+.5)
            observed_min = plot_data.observed_min
            observed_max = plot_data.observed_max
            # Constant distances have no span: render them brightest and show
            # only their observed value on the colorbar.
            maxdistance = observed_max > observed_min ? observed_max : observed_min + 1.
            hm = heatmap!(ax,edges,edges,permutedims(plot_distances);
                colormap=Reverse(:viridis),colorrange=(observed_min,maxdistance),
                nan_color=:transparent,interpolate=false)
            if observed_max == observed_min
                Colorbar(panel[1,2],hm; label, ticks=[observed_min])
            else
                Colorbar(panel[1,2],hm; label)
            end
            limits!(ax,.5,n+.5,n+.5,.5)
        end
    end
    if output_path !== nothing
        mkpath(dirname(output_path))
        save(output_path,fig)
    end
    return fig
end

# Example, after running Language.jl:
# spike_distances = input_hamming_distances(result)
# fig = plot_hamming_heatmaps(spike_distances)
# display(fig)
# man_spikes = spike_distances.spikes["man_s"]
# man_distances = spike_distances.distances["man_s"]
# original_timesteps = spike_distances.timesteps["man_s"]


"""
    last_row_nback(analysis; skip_instances=500, max_nback=nothing)

For each input key, compare its final retained occurrence with earlier retained
occurrences, excluding the first `skip_instances` instances of that key.
Excludes the identity entry. n-back counts occurrences of the SAME
key, not global timesteps. Returns a dictionary of curves and timestep mappings.
"""
function last_row_nback(analysis; skip_instances=500, max_nback=nothing)
    (skip_instances isa Integer && skip_instances >= 0) ||
        throw(ArgumentError("skip_instances must be a nonnegative integer"))
    (max_nback === nothing || (max_nback isa Integer && max_nback >= 1)) ||
        throw(ArgumentError("max_nback must be a positive integer or nothing"))
    curves = Dict{String,NamedTuple}()
    for key in filter(!=("_"),analysis.keys)
        ninstances = size(analysis.distances[key],1)
        keep = collect(min(skip_instances,ninstances)+1:ninstances)
        available = max(length(keep)-1,0)
        n = max_nback === nothing ? available : min(available,max_nback)
        if isempty(keep)
            curves[key] = (; n_back=Int[], distances=Float64[],
                reference_timestep=nothing, compared_timesteps=Int[], timestep_lag=Int[])
            continue
        end
        reference = last(keep)
        # Last row is ordered oldest -> latest; reverse it to obtain 1-back first.
        earlier = reverse(keep[max(1,length(keep)-n):end-1])
        reference_timestep = analysis.timesteps[key][reference]
        compared_timesteps = analysis.timesteps[key][earlier]
        curves[key] = (; n_back=collect(1:n),
            distances=collect(analysis.distances[key][reference,earlier]),
            reference_timestep, compared_timesteps,
            timestep_lag=reference_timestep .- compared_timesteps)
    end
    return curves
end

# Descriptive ordinary least-squares fit: distance = intercept + slope * n_back.
function _nback_linear_fit(curve)
    length(curve.n_back) >= 2 || return nothing
    x = Float64.(curve.n_back)
    y = Float64.(curve.distances)
    xmean, ymean = mean(x), mean(y)
    centered_x = x .- xmean
    slope = sum(centered_x .* (y .- ymean)) / sum(abs2,centered_x)
    intercept = ymean - slope*xmean
    endpoints = [first(x),last(x)]
    return (; intercept,slope,x=endpoints,y=intercept .+ slope .* endpoints)
end

"""
    plot_nback_distances(analysis; skip_instances=500, max_nback=nothing,
                         point_alpha=.15, markersize=5, ncols=3, output_path=...)

One panel per input key: raw distances as translucent points, with a descriptive
ordinary least-squares regression line. Original Hamming units are retained.
The first `skip_instances` occurrences of each key are excluded. Each panel
autoscales its own y-axis to the displayed data and fitted line.
Returns the Makie figure and saves a PNG. At least two points are required to fit.
"""
function plot_nback_distances(analysis; skip_instances=500, max_nback=nothing,
        point_alpha=.15, markersize=5, ncols=3,
        output_path=joinpath(@__DIR__,"Figures","Hamming_nback.png"))
    0 <= point_alpha <= 1 || throw(ArgumentError("point_alpha must be in [0,1]"))
    markersize > 0 || throw(ArgumentError("markersize must be positive"))
    ncols >= 1 || throw(ArgumentError("ncols must be positive"))
    plot_keys = filter(!=("_"),analysis.keys)
    curves = last_row_nback(analysis; skip_instances,max_nback)
    fits = Dict(key => _nback_linear_fit(curves[key]) for key in plot_keys)
    nrows = cld(length(plot_keys),ncols)
    fig = Figure(resolution=(470*ncols,350*nrows))
    ylabel = analysis.normalized ? "Hamming distance (fraction differing)" :
                                  "Hamming distance (nodes differing)"
    for (index,key) in enumerate(plot_keys)
        row,col = fldmod(index-1,ncols)
        curve = curves[key]
        title = key == "_" ? "_ (separator)" : key
        ax = Axis(fig[row+1,col+1];title="$title (n=$(length(curve.distances)))",
            xlabel="n-back (same-key occurrences)",ylabel)
        if isempty(curve.n_back)
            text!(ax,.5,.5;text="No retained pairs",align=(:center,:center))
            limits!(ax,0,1,0,1)
            continue
        end
        scatter!(ax,curve.n_back,curve.distances;
            color=(:royalblue,point_alpha),markersize)
        fit = fits[key]
        if fit !== nothing
            lines!(ax,fit.x,fit.y;color=:black,linewidth=2.5)
        end
    end
    if output_path !== nothing
        mkpath(dirname(output_path))
        save(output_path,fig)
    end
    return fig
end

end # module

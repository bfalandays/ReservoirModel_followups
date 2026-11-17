# Model, visualization, ensemble, and analysis functions. Include in a fresh Julia session.

module RotatingShapeSingleModel

# BEGIN environment
module RotatingShapeSingleEnvironment

using CairoMakie
export initialize_environment, render_frame, frame_at_time, step_environment!,
       rotation_state, visual_hemispheres, plot_frame, plot_rotation_sequence, record_rotation

mutable struct VisualEnvironment
    N::Int
    side_length::Float64
    object_radius::Float64
    angular_speed::Float64
    dt::Float64
    theta0::Float64
    reverse_rotation::Bool
    rotations_per_direction::Int
    vertices::Matrix{Float64}
    xcenters::Vector{Float64}
    ycenters::Vector{Float64}
    step::Int
end

# Asymmetric L with unequal arms, subsequently centered on its area centroid.
const DEFAULT_SHAPE = [-.45 -.50; .45 -.50; .45 -.20; -.15 -.20; -.15 .80; -.45 .80]

function polygon_centroid(vertices)
    twice_area, cx, cy = 0., 0., 0.
    n = size(vertices,1)
    for i in 1:n
        j = mod1(i+1,n)
        cross = vertices[i,1]*vertices[j,2] - vertices[j,1]*vertices[i,2]
        twice_area += cross
        cx += (vertices[i,1]+vertices[j,1])*cross
        cy += (vertices[i,2]+vertices[j,2])*cross
    end
    abs(twice_area) > eps(Float64) || throw(ArgumentError("Object polygon must have nonzero area"))
    return (cx/(3*twice_area),cy/(3*twice_area))
end

"""
    initialize_environment(; N=64, side_length=2., object_radius=.65,
                            angular_speed=pi/2, dt=1/30, theta0=0.,
                            reverse_rotation=false, rotations_per_direction=5, vertices=...)

N-by-N pixel square feeding one shared sensory pool. Angles are radians; speed is
radians per unit time, and dt is time per simulation step. Positive speed means
counterclockwise rotation. The polygon is centered on its area centroid and
scaled so its furthest vertex is object_radius from the square center.
With reverse_rotation=true, alternate direction after each rotations_per_direction
full turns, starting in the direction given by the sign of angular_speed.
"""
function initialize_environment(; N=64, side_length=2., object_radius=.65,
        angular_speed=pi/2, dt=1/30, theta0=0., reverse_rotation=false,
        rotations_per_direction=5, vertices=DEFAULT_SHAPE)
    N isa Integer && !(N isa Bool) && N >= 2 ||
        throw(ArgumentError("N must be an integer >= 2"))
    isfinite(side_length) && side_length > 0 || throw(ArgumentError("side_length must be positive and finite"))
    isfinite(object_radius) && 0 < object_radius < side_length/2 ||
        throw(ArgumentError("object_radius must lie strictly between 0 and side_length/2"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be positive and finite"))
    isfinite(angular_speed) && isfinite(theta0) || throw(ArgumentError("Angles and angular_speed must be finite"))
    size(vertices,2) == 2 && size(vertices,1) >= 3 && all(isfinite,vertices) ||
        throw(ArgumentError("vertices must be a finite M-by-2 simple polygon, M >= 3"))
    reverse_rotation isa Bool || throw(ArgumentError("reverse_rotation must be true or false"))
    rotations_per_direction isa Integer && !(rotations_per_direction isa Bool) && rotations_per_direction>=1 ||
        throw(ArgumentError("rotations_per_direction must be a positive integer"))
    shape = Matrix{Float64}(vertices)
    cx,cy = polygon_centroid(shape)
    shape[:,1] .-= cx
    shape[:,2] .-= cy
    radius = maximum(hypot.(shape[:,1],shape[:,2]))
    shape .*= object_radius/radius
    pixel_width = side_length/N
    xcenters = collect(range(-side_length/2+pixel_width/2;step=pixel_width,length=N))
    ycenters = reverse(copy(xcenters))  # matrix row 1 is the TOP of the image
    return VisualEnvironment(N,side_length,object_radius,angular_speed,dt,theta0,
        reverse_rotation,rotations_per_direction,
        shape,xcenters,ycenters,0)
end

# Pixel-center inclusion in a simple polygon; points on its boundary count inside.
function inside_polygon(x,y,vertices)
    inside = false
    n = size(vertices,1)
    for i in 1:n
        j = mod1(i+1,n)
        ax,ay = vertices[i,1],vertices[i,2]
        bx,by = vertices[j,1],vertices[j,2]
        cross = (x-ax)*(by-ay) - (y-ay)*(bx-ax)
        tolerance = 32eps(Float64)*max(1.,abs(ax),abs(ay),abs(bx),abs(by))
        if abs(cross) <= tolerance && min(ax,bx)-tolerance <= x <= max(ax,bx)+tolerance &&
                min(ay,by)-tolerance <= y <= max(ay,by)+tolerance
            return true
        end
        if (ay > y) != (by > y)
            x_intersection = ax + (y-ay)*(bx-ax)/(by-ay)
            x < x_intersection && (inside = !inside)
        end
    end
    return inside
end

"""Orientation and outgoing velocity at an absolute simulation time.
Reversals occur after each rotations_per_direction complete turns. The analytic
triangle-wave angle integrates any timestep that straddles a reversal without
jumping orientation or losing overshoot. angular_speed retains its initial sign.
"""
function rotation_state(env::VisualEnvironment,time)
    isfinite(time) && time>=0 || throw(ArgumentError("time must be finite and nonnegative"))
    omega = env.angular_speed
    if !env.reverse_rotation || iszero(omega)
        return (;angle=env.theta0+omega*time,angular_velocity=omega,
            rotation_direction=Int8(sign(omega)))
    end
    leg_angle = 2pi*env.rotations_per_direction
    legs = abs(omega)*time/leg_angle
    nearest = round(legs)
    abs(legs-nearest)<=16eps(max(1.,abs(legs))) && (legs=nearest)
    phase = mod(legs,2.)
    displacement = (phase<=1. ? phase : 2.0 - phase)*leg_angle
    direction = Int8(sign(omega)*(phase<1. ? 1 : -1))
    return (;angle=env.theta0+sign(omega)*displacement,
        angular_velocity=abs(omega)*direction,rotation_direction=direction)
end

"""
    frame_at_time(env, time)

Render at the scheduled orientation without changing env. Pixels use optical
intensity: 0.0 = black object, 1.0 = white background. object_mask uses true for
object occupancy. Matrices are [row, column], with row 1 at the top and column 1
at the left. A pixel is black when its center lies inside the rotated polygon.
"""
function frame_at_time(env::VisualEnvironment,time)
    isfinite(time) && time >= 0 || throw(ArgumentError("time must be finite and nonnegative"))
    rotation = rotation_state(env,time)
    angle = rotation.angle
    s,c = sincos(mod(angle,2pi))
    mask = falses(env.N,env.N)
    for col in 1:env.N, row in 1:env.N
        x,y = env.xcenters[col],env.ycenters[row]
        # Rotate pixel centers backwards; test against the fixed original polygon.
        local_x = c*x+s*y
        local_y = -s*x+c*y
        mask[row,col] = inside_polygon(local_x,local_y,env.vertices)
    end
    return (; pixels=Float64.(.!mask),object_mask=mask,time=Float64(time),angle,
        angular_velocity=rotation.angular_velocity,rotation_direction=rotation.rotation_direction)
end

render_frame(env::VisualEnvironment) = frame_at_time(env,env.step*env.dt)

function step_environment!(env::VisualEnvironment; nsteps=1)
    nsteps isa Integer && nsteps >= 0 || throw(ArgumentError("nsteps must be a nonnegative integer"))
    env.step += nsteps
    return render_frame(env)
end

"""Split image columns into equal left/right views without changing pixel order."""
function visual_hemispheres(frame)
    n = size(frame.pixels,2)
    iseven(n) || throw(ArgumentError("Image width must be even"))
    half = n ÷ 2
    return (; left=@view(frame.pixels[:,1:half]),right=@view(frame.pixels[:,half+1:n]))
end

# Makie matrix coordinates are [x,y], while our image coordinates are [row,column].
plot_pixels(frame) = permutedims(reverse(frame.pixels;dims=1))
frame_title(frame) = "t = $(round(frame.time;digits=2)), angle = $(round(rad2deg(mod(frame.angle,2pi));digits=1))°"

function decorate_axis!(ax,env; show_split=false, show_grid=false)
    half = env.side_length/2
    if show_grid
        boundaries = range(-half,half;length=env.N+1)
        vlines!(ax,boundaries;color=(:gray,.2),linewidth=.4)
        hlines!(ax,boundaries;color=(:gray,.2),linewidth=.4)
    end
    if show_split
        vlines!(ax,[0.];color=:darkorange,linestyle=:dash,linewidth=2)
    end
    limits!(ax,-half,half,-half,half)
    return ax
end

function plot_frame(env::VisualEnvironment; time=env.step*env.dt,
        show_split=false, show_grid=false,
        output_path=joinpath(@__DIR__,"Figures","RotatingShapeSingle_frame.png"))
    frame = frame_at_time(env,time)
    fig = Figure(resolution=(700,750))
    ax = Axis(fig[1,1];aspect=DataAspect(),xlabel="x",ylabel="y",title=frame_title(frame))
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    heatmap!(ax,edges,edges,plot_pixels(frame);colormap=[:black,:white],
        colorrange=(0,1),interpolate=false)
    decorate_axis!(ax,env;show_split,show_grid)
    Label(fig[2,1],"$(env.N) × $(env.N) pixels | All pixels feed one sensory pool";tellwidth=false)
    if output_path !== nothing
        mkpath(dirname(output_path))
        save(output_path,fig)
    end
    return fig
end

function plot_rotation_sequence(env::VisualEnvironment; times=collect(0:.5:3.5),
        ncols=4, show_split=false,
        output_path=joinpath(@__DIR__,"Figures","RotatingShapeSingle_sequence.png"))
    !isempty(times) || throw(ArgumentError("times must be nonempty"))
    ncols isa Integer && ncols >= 1 || throw(ArgumentError("ncols must be positive"))
    fig = Figure(resolution=(350*ncols,350*cld(length(times),ncols)))
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    for (i,time) in enumerate(times)
        frame = frame_at_time(env,time)
        row,col = fldmod(i-1,ncols)
        ax = Axis(fig[row+1,col+1];aspect=DataAspect(),title=frame_title(frame))
        heatmap!(ax,edges,edges,plot_pixels(frame);colormap=[:black,:white],
            colorrange=(0,1),interpolate=false)
        decorate_axis!(ax,env;show_split)
    end
    if output_path !== nothing
        mkpath(dirname(output_path))
        save(output_path,fig)
    end
    return fig
end

"""Record an animation following the rotation/reversal schedule without advancing the simulation state."""
function record_rotation(env::VisualEnvironment; duration=4., framerate=30,
        show_split=false,
        output_path=joinpath(@__DIR__,"Figures","RotatingShapeSingle.gif"))
    isfinite(duration) && duration > 0 || throw(ArgumentError("duration must be positive and finite"))
    framerate isa Integer && framerate > 0 || throw(ArgumentError("framerate must be positive"))
    initial = frame_at_time(env,0.)
    image = Observable(plot_pixels(initial))
    fig = Figure(resolution=(650,650))
    ax = Axis(fig[1,1];aspect=DataAspect(),xlabel="x",ylabel="y",title=frame_title(initial))
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    heatmap!(ax,edges,edges,image;colormap=[:black,:white],colorrange=(0,1),interpolate=false)
    decorate_axis!(ax,env;show_split)
    times = range(0.;step=1/framerate,length=max(1,ceil(Int,duration*framerate)))
    mkpath(dirname(output_path))
    record(fig,output_path,times;framerate) do time
        frame = frame_at_time(env,time)
        image[] = plot_pixels(frame)
        ax.title[] = frame_title(frame)
    end
    return output_path
end
end # module
# END environment
using .RotatingShapeSingleEnvironment
# BEGIN reservoir
module RotatingShapeSingleReservoir

using Random, SparseArrays, LinearAlgebra, Statistics, DelimitedFiles
using ..RotatingShapeSingleEnvironment
# Reuse the language model's actual initialization and equations. Keep its module
# nested here to avoid replacing any language model already loaded in Main.
# BEGIN shared_language
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
# END shared_language
export initialize_reservoir, model_step!, run_rotating_model, save_run, new_run_directory

mutable struct VisualReservoir{P}
    env::RotatingShapeSingleEnvironment.VisualEnvironment
    params::P
    link_mat::Matrix{Float64}
    input_wmat::SparseMatrixCSC{Float64,Int}
    wmat::Matrix{Float64}
    acts::Vector{Float64}
    spikes::Vector{Float64}
    targets::Vector{Float64}
    errors::Vector{Float64}
    prev_spikes::Vector{Float64}
    inputs::Vector{Float64}
    n::Int
    mean_act::Float64
    mean_err::Float64
    mean_targ::Float64
end

"""
    initialize_reservoir(env; nnodes=250, p_link=.1, sensory_fraction=.1, input_p_link=.1, ...)

One homogeneous recurrent reservoir, with no self edges or subnetwork partitions.
Choose round(Int, sensory_fraction*nnodes) distinct sensory nodes. Every pixel can
connect to every selected sensory node with probability input_p_link, independent
of which visual half it occupies. Input weights stay fixed at input_amp.
"""
function initialize_reservoir(env; nnodes=250,p_link=.1,input_p_link=.1,sensory_fraction=.1,
        leak=.25,lrate_wmat=1.,lrate_targ=.01,targ_min=1.,input_amp=5.,
        learn_on=true,acts_neg=0,seed=1,input_seed=seed+1)
    nnodes isa Integer && !(nnodes isa Bool) && nnodes >= 2 ||
        throw(ArgumentError("nnodes must be an integer >= 2"))
    0 <= input_p_link <= 1 || throw(ArgumentError("input_p_link must be in [0,1]"))
    0 <= sensory_fraction <= 1 || throw(ArgumentError("sensory_fraction must be in [0,1]"))
    isfinite(input_amp) && input_amp >= 0 || throw(ArgumentError("input_amp must be finite and nonnegative"))
    core = AllostaticLanguage.initialize_model(;nnodes,p_link,leak,lrate_wmat,
        lrate_targ,targ_min,input_amp,learn_on,acts_neg,seed,subnetworks=0,n_subnet_edges=0)
    rng = MersenneTwister(input_seed)
    npixels = env.N^2
    n_sensory = round(Int,sensory_fraction*nnodes)
    sensory_nodes = sort!(randperm(rng,nnodes)[1:n_sensory])
    rows,cols,weights = Int[],Int[],Float64[]
    for node in sensory_nodes
        for pixel in 1:npixels
            if rand(rng) < input_p_link && input_amp != 0
                push!(rows,pixel)
                push!(cols,node)
                push!(weights,input_amp)
            end
        end
    end
    input_wmat = sparse(rows,cols,weights,npixels,nnodes)
    params = merge(core.params,(;input_p_link,sensory_fraction,sensory_nodes,input_seed,N=env.N))
    return VisualReservoir(env,params,core.link_mat,input_wmat,core.wmat,
        core.acts,core.spikes,core.targets,core.errors,core.prev_spikes,
        zeros(npixels),0,0.,0.,Float64(targ_min))
end

"""Present one frame (black=1), update spikes/learning, then advance time by dt."""
function model_step!(model::VisualReservoir; learn_on=model.params.learn_on)
    frame = render_frame(model.env)
    model.inputs .= vec(frame.object_mask) # occupancy, NOT optical intensity
    p = model.params
    model.acts,model.spikes,model.errors,model.prev_spikes =
        AllostaticLanguage.get_acts(model.acts,p.leak,model.spikes,model.wmat,
            model.inputs,model.input_wmat,model.targets;acts_neg=p.acts_neg)
    AllostaticLanguage.learning(learn_on,model.link_mat,model.spikes,model.prev_spikes,
        model.errors,model.wmat,model.targets;
        lrate_wmat=p.lrate_wmat,lrate_targ=p.lrate_targ,targ_min=p.targ_min)
    model.n += 1
    model.mean_act = mean(model.acts)
    model.mean_err = mean(model.errors)
    model.mean_targ = mean(model.targets)
    model.env.step += 1
    return frame
end

function _spike_history(iterations,nnodes)
    return (;spikes=zeros(UInt8,iterations,nnodes),
        time=zeros(iterations),angle=zeros(iterations),
        mean_act=zeros(iterations),mean_error=zeros(iterations),mean_target=zeros(iterations),
        mean_abs_error=zeros(iterations),
        rotation_direction=zeros(Int8,iterations),angular_velocity=zeros(iterations),
        black_pixels=zeros(Int,iterations),black_left=zeros(Int,iterations),black_right=zeros(Int,iterations))
end

function _record_step!(ts,i,model,frame)
        ts.spikes[i,:] .= model.spikes
        ts.time[i],ts.angle[i] = frame.time,frame.angle
        ts.mean_act[i],ts.mean_error[i],ts.mean_target[i] =
            model.mean_act,model.mean_err,model.mean_targ
        errors = abs.(model.acts .- model.targets)
        ts.mean_abs_error[i] = mean(errors)
        ts.rotation_direction[i] = frame.rotation_direction
        ts.angular_velocity[i] = frame.angular_velocity
        ts.black_left[i] = count(@view frame.object_mask[:,1:model.env.N÷2])
        ts.black_right[i] = count(@view frame.object_mask[:,model.env.N÷2+1:end])
        ts.black_pixels[i] = ts.black_left[i]+ts.black_right[i]
    return ts
end

function new_run_directory(root)
    mkpath(root)
    index = 1
    while true
        path = joinpath(root,"run_"*lpad(string(index),4,'0'))
        if !ispath(path)
            mkdir(path)
            return abspath(path)
        end
        index += 1
    end
end

"""
    run_rotating_model(env; iterations=40000, save=true, output_dir=..., kwargs...)

One iteration = one rendered frame and one reservoir update. ts.spikes is
iterations×nnodes UInt8; time/angle rows describe the input that produced that
spike row. The environment is copied by default so previewing does not advance
or alter the run. The copied environment starts at the supplied env.step.
"""
function run_rotating_model(env;iterations=40000,save=true,
        output_dir=joinpath(@__DIR__,"Data"),progress_every=1000,kwargs...)
    iterations isa Integer && iterations > 0 || throw(ArgumentError("iterations must be positive"))
    progress_every isa Integer && progress_every >= 0 || throw(ArgumentError("progress_every must be nonnegative"))
    model = initialize_reservoir(deepcopy(env);kwargs...)
    start_step = model.env.step
    start_wmat = copy(model.wmat)
    ts = _spike_history(iterations,model.params.nnodes)
    for i in 1:iterations
        frame = model_step!(model)
        _record_step!(ts,i,model,frame)
        if progress_every > 0 && (i==1 || i%progress_every==0 || i==iterations)
            println("Iteration $i/$iterations; time=$(round(frame.time;digits=3)); mean error=$(model.mean_err)")
            flush(stdout)
        end
    end
    run_dir = save ? new_run_directory(output_dir) : nothing
    result = (;model,ts,start_wmat,iterations,start_step,output_dir=run_dir)
    save && save_run(result,run_dir)
    return result
end

function write_csv(path,columns,data)
    open(path,"w") do io
        writedlm(io,permutedims(columns),',')
        writedlm(io,data,',')
    end
end

"""Save all spike rows, time/angle metadata, topology, weights and visual parameters."""
function save_run(result,dir)
    mkpath(dir)
    model,ts = result.model,result.ts
    node_names = ["node_$j" for j in axes(ts.spikes,2)]
    write_csv(joinpath(dir,"spikes.csv"),node_names,ts.spikes)
    write_csv(joinpath(dir,"timesteps.csv"),
        ["iteration","time","angle_radians","angle_degrees_wrapped","black_pixels","black_left","black_right","rotation_direction","angular_velocity"],
        hcat(1:result.iterations,ts.time,ts.angle,rad2deg.(mod.(ts.angle,2pi)),
            ts.black_pixels,ts.black_left,ts.black_right,ts.rotation_direction,ts.angular_velocity))
    write_csv(joinpath(dir,"reservoir_summary.csv"),
        ["mean_activation","mean_signed_error","mean_target","spike_count","mean_absolute_error"],
        hcat(ts.mean_act,ts.mean_error,ts.mean_target,sum(ts.spikes;dims=2),ts.mean_abs_error))
    write_csv(joinpath(dir,"link_mat.csv"),node_names,model.link_mat)
    write_csv(joinpath(dir,"start_wmat.csv"),node_names,result.start_wmat)
    write_csv(joinpath(dir,"end_wmat.csv"),node_names,model.wmat)
    write_csv(joinpath(dir,"final_targets.csv"),["target"],model.targets)
    # Save the sparse input graph as exact edge triples rather than a mostly-zero matrix.
    pixel,node,weight = findnz(model.input_wmat)
    row = mod.(pixel .- 1,model.env.N) .+ 1
    column = div.(pixel .- 1,model.env.N) .+ 1
    write_csv(joinpath(dir,"input_edges.csv"),
        ["pixel_index","pixel_row","pixel_column","reservoir_node","weight"],
        hcat(pixel,row,column,node,weight))
    write_csv(joinpath(dir,"sensory_nodes.csv"),["reservoir_node"],model.params.sensory_nodes)
    write_csv(joinpath(dir,"shape_vertices.csv"),["x","y"],model.env.vertices)
    open(joinpath(dir,"run_parameters.txt"),"w") do io
        println(io,"julia_version = ",VERSION)
        println(io,"iterations = ",result.iterations)
        println(io,"start_environment_step = ",result.start_step)
        for (key,value) in pairs(model.params)
            println(io,key," = ",value)
        end
        for key in (:side_length,:object_radius,:angular_speed,:dt,:theta0,:reverse_rotation,:rotations_per_direction)
            println(io,key," = ",getproperty(model.env,key))
        end
        println(io,"input_encoding = black:1 white:0")
        println(io,"pixel_index = row + (column - 1) * N")
        println(io,"input_projection = all pixels to one shared sensory pool")
        println(io,"rotation_direction = +1 counterclockwise; -1 clockwise; 0 stationary")
    end
    return abspath(dir)
end
end # module
# END reservoir
using .RotatingShapeSingleReservoir
module RotatingShapeSingleInteractive

using CairoMakie
using ..RotatingShapeSingleEnvironment
using ..RotatingShapeSingleReservoir
export launch_rotating_dashboard, interactive_result

mutable struct InteractiveRun{M,T,F}
    model::M
    initial_model::M
    ts::T
    last_frame::F
    iterations::Int
    start_wmat::Matrix{Float64}
    start_step::Int
    output_root::String
    run_dir::String
    saved_iterations::Int
end

function _interactive_state(env; iterations=40000,output_dir,kwargs...)
    iterations isa Integer && iterations > 0 || throw(ArgumentError("iterations must be positive"))
    model = initialize_reservoir(deepcopy(env);kwargs...)
    ts = RotatingShapeSingleReservoir._spike_history(iterations,model.params.nnodes)
    return InteractiveRun(model,deepcopy(model),ts,render_frame(model.env),iterations,
        copy(model.wmat),model.env.step,abspath(output_dir),new_run_directory(output_dir),0)
end

"""Take a copied snapshot of all completed interactive iterations for analysis/export."""
function interactive_result(state::InteractiveRun)
    n = state.model.n
    ts = (; (key => (value isa AbstractMatrix ? copy(value[1:n,:]) : copy(value[1:n]))
             for (key,value) in pairs(state.ts))...)
    return (;model=deepcopy(state.model),ts,start_wmat=copy(state.start_wmat),
        iterations=n,start_step=state.start_step,output_dir=state.run_dir)
end
interactive_result(dashboard::NamedTuple) = interactive_result(dashboard.state)

function _save_state!(state)
    state.model.n == 0 && return nothing
    result = interactive_result(state)
    save_run(result,state.run_dir)
    state.saved_iterations = state.model.n
    return state.run_dir
end

function _step_state!(state,nsteps=1)
    nsteps isa Integer && nsteps >= 0 || throw(ArgumentError("nsteps must be nonnegative"))
    count = min(nsteps,state.iterations-state.model.n)
    for _ in 1:count
        frame = RotatingShapeSingleReservoir.model_step!(state.model)
        RotatingShapeSingleReservoir._record_step!(state.ts,state.model.n,state.model,frame)
        state.last_frame = frame
    end
    return count
end

function _reset_state!(state)
    # Keep the model object itself, so the main script's `model` binding stays live.
    for name in fieldnames(typeof(state.model))
        setproperty!(state.model,name,deepcopy(getproperty(state.initial_model,name)))
    end
    state.ts = RotatingShapeSingleReservoir._spike_history(state.iterations,state.model.params.nnodes)
    state.last_frame = render_frame(state.model.env)
    state.saved_iterations = 0
    state.run_dir = new_run_directory(state.output_root)
    return state
end

function _network_geometry(model)
    n = model.params.nnodes
    points = [Point2f(cos(2pi*(i-1)/n),sin(2pi*(i-1)/n)) for i in 1:n]
    within_points,within_dirs = Point2f[],Vec2f[]
    for src in 1:n,dst in 1:n
        (src==dst || iszero(model.link_mat[src,dst])) && continue
        p,q = points[src],points[dst]
        dx,dy = q[1]-p[1],q[2]-p[2]
        len = hypot(dx,dy)
        push!(within_points,Point2f(p[1]+.035dx/len,p[2]+.035dy/len))
        push!(within_dirs,Vec2f(dx-.07dx/len,dy-.07dy/len))
    end
    return (;points,within_points,within_dirs)
end

_node_colors(model) = [s>0 ? :yellow : :royalblue for s in model.spikes]

function _build_dashboard(env;iterations=40000,output_dir,playback_fps=30,
        autoplay=false,show=true,start_timer=true,backend_module=CairoMakie,
        save_on_close=true,kwargs...)
    playback_fps isa Integer && 1<=playback_fps<=60 ||
        throw(ArgumentError("playback_fps must be an integer in 1:60"))
    state = _interactive_state(env;iterations,output_dir,kwargs...)
    model = state.model
    geometry = _network_geometry(model)
    running = Observable(false)
    colors = Observable(_node_colors(model))
    image_data = Observable(RotatingShapeSingleEnvironment.plot_pixels(state.last_frame))
    absolute_errors = Observable(abs.(model.acts .- model.targets))
    error_history = Observable(Point2f[])
    target_values = Observable(copy(model.targets))
    target_bins = lift(target_values) do values
        lo,hi = extrema(values)
        if lo==hi
            padding = max(abs(lo)*.05,.5)
            lo,hi = lo-padding,hi+padding
        end
        range(lo,hi;length=21)
    end
    status = Observable("Ready — iteration 0 / $iterations")
    message = Observable("Paused. Run advances one recorded model iteration per displayed frame.")
    fig = Figure(resolution=(1200,1100))
    Label(fig[1,1:2],status;fontsize=21,tellwidth=false)
    ax_network = Axis(fig[2,1];aspect=DataAspect(),title="Reservoir network — current spikes")
    hidedecorations!(ax_network)
    hidespines!(ax_network)
    if !isempty(geometry.within_points)
        arrows!(ax_network,geometry.within_points,geometry.within_dirs;
            color=(:slategray,.2),linewidth=.6,arrowsize=6)
    end
    scatter!(ax_network,geometry.points;color=colors,markersize=15,
        strokecolor=:black,strokewidth=.6)
    limits!(ax_network,-1.35,1.35,-1.35,1.35)
    Label(fig[3,1],"Yellow: spiking | Blue: silent\nGray arrows: recurrent connections";tellwidth=false)

    ax_stimulus = Axis(fig[2,2];aspect=DataAspect(),title="Stimulus — frame driving current spikes",xlabel="x",ylabel="y")
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    heatmap!(ax_stimulus,edges,edges,image_data;colormap=[:black,:white],
        colorrange=(0,1),interpolate=false)
    RotatingShapeSingleEnvironment.decorate_axis!(ax_stimulus,env;show_split=false)
    Label(fig[3,2],"$(env.N) × $(env.N) pixels | Black = sensory 1, white = sensory 0\nAll pixels → one shared sensory pool";tellwidth=false)

    ax_errors = Axis(fig[4,1];title="Mean absolute error over time",
        xlabel="Model iteration",ylabel="Mean absolute error")
    lines!(ax_errors,error_history;color=:royalblue,linewidth=2)
    Label(fig[5,1],"mean(abs(activation − target))\nMean over all nodes, per iteration";tellwidth=false)

    ax_targets = Axis(fig[4,2];title="Target value distribution",
        xlabel="Target value",ylabel="Number of nodes")
    hist!(ax_targets,target_values;bins=target_bins,color=(:royalblue,.7),
        strokecolor=:white,strokewidth=.5)
    Label(fig[5,2],"Current targets\nOne value per reservoir node";tellwidth=false)

    controls = GridLayout()
    fig[6,1:2] = controls
    run_button = Button(controls[1,1];label="Run")
    step_button = Button(controls[1,2];label="Step")
    reset_button = Button(controls[1,3];label="Reset")
    save_button = Button(controls[1,4];label="Save spikes")
    fps_label = Observable("Playback: $playback_fps iterations / second")
    Label(fig[7,1],fps_label;tellwidth=false)
    fps_slider = Slider(fig[7,2];range=1:60,startvalue=playback_fps)
    Label(fig[8,1:2],message;tellwidth=false,fontsize=13)

    timer_ref = Ref{Union{Nothing,Timer}}(nothing)
    function refresh!()
        colors[] = _node_colors(state.model)
        image_data[] = RotatingShapeSingleEnvironment.plot_pixels(state.last_frame)
        # Recompute using current targets, which learning may have updated.
        absolute_errors[] = abs.(state.model.acts .- state.model.targets)
        # All completed steps are recorded, including advance(nsteps) calls.
        n = state.model.n
        for (history,values) in ((error_history,state.ts.mean_abs_error),)
            points = history[]
            length(points)>n && empty!(points)
            for i in length(points)+1:n
                push!(points,Point2f(i,values[i]))
            end
            notify(history)
        end
        autolimits!(ax_errors)
        target_values[] = copy(state.model.targets)
        autolimits!(ax_targets)
        frame = state.last_frame
        angle = round(rad2deg(mod(frame.angle,2pi));digits=1)
        direction = frame.rotation_direction>0 ? "CCW" : frame.rotation_direction<0 ? "CW" : "stationary"
        status[] = "Iteration $(state.model.n) / $iterations  |  Input t=$(round(frame.time;digits=3)) s  |  Angle $(angle)° ($direction)  |  Spikes $(Int(sum(state.model.spikes))) / $(state.model.params.nnodes)"
        return nothing
    end
    function save!()
        path = _save_state!(state)
        message[] = path===nothing ? "No recorded frames to save yet." :
            "Saved $(state.model.n) spike vectors to $path"
        return path
    end
    function advance!(nsteps=1)
        _step_state!(state,nsteps)
        refresh!()
        if state.model.n>=iterations
            running[] = false
            state.saved_iterations<state.model.n && save!()
            run_button.label[] = "Complete"
        end
        return state.model.n
    end
    function reset!()
        running[] = false
        # Preserve an unfinished run before clearing its buffers.
        state.model.n>state.saved_iterations && save!()
        _reset_state!(state)
        refresh!()
        status[] = "Ready — iteration 0 / $iterations"
        run_button.label[] = "Run"
        message[] = "Reset to original weights and angle. New run: $(state.run_dir)"
        return nothing
    end
    function guarded(action)
        try
            action()
        catch err
            running[] = false
            message[] = "Stopped: "*sprint(showerror,err)
            @error "RotatingShapeSingle interactive action failed" exception=(err,catch_backtrace())
        end
        return nothing
    end
    on(running) do value
        run_button.label[] = value ? "Pause" : (state.model.n>=iterations ? "Complete" : "Run")
        message[] = value ? "Running — every displayed frame records one model iteration." :
            "Paused at iteration $(state.model.n). Step advances one frame; Run continues playback."
    end
    on(run_button.clicks) do _
        state.model.n<iterations && (running[] = !running[])
    end
    on(step_button.clicks) do _
        running[] = false
        guarded(()->advance!(1))
    end
    on(reset_button.clicks) do _
        guarded(reset!)
    end
    on(save_button.clicks) do _
        running[] = false
        guarded(save!)
    end
    on(fps_slider.value) do value
        fps_label[] = "Playback: $(Int(value)) iterations / second"
    end
    function stop!()
        running[] = false
        timer_ref[] !== nothing && close(timer_ref[])
        if save_on_close && state.model.n>state.saved_iterations
            guarded(save!)
        end
        return nothing
    end
    screen = show ? display(fig;backend=backend_module,inline=false,title="RotatingShapeSingle — live reservoir") : nothing
    if show
        on(events(fig).window_open) do open
            !open && stop!()
        end
    end
    if start_timer
        last_tick = Ref(time())
        timer_ref[] = Timer(0;interval=1/120) do timer
            if show && !events(fig).window_open[]
                close(timer)
                return
            end
            if running[] && time()-last_tick[]>=1/fps_slider.value[]
                guarded(()->advance!(1))
                last_tick[] = time()
            end
        end
    end
    running[] = autoplay
    return (;fig,screen,state,running,node_colors=colors,image_data,absolute_errors,error_history,target_values,target_bins,status,message,
        buttons=(;run=run_button,step=step_button,reset=reset_button,save=save_button),
        fps_slider,timer=timer_ref,advance=advance!,reset=reset!,save=save!,stop=stop!)
end

"""
    launch_rotating_dashboard(env;iterations=40000,playback_fps=30,autoplay=false,...)

Open a GLMakie window with a 2×2 grid of network/spike, stimulus, mean-absolute-error time series and target views. Starts paused
by default. Save, completion, reset and window close preserve recorded spikes in
numbered run directories. backend=:cairo, show=false, start_timer=false are
available for offscreen checks. Playback changes wall-clock pace only, not dt.
"""
function launch_rotating_dashboard(env;iterations=40000,playback_fps=30,
        autoplay=false,output_dir=joinpath(@__DIR__,"Data","Interactive"),
        backend=:gl,show=true,start_timer=true,kwargs...)
    backend_module = if backend==:gl
        @eval import GLMakie
        Base.invokelatest(GLMakie.activate!;inline=false)
        GLMakie
    elseif backend==:cairo
        CairoMakie.activate!()
        CairoMakie
    else
        throw(ArgumentError("backend must be :gl or :cairo"))
    end
    return Base.invokelatest(_build_dashboard,env;iterations,playback_fps,autoplay,
        output_dir,backend_module,show,start_timer,kwargs...)
end
end # module
using .RotatingShapeSingleInteractive
module RotatingShapeSingleDrift

using CairoMakie, LinearAlgebra, Statistics, DelimitedFiles
export window_spike_pca, plot_orientation_pc1, plot_spike_pc12, plot_spike_pc12_time, spike_hamming_distances, plot_spike_hamming_heatmap, binned_orientation_correlations, plot_binned_orientation_correlations, binned_orientation_cosine_similarity, plot_orientation_cosine_similarity, binned_orientation_cosine_matrix, plot_orientation_cosine_heatmap, binned_orientation_cosines, plot_binned_orientation_cosines, binned_orientation_correlation, plot_orientation_correlation, binned_orientation_correlation_matrix, plot_orientation_correlation_heatmap, plot_orientation_correlation_triptych, load_drift_run

"""Read spike vectors and their matching input orientations from a saved run directory."""
function load_drift_run(dir::AbstractString)
    spikes = readdlm(joinpath(dir,"spikes.csv"),',',Float64;skipstart=1)
    metadata,header = readdlm(joinpath(dir,"timesteps.csv"),',',Float64;header=true)
    angle_col = findfirst(==("angle_radians"),vec(header))
    angle_col === nothing && throw(ArgumentError("timesteps.csv lacks angle_radians"))
    size(spikes,1)==size(metadata,1) || throw(ArgumentError("Spike and timestep row counts differ"))
    metadata[:,1]==collect(1:size(spikes,1)) || throw(ArgumentError("Expected consecutive iteration rows starting at 1"))
    direction_col = findfirst(==("rotation_direction"),vec(header))
    ts = (;spikes,angle=metadata[:,angle_col])
    if direction_col !== nothing
        ts = merge(ts,(;rotation_direction=metadata[:,direction_col]))
    end
    return (;ts)
end

"""
    window_spike_pca(result; window=5000:6000, nodes=:all, ncomponents=1)

Fit PCA to the selected inclusive iteration window. Rows are observations and
columns are neurons. Center each neuron's binary spikes without variance scaling.
`nodes` may be :all or a vector of node indices.
Returns PC1 scores, loadings, the training mean and explained variance, as well as
the selected spike matrix and matching wrapped input orientations. Use ncomponents
to retain additional PCs in component_scores, component_loadings and
explained_variances. The original scores/loadings fields always refer to PC1.
No model is run.
"""
function window_spike_pca(result;window=5000:6000,nodes=:all,ncomponents=1)
    ts = result.ts
    n,nneurons = size(ts.spikes)
    length(ts.angle)==n || throw(ArgumentError("One orientation is required per spike row"))
    window isa AbstractUnitRange{<:Integer} || throw(ArgumentError("window must be a consecutive integer range"))
    length(window)>=2 || throw(ArgumentError("PCA requires at least two iterations"))
    first(window)>=1 && last(window)<=n || throw(ArgumentError("Window must lie within completed iterations 1:$n"))
    if nodes===:all
        ids = collect(1:nneurons)
    elseif nodes isa AbstractVector{<:Integer}
        ids = collect(nodes)
    else
        throw(ArgumentError("nodes must be :all or a vector of indices"))
    end
    !isempty(ids) && all(i->1<=i<=nneurons,ids) && allunique(ids) ||
        throw(ArgumentError("Node indices must be nonempty, unique and in bounds"))
    spikes = Float64.(ts.spikes[window,ids])
    ncomponents isa Integer && 1<=ncomponents<=min(size(spikes,1)-1,size(spikes,2)) ||
        throw(ArgumentError("ncomponents must be between 1 and min(window length - 1, selected neurons)"))
    angles = Float64.(ts.angle[window])
    all(isfinite,spikes) && all(isfinite,angles) || throw(ArgumentError("Spikes and orientations must be finite"))
    center = vec(mean(spikes;dims=1))
    centered = spikes .- transpose(center)
    fit = svd(centered;full=false)
    total = sum(abs2,fit.S)
    total>0 || throw(ArgumentError("No spike variance in this window for nodes $nodes; PCA is undefined"))
    component_loadings = copy(fit.V[:,1:ncomponents])
    # Fix each arbitrary sign reproducibly; this does not align separate PCA fits.
    for j in 1:ncomponents
        loading = @view component_loadings[:,j]
        loading[argmax(abs.(loading))]<0 && (loading .*= -1)
    end
    component_scores = centered*component_loadings
    explained_variances = fit.S[1:ncomponents].^2 ./ total
    return (;iterations=collect(window),nodes=ids,spikes,angles,
        orientation_degrees=rad2deg.(mod.(angles,2pi)),
        scores=component_scores[:,1],loadings=component_loadings[:,1],center,
        explained_variance=explained_variances[1],
        component_scores,component_loadings,explained_variances)
end

"""
    plot_orientation_pc1(result_or_directory; window=5000:6000,
        groups=(:all,), savepath=nothing)

Static time-colored PC1 versus orientation trajectories. Fit each group's PCA on
the requested window only. Lines follow chronological order; connections across
0/360 degrees are omitted. Returns (;fig, analyses). For later cross-window
comparisons, reuse a fixed PCA basis rather than comparing independent axes.
"""
function plot_orientation_pc1(source;window=5000:6000,groups=(:all,),savepath=nothing)
    result = source isa AbstractString ? load_drift_run(source) : source
    !isempty(groups) || throw(ArgumentError("Provide at least one node group"))
    analyses = [window_spike_pca(result;window,nodes=group) for group in groups]
    CairoMakie.activate!()
    fig = Figure(resolution=(600*length(groups)+100,550))
    colorrange = (Float64(first(window)),Float64(last(window)))
    for (j,(group,a)) in enumerate(zip(groups,analyses))
        label = group isa Symbol ? string(group) : "selected nodes"
        pct = round(100*a.explained_variance;digits=1)
        ax = Axis(fig[1,j];title="$label — PC1 ($pct% variance)",
            xlabel="Shape orientation (degrees)",ylabel="PC1 score",xticks=0:90:360)
        points,colors = Point2f[],Float64[]
        for i in 1:length(a.scores)-1
            abs(a.orientation_degrees[i+1]-a.orientation_degrees[i])>180 && continue
            push!(points,Point2f(a.orientation_degrees[i],a.scores[i]),
                Point2f(a.orientation_degrees[i+1],a.scores[i+1]))
            push!(colors,a.iterations[i],a.iterations[i+1])
        end
        linesegments!(ax,points;color=colors,colormap=:viridis,colorrange,linewidth=1.5)
        scatter!(ax,a.orientation_degrees,a.scores;color=a.iterations,
            colormap=:viridis,colorrange,markersize=3)
        xlims!(ax,0,360)
    end
    Colorbar(fig[1,length(groups)+1];colormap=:viridis,colorrange,label="Model iteration")
    Label(fig[0,1:length(groups)],"Iterations $(first(window))–$(last(window)) | centered spikes; PCA fitted within window";tellwidth=false)
    if savepath !== nothing
        mkpath(dirname(abspath(savepath)))
        save(savepath,fig)
    end
    return (;fig,analyses)
end
"""
    plot_spike_pc12(result_or_directory; window=5000:6000, savepath=nothing)

Fit PCA jointly to all reservoir neurons in the selected window and plot PC1
against PC2. Connect consecutive iterations, colored by iteration. Center each
neuron without scaling. Returns (;fig, analysis); analysis.component_scores has
one row per iteration and two columns. Fits are independent across windows.
"""
function plot_spike_pc12(source;window=5000:6000,savepath=nothing)
    result = source isa AbstractString ? load_drift_run(source) : source
    analysis = window_spike_pca(result;window,nodes=:all,ncomponents=2)
    a = analysis
    CairoMakie.activate!()
    fig = Figure(resolution=(850,700))
    pct = round.(100 .* a.explained_variances;digits=1)
    ax = Axis(fig[1,1];aspect=DataAspect(),title="Full-network spike trajectory",
        xlabel="PC1 ($(pct[1])% variance)",ylabel="PC2 ($(pct[2])% variance)")
    colorrange = (Float64(first(window)),Float64(last(window)))
    lines!(ax,a.component_scores[:,1],a.component_scores[:,2];
        color=a.iterations,colormap=:viridis,colorrange,linewidth=1.5)
    scatter!(ax,a.component_scores[:,1],a.component_scores[:,2];
        color=a.iterations,colormap=:viridis,colorrange,markersize=3)
    Colorbar(fig[1,2];colormap=:viridis,colorrange,label="Model iteration")
    Label(fig[0,1],"Iterations $(first(window))–$(last(window)) | all $(length(a.nodes)) neurons | centered spikes";tellwidth=false)
    if savepath !== nothing
        mkpath(dirname(abspath(savepath)))
        save(savepath,fig)
    end
    return (;fig,analysis)
end
"""
    plot_spike_pc12_time(result_or_directory; window=5000:6000, interactive=false,
        savepath=nothing, azimuth=1.55pi, elevation=pi/8)

Fit two PCs jointly across all neurons within the selected window. Plot model
iteration on x, PC1 on y, and PC2 on z, with line color indicating iteration.
Use interactive=true to open a rotatable GLMakie window with zoom/reset controls.
The default static figure uses CairoMakie. Center each neuron's spikes without
scaling. Returns (;fig, analysis, coordinates, axis, screen, zoom_slider, reset_button).
coordinates has columns [iteration, PC1, PC2]; analysis includes scores, loadings,
training means and explained variance. This does not run or advance the model.
backend=:cairo, show=false supports offscreen control checks.
"""
function plot_spike_pc12_time(source;window=5000:6000,savepath=nothing,
        azimuth=1.55pi,elevation=pi/8,interactive=false,
        backend=interactive ? :gl : :cairo,show=interactive)
    backend_module = if backend==:gl
        @eval import GLMakie
        Base.invokelatest(GLMakie.activate!;inline=false)
        GLMakie
    elseif backend==:cairo
        CairoMakie.activate!()
        CairoMakie
    else
        throw(ArgumentError("backend must be :gl or :cairo"))
    end
    return Base.invokelatest(_build_spike_pc12_time,source;window,savepath,
        azimuth,elevation,interactive,show,backend_module)
end

function _build_spike_pc12_time(source;window,savepath,azimuth,elevation,
        interactive,show,backend_module)
    result = source isa AbstractString ? load_drift_run(source) : source
    analysis = window_spike_pca(result;window,nodes=:all,ncomponents=2)
    a = analysis
    coordinates = hcat(a.iterations,a.component_scores)
    fig = Figure(resolution=(1000,850))
    pct = round.(100 .* a.explained_variances;digits=1)
    ax = Axis3(fig[1,1];title="Full-network PC1–PC2 trajectory through time",
        xlabel="Model iteration",
        ylabel="PC1 ($(pct[1])% variance)",zlabel="PC2 ($(pct[2])% variance)",
        azimuth,elevation,aspect=(1.6,1,1))
    colorrange = (Float64(first(window)),Float64(last(window)))
    lines!(ax,a.iterations,a.component_scores[:,1],a.component_scores[:,2];color=a.iterations,
        colormap=:viridis,colorrange,linewidth=2)
    scatter!(ax,a.iterations,a.component_scores[:,1],a.component_scores[:,2];color=a.iterations,
        colormap=:viridis,colorrange,markersize=3)
    Colorbar(fig[1,2];colormap=:viridis,colorrange,label="Model iteration")
    colgap!(fig.layout,1,90) # Leave room for the 3D axis labels beside the colorbar.
    Label(fig[0,1],"Iterations $(first(window))–$(last(window)) | all $(length(a.nodes)) neurons | centered spikes";tellwidth=false)
    zoom_slider = nothing
    reset_button = nothing
    if interactive
        controls = GridLayout()
        fig[2,1] = controls
        Label(controls[1,1],"Zoom")
        zoom_slider = Slider(controls[1,2];range=.25:.05:4.,startvalue=1.)
        reset_button = Button(controls[1,3];label="Reset view")
        bounds = [extrema(coordinates[:,j]) for j in 1:3]
        function apply_zoom!(zoom)
            limits = Float64[]
            for (lo,hi) in bounds
                center = (lo+hi)/2
                radius = max((hi-lo)/2, .5)*1.05/zoom
                append!(limits,(center-radius,center+radius))
            end
            limits!(ax,limits...)
        end
        on(apply_zoom!,zoom_slider.value)
        apply_zoom!(1.)
        on(reset_button.clicks) do _
            ax.azimuth[] = azimuth
            ax.elevation[] = elevation
            set_close_to!(zoom_slider,1.)
            apply_zoom!(1.)
        end
        Label(fig[3,1],"Drag with the left mouse button to rotate | Slider to zoom | Reset view to restore";tellwidth=false)
    end
    if savepath !== nothing
        mkpath(dirname(abspath(savepath)))
        save(savepath,fig)
    end
    screen = show ? display(fig;backend=backend_module,inline=false,
        title="Full-network PC1–PC2 trajectory through time") : nothing
    return (;fig,analysis,coordinates,axis=ax,screen,zoom_slider,reset_button)
end
"""
    spike_hamming_distances(result_or_directory; window=5000:6000,
        stride=1, nodes=:all)

Time-by-time count of differing binary neuron values. No PCA, normalization or
mean subtraction is applied. Identical vectors, including silent ones, have
zero distance. Pairwise distances are computed using packed UInt64 words and
popcounts. Storage is UInt8 for up to 255 selected neurons, UInt16 for up to
65535, then UInt32 or UInt64. The symmetric matrix includes a zero diagonal.
stride samples every kth iteration from first(window), without averaging.
Returns (;distances, iterations, nodes).
"""
function spike_hamming_distances(source;window=5000:6000,stride=1,nodes=:all)
    result = source isa AbstractString ? load_drift_run(source) : source
    spikes = result.ts.spikes
    n,nneurons = size(spikes)
    window isa AbstractUnitRange{<:Integer} && !isempty(window) &&
        first(window)>=1 && last(window)<=n ||
        throw(ArgumentError("window must be a nonempty consecutive range within completed iterations 1:$n"))
    stride isa Integer && !(stride isa Bool) && stride>=1 ||
        throw(ArgumentError("stride must be a positive integer"))
    iterations = collect(first(window):stride:last(window))
    length(iterations)>=2 || throw(ArgumentError("Select at least two iterations after sampling"))
    if nodes===:all
        ids = collect(1:nneurons)
    elseif nodes isa AbstractVector{<:Integer}
        ids = collect(nodes)
    else
        throw(ArgumentError("nodes must be :all or a vector of indices"))
    end
    !isempty(ids) && all(i->1<=i<=nneurons,ids) && allunique(ids) ||
        throw(ArgumentError("Select nonempty, unique neuron indices in bounds"))
    selected = spikes[iterations,ids]
    all(x->x==0 || x==1,selected) || throw(ArgumentError("Hamming analysis requires binary spikes (0 or 1)"))
    nframes,nselected = size(selected)
    packed = zeros(UInt64,cld(nselected,64),nframes)
    @inbounds for node in 1:nselected
        word = div(node-1,64)+1
        mask = UInt64(1)<<mod(node-1,64)
        for frame in 1:nframes
            selected[frame,node]!=0 && (packed[word,frame] |= mask)
        end
    end
    T = nselected<=typemax(UInt8) ? UInt8 : nselected<=typemax(UInt16) ? UInt16 :
        nselected<=typemax(UInt32) ? UInt32 : UInt64
    distances = zeros(T,nframes,nframes)
    @inbounds for j in 1:nframes-1
        for i in j+1:nframes
            d = 0
            for word in axes(packed,1)
                d += count_ones(xor(packed[word,i],packed[word,j]))
            end
            distances[i,j] = distances[j,i] = T(d)
        end
    end
    return (;distances,iterations,nodes=ids)
end

"""
    plot_spike_hamming_heatmap(result_or_directory; window=5000:6000,
        stride=1, nodes=:all, savepath=nothing)

Static time-by-time heatmap of Hamming count distances between network spike
vectors. Both axes use original model iterations. Reversed jet maps zero to red, increasing through
yellow, green and cyan to dark blue at the largest observed distance. Identical/silent patterns are valid. Returns the
figure and matrix data as (;fig, distances, iterations, nodes). For long windows,
stride reduces quadratic pairwise computation and storage without averaging.
"""
function plot_spike_hamming_heatmap(source;window=5000:6000,stride=1,nodes=:all,savepath=nothing)
    data = spike_hamming_distances(source;window,stride,nodes)
    observed_max = Int(maximum(data.distances))
    CairoMakie.activate!()
    fig = Figure(resolution=(950,850))
    ax = Axis(fig[1,1];aspect=DataAspect(),xlabel="Model iteration",
        ylabel="Model iteration",title="Spike-pattern Hamming distance across time")
    hm = heatmap!(ax,data.iterations,data.iterations,data.distances;
        colormap=Reverse(:jet),colorrange=(0,max(1,observed_max)),interpolate=false)
    if observed_max==0
        Colorbar(fig[1,2],hm;label="Hamming distance (differing neurons)",ticks=[0])
    else
        Colorbar(fig[1,2],hm;label="Hamming distance (differing neurons)")
    end
    Label(fig[0,1],"Iterations $(first(window))–$(last(window)) | $(length(data.iterations)) frames | $(length(data.nodes)) neurons";tellwidth=false)
    caption = observed_max==0 ? "All selected spike vectors are identical (distance 0)" :
        "Red: identical patterns | Dark blue: largest observed distance ($observed_max neurons)"
    Label(fig[2,1],caption;tellwidth=false)
    if savepath !== nothing
        mkpath(dirname(abspath(savepath)))
        save(savepath,fig)
    end
    return (;fig,data...)
end
# Included inside RotatingShapeSingleDrift; uses its imports and saved-run loader.

function _bin_mean_correlations(means,counts)
    nbins,nneurons = size(means)
    valid_bins = falses(nbins)
    status = fill(:empty,nbins)
    normalized = zeros(nbins,nneurons)
    for b in 1:nbins
        counts[b]==0 && continue
        row = @view means[b,:]
        status[b] = :constant
        all(row .== row[1]) && continue
        centered = row .- mean(row)
        magnitude = norm(centered)
        magnitude>0 || continue
        normalized[b,:] .= centered ./ magnitude
        valid_bins[b] = true
        status[b] = :valid
    end
    correlations = fill(NaN32,nbins,nbins)
    for j in 1:nbins
        valid_bins[j] || continue
        correlations[j,j] = 1f0
        for i in j+1:nbins
            valid_bins[i] || continue
            r = clamp(dot(@view(normalized[i,:]),@view(normalized[j,:])),-1.,1.)
            correlations[i,j] = correlations[j,i] = Float32(r)
        end
    end
    return (;correlations,valid_bins,status)
end

"""
    binned_orientation_correlations(result_or_directory; warmup=1000,
        bin_size=1200, orientations=(90.,270.), tolerance_degrees=1.)

Discard rows 1:warmup. Select matching orientations by circular angular distance.
Bins are anchored immediately after warmup: 1001:2200, 2201:3400, ... by default,
with the final partial bin retained. Each bin's mean is computed only over the
selected frames at that orientation, with no thresholding. Pearson correlations
compare mean vectors across neurons, separately for each orientation. Empty bins
and constant mean vectors have undefined correlations (NaN). Return all selected
spikes/metadata, bin ranges/counts/means and correlation matrices. Original row
indices identify model iterations, just as in the existing PCA plots.
"""
function binned_orientation_correlations(source;warmup=1000,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.)
    result = source isa AbstractString ? load_drift_run(source) : source
    ts = result.ts
    n,nneurons = size(ts.spikes)
    length(ts.angle)==n || throw(ArgumentError("One orientation is required per spike row"))
    nneurons>=2 || throw(ArgumentError("Pearson correlations require at least two neurons"))
    warmup isa Integer && !(warmup isa Bool) && 0<=warmup<n ||
        throw(ArgumentError("warmup must be a nonnegative integer smaller than the number of completed iterations ($n)"))
    bin_size isa Integer && !(bin_size isa Bool) && bin_size>=1 ||
        throw(ArgumentError("bin_size must be a positive integer"))
    isfinite(tolerance_degrees) && 0<=tolerance_degrees<180 ||
        throw(ArgumentError("tolerance_degrees must be finite and in [0,180)"))
    targets = mod.(Float64.(collect(orientations)),360.)
    !isempty(targets) && all(isfinite,targets) && allunique(targets) ||
        throw(ArgumentError("Provide finite, distinct target orientations"))
    for i in eachindex(targets),j in i+1:length(targets)
        separation = abs(mod(targets[i]-targets[j]+180.,360.)-180.)
        2*tolerance_degrees+2e-8<separation ||
            throw(ArgumentError("Orientation tolerance windows overlap; reduce tolerance_degrees"))
    end
    remaining = collect(warmup+1:n)
    angles = rad2deg.(mod.(Float64.(ts.angle[remaining]),2pi))
    all(isfinite,angles) || throw(ArgumentError("Orientations after warmup must be finite"))
    bin_starts = collect(warmup+1:bin_size:n)
    nbins = length(bin_starts)
    bin_ends = min.(bin_starts .+ bin_size .- 1,n)
    bin_ids = collect(1:nbins)
    bins = (;ids=bin_ids,starts=bin_starts,ends=bin_ends,
        is_partial=(bin_ends.-bin_starts.+1) .< bin_size)
    selected_iterations,selected_bin_ids = Int[],Int[]
    selected_angles,selected_targets = Float64[],Float64[]
    for (i,iteration) in enumerate(remaining)
        for target in targets
            distance = abs(mod(angles[i]-target+180.,360.)-180.)
            if distance<=tolerance_degrees+1e-8
                push!(selected_iterations,iteration)
                push!(selected_bin_ids,div(iteration-warmup-1,bin_size)+1)
                push!(selected_angles,angles[i])
                push!(selected_targets,target)
                break
            end
        end
    end
    raw_spikes = ts.spikes[selected_iterations,:]
    all(x->x==0 || x==1,raw_spikes) || throw(ArgumentError("Selected spike vectors must be binary (0 or 1)"))
    direction = nothing
    if hasproperty(ts,:rotation_direction)
        length(ts.rotation_direction)==n || throw(ArgumentError("Rotation direction rows must match spike rows"))
        direction = copy(ts.rotation_direction[selected_iterations])
    end
    samples = (;iterations=selected_iterations,orientation_degrees=selected_angles,
        orientation_group_degrees=selected_targets,bin_ids=selected_bin_ids,
        spikes=UInt8.(raw_spikes),rotation_direction=direction)
    groups = Dict{Float64,NamedTuple}()
    for target in targets
        counts = zeros(Int,nbins)
        sums = zeros(nbins,nneurons)
        for i in eachindex(selected_iterations)
            selected_targets[i]==target || continue
            b = selected_bin_ids[i]
            counts[b] += 1
            sums[b,:] .+= @view samples.spikes[i,:]
        end
        means = fill(NaN,nbins,nneurons)
        for b in 1:nbins
            counts[b]>0 && (means[b,:] .= sums[b,:] ./ counts[b])
        end
        correlation = _bin_mean_correlations(means,counts)
        groups[target] = (;orientation_degrees=target,counts,mean_spikes=means,
            correlation.correlations,correlation.valid_bins,correlation.status)
    end
    return (;samples,bins,groups,orientations=targets,warmup,bin_size,
        tolerance_degrees=Float64(tolerance_degrees),n_iterations=n,nneurons)
end

_orientation_bin_label(angle) = isinteger(angle) ? string(Int(angle)) : string(angle)

function _new_orientation_analysis_directory(root)
    mkpath(root)
    index = 1
    while true
        path = joinpath(root,"run_"*lpad(string(index),4,'0'))
        if !ispath(path)
            mkdir(path)
            return abspath(path)
        end
        index += 1
    end
end

function _write_orientation_csv(path,columns,data)
    open(path,"w") do io
        writedlm(io,permutedims(columns),',')
        writedlm(io,data,',')
    end
end

function _save_binned_orientation_analysis(a,fig,output_dir)
    dir = _new_orientation_analysis_directory(output_dir)
    s,b = a.samples,a.bins
    node_names = ["node_$j" for j in 1:a.nneurons]
    sample_headers = ["iteration","orientation_degrees","orientation_group_degrees","bin"]
    sample_metadata = hcat(s.iterations,s.orientation_degrees,s.orientation_group_degrees,s.bin_ids)
    if s.rotation_direction !== nothing
        push!(sample_headers,"rotation_direction")
        sample_metadata = hcat(sample_metadata,s.rotation_direction)
    end
    _write_orientation_csv(joinpath(dir,"selected_spikes.csv"),vcat(sample_headers,node_names),
        hcat(sample_metadata,s.spikes))
    _write_orientation_csv(joinpath(dir,"bins.csv"),["bin","iteration_start","iteration_end","is_partial"],
        hcat(b.ids,b.starts,b.ends,b.is_partial))
    for target in a.orientations
        g = a.groups[target]
        label = replace(_orientation_bin_label(target),'.'=>'p')*"deg"
        _write_orientation_csv(joinpath(dir,"mean_spikes_"*label*".csv"),
            vcat(["bin","iteration_start","iteration_end","n_samples","pearson_status"],node_names),
            hcat(b.ids,b.starts,b.ends,g.counts,string.(g.status),g.mean_spikes))
        _write_orientation_csv(joinpath(dir,"correlations_"*label*".csv"),
            vcat(["bin"],["bin_$i" for i in b.ids]),hcat(b.ids,g.correlations))
    end
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:warmup,:bin_size,:orientations,:tolerance_degrees,:n_iterations,:nneurons)
            println(io,key," = ",getproperty(a,key))
        end
        println(io,"bin_anchor = first post-warmup iteration (warmup + 1)")
        println(io,"final_partial_bin = retained")
        println(io,"selected_spike_rows = ",length(s.iterations))
        println(io,"correlation = Pearson across neurons between raw bin-mean spike vectors")
        println(io,"directions = pooled within orientation; retained in samples when available")
        println(io,"undefined = NaN (empty bin or constant mean vector)")
    end
    CairoMakie.save(joinpath(dir,"Binned_orientation_correlations.png"),fig)
    return dir
end

# Makie heatmaps index x first and y second. Keep the visual upper triangle
# (y >= x), including the diagonal. Reserve lowclip for the hidden cells.
function _upper_correlation_display(xy)
    return [y < x ? -2.0 : max(0.0, xy[x,y])
        for x in axes(xy,1), y in axes(xy,2)]
end

"""
    plot_binned_orientation_correlations(result_or_directory; warmup=1000,
        bin_size=1200, orientations=(90.,270.), tolerance_degrees=1.,
        save_outputs=true, output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationBins"))

Plot one bin-by-bin Pearson heatmap per orientation using a shared 0:1 display scale. Negative correlations remain in the data but display
at the lowest color.
Warmup is removed before angle selection and averaging. Pass a precomputed
binned_orientation_correlations result as source to plot without recomputation.
By default save selected spike vectors/metadata, bin means/counts, correlations,
parameters and the figure in a new numbered output folder. With save_outputs=false
nothing is written. Returns (;fig, analysis, output_dir).
"""
function plot_binned_orientation_correlations(source;warmup=1000,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationBins"))
    a = hasproperty(source,:groups) && hasproperty(source,:samples) && hasproperty(source,:bins) ? source :
        binned_orientation_correlations(source;warmup,bin_size,orientations,tolerance_degrees)
    CairoMakie.activate!()
    ngroups,nbins = length(a.orientations),length(a.bins.ids)
    fig = Figure(resolution=(620*ngroups+100,730))
    edges = collect(.5:1.0:(nbins+.5))
    tick_ids = nbins<=12 ? a.bins.ids : unique(round.(Int,range(1,nbins;length=10)))
    for (j,target) in enumerate(a.orientations)
        g = a.groups[target]
        label = _orientation_bin_label(target)
        ax = Axis(fig[1,j];aspect=DataAspect(),title="Autocorrelation: $(label)°",
            xlabel="Time bin",ylabel="Time bin",xticks=tick_ids,yticks=tick_ids)
        heatmap!(ax,edges,edges,_upper_correlation_display(g.correlations);
            colormap=:jet,colorrange=(0,1),lowclip=:transparent,
            nan_color=:lightgray,interpolate=false)
    end
    Colorbar(fig[1,ngroups+1];colormap=:jet,colorrange=(0,1),label="Pearson correlation",ticks=0:.2:1)
    dir = save_outputs ? _save_binned_orientation_analysis(a,fig,output_dir) : nothing
    return (;fig,analysis=a,output_dir=dir)
end

"""
    binned_orientation_cosine_similarity(source; warmup=1000, bin_size=1200,
        tolerance_degrees=1.)

Compare the raw 90-degree and 270-degree average spike vectors within each bin:
dot(mean90,mean270)/(norm(mean90)*norm(mean270)). No centering or thresholding.
source may be a result, saved-run directory, precomputed binned analysis, or the
output of plot_binned_orientation_correlations. Existing binning is reused when
provided. Empty orientation/bin groups and zero vectors yield NaN, not zero.
Nonzero constant vectors are valid here even when Pearson correlation is undefined.
Returns (;analysis, bins, cosine_similarity, valid_bins, status, counts_90, counts_270).
"""
function binned_orientation_cosine_similarity(source;warmup=1000,bin_size=1200,tolerance_degrees=1.)
    base = hasproperty(source,:analysis) ? source.analysis : source
    a = hasproperty(base,:groups) && hasproperty(base,:bins) ? base :
        binned_orientation_correlations(base;warmup,bin_size,orientations=(90.,270.),tolerance_degrees)
    haskey(a.groups,90.) && haskey(a.groups,270.) ||
        throw(ArgumentError("The binned analysis must include both 90 and 270 degrees"))
    first,second = a.groups[90.],a.groups[270.]
    nbins = length(a.bins.ids)
    size(first.mean_spikes)==size(second.mean_spikes) && size(first.mean_spikes,1)==nbins ||
        throw(ArgumentError("Orientation mean matrices must have matching neuron columns and bin rows"))
    values = fill(NaN,nbins)
    valid_bins = falses(nbins)
    status = fill(:empty,nbins)
    for b in 1:nbins
        (first.counts[b]==0 || second.counts[b]==0) && continue
        u,v = @view(first.mean_spikes[b,:]),@view(second.mean_spikes[b,:])
        all(isfinite,u) && all(isfinite,v) || throw(ArgumentError("Nonempty bin means must be finite"))
        nu,nv = norm(u),norm(v)
        status[b] = :zero_vector
        (iszero(nu) || iszero(nv)) && continue
        values[b] = clamp(dot(u,v)/(nu*nv),-1.,1.)
        valid_bins[b] = true
        status[b] = :valid
    end
    return (;analysis=a,bins=a.bins,cosine_similarity=values,valid_bins,status,
        counts_90=copy(first.counts),counts_270=copy(second.counts))
end

function _save_orientation_cosine(data,fig,output_dir)
    dir = _new_orientation_analysis_directory(output_dir)
    b,a = data.bins,data.analysis
    _write_orientation_csv(joinpath(dir,"cosine_similarity_by_bin.csv"),
        ["bin","iteration_start","iteration_end","n_samples_90deg","n_samples_270deg",
            "cosine_similarity","valid","status"],
        hcat(b.ids,b.starts,b.ends,data.counts_90,data.counts_270,data.cosine_similarity,
            data.valid_bins,string.(data.status)))
    for target in (90.,270.)
        group = a.groups[target]
        _write_orientation_csv(joinpath(dir,"mean_spikes_"*_orientation_bin_label(target)*"deg.csv"),
            vcat(["bin","iteration_start","iteration_end","n_samples"],["node_$j" for j in axes(group.mean_spikes,2)]),
            hcat(b.ids,b.starts,b.ends,group.counts,group.mean_spikes))
    end
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:warmup,:bin_size,:tolerance_degrees,:n_iterations,:nneurons)
            println(io,key," = ",getproperty(a,key))
        end
        println(io,"orientation_pair = 90 degrees vs 270 degrees within the same bin")
        println(io,"formula = dot(mean90, mean270) / (norm(mean90) * norm(mean270))")
        println(io,"centering = none; thresholding = none")
        println(io,"undefined = NaN (empty group or zero-length mean vector)")
        println(io,"directions = pooled within orientation, as in the binned analysis")
    end
    CairoMakie.save(joinpath(dir,"Orientation_cosine_similarity.png"),fig)
    return dir
end

"""
    plot_orientation_cosine_similarity(source; warmup=1000, bin_size=1200,
        tolerance_degrees=1., save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationCosine"))

Plot 90-vs-270 degree mean-vector cosine similarity against time bin. Accept the
existing p_bins or p_bins.analysis to reuse exact averages, or a result/saved-run
directory to compute them. Missing/zero-vector bins form gaps; a shared 0:1 y range
is used for nonnegative spike fractions. Save curve, values/counts/status and both
mean matrices in a new numbered folder unless save_outputs=false. Returns
(;fig, analysis, bins, cosine_similarity, valid_bins, status, counts_90, counts_270, output_dir).
"""
function plot_orientation_cosine_similarity(source;warmup=1000,bin_size=1200,
        tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationCosine"))
    data = binned_orientation_cosine_similarity(source;warmup,bin_size,tolerance_degrees)
    a,b = data.analysis,data.bins
    CairoMakie.activate!()
    fig = Figure(resolution=(1000,580))
    tick_ids = length(b.ids)<=15 ? b.ids : unique(round.(Int,range(first(b.ids),last(b.ids);length=10)))
    ax = Axis(fig[1,1];title="Within-bin cosine similarity: 90° vs 270°",
        xlabel="Time bin",ylabel="Cosine similarity",xticks=tick_ids,yticks=0:.2:1)
    lines!(ax,b.ids,data.cosine_similarity;color=:royalblue,linewidth=2)
    scatter!(ax,b.ids[data.valid_bins],data.cosine_similarity[data.valid_bins];
        color=:royalblue,markersize=10)
    xlims!(ax,first(b.ids)-.5,last(b.ids)+.5)
    ylims!(ax,-.03,1.03)
    if !any(data.valid_bins)
        text!(ax,(first(b.ids)+last(b.ids))/2,.5;
            text="No defined cosine similarities\nEach bin is missing an orientation or has a zero vector",
            align=(:center,:center),fontsize=17)
    end
    Label(fig[0,1],"Warmup: $(a.warmup) | $(a.bin_size) iterations/bin | Bin 1 starts at $(a.warmup+1) | Angle tolerance ±$(a.tolerance_degrees)°";tellwidth=false)
    Label(fig[2,1],"Raw mean vectors; no centering | $(count(!,data.valid_bins)) undefined bins omitted (empty group or zero vector)";tellwidth=false)
    dir = save_outputs ? _save_orientation_cosine(data,fig,output_dir) : nothing
    return (;fig,data...,output_dir=dir)
end

"""
    binned_orientation_cosine_matrix(source; warmup=1000, bin_size=1200,
        tolerance_degrees=1.)

All cross-bin cosines between the 90-degree and 270-degree bin-average vectors.
cosine_matrix[j,i] compares 270 degrees in bin j with 90 degrees in bin i.
Rows represent 270 degrees; columns represent 90 degrees. The diagonal equals
the within-bin cosine curve, not necessarily one; the matrix is generally not
symmetric. Normalize raw means without centering. Empty or zero-vector groups
produce NaN rows/columns, independently for each orientation. Accepts the same
sources as binned_orientation_cosine_similarity, including existing plot results.
"""
function binned_orientation_cosine_matrix(source;warmup=1000,bin_size=1200,tolerance_degrees=1.)
    resolved = binned_orientation_cosine_similarity(source;warmup,bin_size,tolerance_degrees)
    a,b = resolved.analysis,resolved.bins
    nbins,nneurons = size(a.groups[90.].mean_spikes)
    normalized = Dict{Float64,Matrix{Float64}}()
    valid = Dict{Float64,BitVector}()
    status = Dict{Float64,Vector{Symbol}}()
    for target in (90.,270.)
        g = a.groups[target]
        normalized[target] = zeros(nbins,nneurons)
        valid[target] = falses(nbins)
        status[target] = fill(:empty,nbins)
        for bin in 1:nbins
            g.counts[bin]==0 && continue
            row = @view g.mean_spikes[bin,:]
            all(isfinite,row) || throw(ArgumentError("Nonempty bin means must be finite"))
            magnitude = norm(row)
            status[target][bin] = :zero_vector
            iszero(magnitude) && continue
            normalized[target][bin,:] .= row ./ magnitude
            valid[target][bin] = true
            status[target][bin] = :valid
        end
    end
    cosine_matrix = normalized[270.]*transpose(normalized[90.])
    clamp!(cosine_matrix,-1.,1.)
    cosine_matrix[.!valid[270.],:] .= NaN
    cosine_matrix[:,.!valid[90.]] .= NaN
    return (;analysis=a,bins=b,cosine_matrix,valid_90=valid[90.],valid_270=valid[270.],
        status_90=status[90.],status_270=status[270.],
        counts_90=copy(a.groups[90.].counts),counts_270=copy(a.groups[270.].counts))
end

function _save_orientation_cosine_matrix(data,fig,output_dir)
    dir = _new_orientation_analysis_directory(output_dir)
    a,b = data.analysis,data.bins
    _write_orientation_csv(joinpath(dir,"cosine_matrix_270rows_90cols.csv"),
        vcat(["bin_270deg"],["bin_90deg_$i" for i in b.ids]),hcat(b.ids,data.cosine_matrix))
    for target in (90.,270.)
        group = a.groups[target]
        status = target==90. ? data.status_90 : data.status_270
        _write_orientation_csv(joinpath(dir,"mean_spikes_"*_orientation_bin_label(target)*"deg.csv"),
            vcat(["bin","iteration_start","iteration_end","n_samples","cosine_status"],
                ["node_$j" for j in axes(group.mean_spikes,2)]),
            hcat(b.ids,b.starts,b.ends,group.counts,string.(status),group.mean_spikes))
    end
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:warmup,:bin_size,:tolerance_degrees,:n_iterations,:nneurons)
            println(io,key," = ",getproperty(a,key))
        end
        println(io,"matrix_rows = 270-degree time bins")
        println(io,"matrix_columns = 90-degree time bins")
        println(io,"plot_x = 90-degree time bins; plot_y = 270-degree time bins")
        println(io,"formula = dot(mean90_i, mean270_j) / (norm(mean90_i) * norm(mean270_j))")
        println(io,"centering = none; thresholding = none")
        println(io,"undefined = NaN (empty orientation/bin group or zero-length mean vector)")
        println(io,"directions = pooled within orientation, as in the binned analysis")
    end
    CairoMakie.save(joinpath(dir,"Between_bin_orientation_cosine.png"),fig)
    return dir
end

"""
    plot_orientation_cosine_heatmap(source; warmup=1000, bin_size=1200,
        tolerance_degrees=1., save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationCosineMatrix"))

Cross-orientation, cross-bin cosine heatmap: x is the 90-degree time bin and y
is the 270-degree time bin. Uses a fixed 0:1 jet scale, blue low and red high;
gray is undefined. Accept an existing p_bins, p_cosine, or analysis to reuse means,
or compute from a model result/saved-run directory. Save figure, matrix, bin means
and metadata to a new numbered folder unless save_outputs=false. Matrix storage
follows conventional rows=y (270 degrees), columns=x (90 degrees). Returns figure,
all matrix data, axis, heatmap plot and output_dir.
"""
function plot_orientation_cosine_heatmap(source;warmup=1000,bin_size=1200,
        tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationCosineMatrix"))
    data = binned_orientation_cosine_matrix(source;warmup,bin_size,tolerance_degrees)
    a,b = data.analysis,data.bins
    nbins = length(b.ids)
    edges = collect(.5:1.0:(nbins+.5))
    ticks = nbins<=15 ? b.ids : unique(round.(Int,range(first(b.ids),last(b.ids);length=10)))
    CairoMakie.activate!()
    fig = Figure(resolution=(1000,850))
    ax = Axis(fig[1,1];aspect=DataAspect(),title="Between-bin cosine similarity: 90° vs 270°",
        xlabel="90° time bin",ylabel="270° time bin",xticks=ticks,yticks=ticks)
    # Makie uses matrix dimension 1 for x. Our returned/CSV matrix uses rows=y.
    hm = heatmap!(ax,edges,edges,permutedims(data.cosine_matrix);
        colormap=:jet,colorrange=(0,1),nan_color=:lightgray,interpolate=false)
    Colorbar(fig[1,2],hm;label="Cosine similarity",ticks=0:.2:1)
    Label(fig[0,1],"Warmup: $(a.warmup) | $(a.bin_size) iterations/bin | Bin 1 starts at $(a.warmup+1) | Angle tolerance ±$(a.tolerance_degrees)°";tellwidth=false)
    Label(fig[2,1],"Raw mean vectors | Diagonal: same-bin comparison | Gray: undefined (empty group or zero vector)";tellwidth=false)
    dir = save_outputs ? _save_orientation_cosine_matrix(data,fig,output_dir) : nothing
    return (;fig,data...,axis=ax,heatmap=hm,output_dir=dir)
end
# Included inside RotatingShapeSingleDrift after the binned analysis helpers.

function _resolve_orientation_bin_analysis(source;warmup,bin_size,orientations,tolerance_degrees)
    base = hasproperty(source,:analysis) ? source.analysis : source
    return hasproperty(base,:groups) && hasproperty(base,:bins) ? base :
        binned_orientation_correlations(base;warmup,bin_size,orientations,tolerance_degrees)
end

function _bin_mean_cosines(means,counts)
    nbins,nneurons = size(means)
    normalized = zeros(nbins,nneurons)
    valid_bins = falses(nbins)
    status = fill(:empty,nbins)
    for b in 1:nbins
        counts[b]==0 && continue
        row = @view means[b,:]
        all(isfinite,row) || throw(ArgumentError("Nonempty bin means must be finite"))
        magnitude = norm(row)
        status[b] = :zero_vector
        iszero(magnitude) && continue
        normalized[b,:] .= row ./ magnitude
        valid_bins[b] = true
        status[b] = :valid
    end
    cosine_matrix = normalized*transpose(normalized)
    clamp!(cosine_matrix,-1.,1.)
    cosine_matrix[.!valid_bins,:] .= NaN
    cosine_matrix[:,.!valid_bins] .= NaN
    for b in findall(valid_bins)
        cosine_matrix[b,b] = 1.
    end
    return (;cosine_matrix,valid_bins,status)
end

"""
    binned_orientation_cosines(source; warmup=1000, bin_size=1200,
        orientations=(90.,270.), tolerance_degrees=1.)

Per-orientation bin-by-bin cosine matrices of raw average spike vectors. Reuse
p_bins/analysis/another binned plot result, or compute from a result or run folder.
No centering or thresholding. Empty/zero-vector bins are NaN; nonzero constant
vectors are valid. Returns (;analysis, bins, orientations, groups), where each
group contains cosine_matrix, valid_bins, status and counts.
"""
function binned_orientation_cosines(source;warmup=1000,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.)
    a = _resolve_orientation_bin_analysis(source;warmup,bin_size,orientations,tolerance_degrees)
    groups = Dict{Float64,NamedTuple}()
    for target in a.orientations
        g = a.groups[target]
        similarities = _bin_mean_cosines(g.mean_spikes,g.counts)
        groups[target] = (;similarities...,counts=copy(g.counts))
    end
    return (;analysis=a,bins=a.bins,orientations=a.orientations,groups)
end

function _write_variant_parameters(dir,a,metric,comparison)
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:warmup,:bin_size,:tolerance_degrees,:n_iterations,:nneurons)
            println(io,key," = ",getproperty(a,key))
        end
        println(io,"metric = ",metric)
        println(io,"comparison = ",comparison)
        println(io,"thresholding = none")
        println(io,"directions = pooled within orientation")
    end
end

function _write_variant_means(dir,a,targets)
    for target in targets
        g = a.groups[target]
        _write_orientation_csv(joinpath(dir,"mean_spikes_"*_orientation_bin_label(target)*"deg.csv"),
            vcat(["bin","iteration_start","iteration_end","n_samples"],
                ["node_$j" for j in axes(g.mean_spikes,2)]),
            hcat(a.bins.ids,a.bins.starts,a.bins.ends,g.counts,g.mean_spikes))
    end
end

"""
    plot_binned_orientation_cosines(source; warmup=1000, bin_size=1200,
        orientations=(90.,270.), tolerance_degrees=1., save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationBinCosine"))

Cosine versions of the per-orientation bin-by-bin heatmaps. Both panels share a
0:1 jet color scale. Gray denotes an empty bin or zero mean vector. Existing
p_bins averages are reused without changing selection/binning. Save matrix CSVs,
means, bin statuses and figure in a new numbered folder unless save_outputs=false.
Returns (;fig, analysis, bins, orientations, groups, output_dir).
"""
function plot_binned_orientation_cosines(source;warmup=1000,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationBinCosine"))
    data = binned_orientation_cosines(source;warmup,bin_size,orientations,tolerance_degrees)
    a,b = data.analysis,data.bins
    nbins,ngroups = length(b.ids),length(data.orientations)
    edges = collect(.5:1.0:(nbins+.5))
    ticks = nbins<=12 ? b.ids : unique(round.(Int,range(first(b.ids),last(b.ids);length=10)))
    CairoMakie.activate!()
    fig = Figure(resolution=(620*ngroups+100,730))
    for (j,target) in enumerate(data.orientations)
        g = data.groups[target]
        label = _orientation_bin_label(target)
        ax = Axis(fig[1,j];aspect=DataAspect(),title="$(label)° — cosine of bin-average spikes",
            xlabel="Time bin",ylabel="Time bin",xticks=ticks,yticks=ticks)
        heatmap!(ax,edges,edges,g.cosine_matrix;colormap=:jet,colorrange=(0,1),
            nan_color=:lightgray,interpolate=false)
        Label(fig[2,j],"$(sum(g.counts)) selected frames | $(count(!,g.valid_bins)) undefined bins";tellwidth=false)
    end
    Colorbar(fig[1,ngroups+1];colormap=:jet,colorrange=(0,1),label="Cosine similarity",ticks=0:.2:1)
    Label(fig[0,1:ngroups],"Warmup: $(a.warmup) | $(a.bin_size) iterations/bin | Bin 1 starts at $(a.warmup+1) | Angle tolerance ±$(a.tolerance_degrees)°";tellwidth=false)
    Label(fig[3,1:ngroups],"Raw means; no centering | Blue: 0 | Red: 1 | Gray: undefined (empty group or zero vector)";tellwidth=false)
    dir = nothing
    if save_outputs
        dir = _new_orientation_analysis_directory(output_dir)
        _write_variant_means(dir,a,data.orientations)
        for target in data.orientations
            g = data.groups[target]
            label = _orientation_bin_label(target)*"deg"
            _write_orientation_csv(joinpath(dir,"cosine_matrix_"*label*".csv"),
                vcat(["bin"],["bin_$i" for i in b.ids]),hcat(b.ids,g.cosine_matrix))
            _write_orientation_csv(joinpath(dir,"bin_status_"*label*".csv"),
                ["bin","iteration_start","iteration_end","n_samples","valid","status"],
                hcat(b.ids,b.starts,b.ends,g.counts,g.valid_bins,string.(g.status)))
        end
        _write_variant_parameters(dir,a,"cosine of uncentered raw bin means","between bins, within each orientation")
        CairoMakie.save(joinpath(dir,"Binned_orientation_cosines.png"),fig)
    end
    return (;fig,data...,output_dir=dir)
end

"""
    binned_orientation_correlation(source; warmup=1000, bin_size=1200,
        tolerance_degrees=1.)

Pearson correlation between the 90-degree and 270-degree mean vectors WITHIN each
bin, across neurons. This is a cross-orientation correlation, not a lag-averaged
autocorrelation function. Reuse existing binned means, or compute from a run result.
Empty groups or constant mean vectors have NaN correlation. Negative correlations
are retained. Returns (;analysis, bins, correlation, valid_bins, status, counts_90, counts_270).
"""
function binned_orientation_correlation(source;warmup=1000,bin_size=1200,tolerance_degrees=1.)
    a = _resolve_orientation_bin_analysis(source;warmup,bin_size,orientations=(90.,270.),tolerance_degrees)
    haskey(a.groups,90.) && haskey(a.groups,270.) ||
        throw(ArgumentError("The binned analysis must include both 90 and 270 degrees"))
    ugroup,vgroup = a.groups[90.],a.groups[270.]
    nbins = length(a.bins.ids)
    values = fill(NaN,nbins)
    valid_bins = falses(nbins)
    status = fill(:empty,nbins)
    for b in 1:nbins
        (ugroup.counts[b]==0 || vgroup.counts[b]==0) && continue
        u,v = @view(ugroup.mean_spikes[b,:]),@view(vgroup.mean_spikes[b,:])
        all(isfinite,u) && all(isfinite,v) || throw(ArgumentError("Nonempty bin means must be finite"))
        status[b] = :constant
        (all(u .== u[1]) || all(v .== v[1])) && continue
        cu,cv = u .- mean(u),v .- mean(v)
        nu,nv = norm(cu),norm(cv)
        (iszero(nu) || iszero(nv)) && continue
        values[b] = clamp(dot(cu,cv)/(nu*nv),-1.,1.)
        valid_bins[b] = true
        status[b] = :valid
    end
    return (;analysis=a,bins=a.bins,correlation=values,valid_bins,status,
        counts_90=copy(ugroup.counts),counts_270=copy(vgroup.counts))
end

"""
    plot_orientation_correlation(source; warmup=1000, bin_size=1200,
        tolerance_degrees=1., save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationCorrelation"))

Pearson version of the within-bin 90-versus-270 cosine curve. X is time bin;
y is Pearson r on -1:1, retaining negative correlations. Undefined bins form gaps.
Reuse p_bins means or compute from a result/run directory. Save curve, values,
counts/status and mean vectors to a new numbered folder unless save_outputs=false.
Returns (;fig, analysis, bins, correlation, valid_bins, status, counts_90, counts_270, output_dir).
"""
function plot_orientation_correlation(source;warmup=1000,bin_size=1200,
        tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationCorrelation"))
    data = binned_orientation_correlation(source;warmup,bin_size,tolerance_degrees)
    a,b = data.analysis,data.bins
    ticks = length(b.ids)<=15 ? b.ids : unique(round.(Int,range(first(b.ids),last(b.ids);length=10)))
    CairoMakie.activate!()
    fig = Figure(resolution=(1000,580))
    ax = Axis(fig[1,1];title="Within-bin Pearson correlation: 90° vs 270°",
        xlabel="Time bin",ylabel="Pearson correlation",xticks=ticks,yticks=-1:.5:1)
    hlines!(ax,[0.];color=(:gray,.5),linestyle=:dash)
    lines!(ax,b.ids,data.correlation;color=:royalblue,linewidth=2)
    scatter!(ax,b.ids[data.valid_bins],data.correlation[data.valid_bins];color=:royalblue,markersize=10)
    xlims!(ax,first(b.ids)-.5,last(b.ids)+.5)
    ylims!(ax,-1.03,1.03)
    if !any(data.valid_bins)
        text!(ax,(first(b.ids)+last(b.ids))/2,0.;text="No defined correlations\nEmpty group or constant mean vector in every bin",
            align=(:center,:center),fontsize=17)
    end
    Label(fig[0,1],"Warmup: $(a.warmup) | $(a.bin_size) iterations/bin | Bin 1 starts at $(a.warmup+1) | Angle tolerance ±$(a.tolerance_degrees)°";tellwidth=false)
    Label(fig[2,1],"Mean-centered vectors | $(count(!,data.valid_bins)) undefined bins omitted (empty group or constant mean vector)";tellwidth=false)
    dir = nothing
    if save_outputs
        dir = _new_orientation_analysis_directory(output_dir)
        _write_orientation_csv(joinpath(dir,"correlation_by_bin.csv"),
            ["bin","iteration_start","iteration_end","n_samples_90deg","n_samples_270deg","pearson_correlation","valid","status"],
            hcat(b.ids,b.starts,b.ends,data.counts_90,data.counts_270,data.correlation,data.valid_bins,string.(data.status)))
        _write_variant_means(dir,a,(90.,270.))
        _write_variant_parameters(dir,a,"Pearson correlation across neurons (mean-centered)","90 versus 270 degrees within each bin")
        CairoMakie.save(joinpath(dir,"Orientation_correlation.png"),fig)
    end
    return (;fig,data...,axis=ax,output_dir=dir)
end

function _centered_bin_profiles(group)
    profiles = zeros(size(group.mean_spikes))
    status = fill(:empty, size(profiles,1))
    for b in axes(profiles,1)
        group.counts[b]==0 && continue
        row = @view group.mean_spikes[b,:]
        all(isfinite,row) || throw(ArgumentError("Nonempty bin means must be finite"))
        status[b] = :constant
        all(==(row[1]),row) && continue
        centered = row .- mean(row)
        profiles[b,:] .= centered ./ norm(centered)
        status[b] = :valid
    end
    return profiles,status
end

"""
    binned_orientation_correlation_matrix(source; warmup=1000, bin_size=1200,
        orientations=(90.,270.), tolerance_degrees=1.)

Pearson correlation across neurons between bin means for two selected angles.
Rows are bins for the second angle; columns are bins for the first. Empty or constant means give NaN.
Accepts p_bins, p_cross, an existing binned analysis, or a run result/directory.
"""
function binned_orientation_correlation_matrix(source;warmup=1000,bin_size=1200,orientations=(90.,270.), tolerance_degrees=1.)
    length(orientations)==2 && all(isfinite,orientations) ||
        throw(ArgumentError("Provide two finite orientations in degrees"))
    angles = Tuple(mod.(Float64.(orientations),360.))
    allunique(angles) || throw(ArgumentError("Orientations must be distinct"))
    a = _resolve_orientation_bin_analysis(source;warmup,bin_size,
        orientations=angles,tolerance_degrees)
    all(angle->haskey(a.groups,angle),angles) ||
        throw(ArgumentError("The binned analysis must include orientations $angles"))
    g90,g270 = a.groups[angles[1]],a.groups[angles[2]]
    u,status_90 = _centered_bin_profiles(g90)
    v,status_270 = _centered_bin_profiles(g270)
    correlation_matrix = v*transpose(u)
    clamp!(correlation_matrix,-1.,1.)
    correlation_matrix[status_270 .!= :valid,:] .= NaN
    correlation_matrix[:,status_90 .!= :valid] .= NaN
    return (;analysis=a,bins=a.bins,orientations=angles,correlation_matrix,
        status_x=status_90,status_y=status_270,counts_x=copy(g90.counts),counts_y=copy(g270.counts),
        status_90,status_270,counts_90=copy(g90.counts),counts_270=copy(g270.counts))
end

"""
    plot_orientation_correlation_heatmap(source; warmup=1000, bin_size=1200,
        orientations=(90.,270.), tolerance_degrees=1., save_outputs=true, output_dir=...)

The first orientation is on x and the second on y. The display scale is 0–1, with negative values shown at the lowest
color. Returned and saved correlations retain their full signed values.
"""
function plot_orientation_correlation_heatmap(source;warmup=1000,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.,save_outputs=true,run_label=nothing,
        output_dir=joinpath(@__DIR__,"Data","Analysis","OrientationCorrelationMatrix"))
    data = binned_orientation_correlation_matrix(source;warmup,bin_size,orientations,tolerance_degrees)
    a,b = data.analysis,data.bins
    x,y = _orientation_bin_label.(data.orientations)
    title = "Cross-correlation: $(x)° vs $(y)°"
    nbins = length(b.ids)
    edges = collect(.5:1.0:(nbins+.5))
    ticks = nbins<=15 ? b.ids : unique(round.(Int,range(first(b.ids),last(b.ids);length=10)))
    CairoMakie.activate!()
    fig = Figure(resolution=(1000,850))
    ax = Axis(fig[1,1];aspect=DataAspect(),title,
        xlabel="$(x)° time bin",ylabel="$(y)° time bin",xticks=ticks,yticks=ticks)
    # Makie's first matrix dimension is x; exported matrices use rows=y.
    hm = heatmap!(ax,edges,edges,_upper_correlation_display(transpose(data.correlation_matrix));
        colormap=:jet,colorrange=(0,1),lowclip=:transparent,nan_color=:lightgray,interpolate=false)
    Colorbar(fig[1,2],hm;label="Pearson correlation",ticks=0:.2:1)
    dir = nothing
    if save_outputs
        dir = _new_orientation_analysis_directory(output_dir)
        _write_orientation_csv(joinpath(dir,"correlation_matrix_$(y)rows_$(x)cols.csv"),
            vcat(["bin_$(y)deg"],["bin_$(x)deg_$i" for i in b.ids]),
            hcat(b.ids,data.correlation_matrix))
        _write_variant_means(dir,a,data.orientations)
        _write_variant_parameters(dir,a,"Pearson correlation across neurons",
            "$y-degree bins in rows versus $x-degree bins in columns; display range 0–1")
        CairoMakie.save(joinpath(dir,"Between_bin_orientation_correlation.png"),fig)
    end
    return (;fig,data...,axis=ax,heatmap=hm,output_dir=dir)
end

"""
    plot_orientation_correlation_triptych(source; orientations=(90.,270.),
        warmup=2400, bin_size=1200, tolerance_degrees=1., fontsize=22)

Plot the two autocorrelation matrices and their cross-correlation matrix in
one figure. The first orientation is on the cross-correlation x-axis. All
panels show the visual upper triangle, including the diagonal, and share a
0–1 color scale. The returned matrices retain all signed correlations.
"""
function plot_orientation_correlation_triptych(source;orientations=(90.,270.),
        warmup=2400,bin_size=1200,tolerance_degrees=1.,fontsize=22)
    data = binned_orientation_correlation_matrix(source;
        orientations,warmup,bin_size,tolerance_degrees)
    x,y = _orientation_bin_label.(data.orientations)
    a,b = data.analysis,data.bins
    auto = [a.groups[angle].correlations for angle in data.orientations]
    nbins = length(b.ids)
    edges = collect(.5:1.0:(nbins+.5))
    ticks = unique(round.(Int,range(first(b.ids),last(b.ids);length=min(nbins,5))))
    CairoMakie.activate!()
    fig = Figure(;resolution=(1700,600),fontsize)
    panels = (
        ("Autocorrelation: $(x)°","Time bin","Time bin",auto[1]),
        ("Autocorrelation: $(y)°","Time bin","Time bin",auto[2]),
        ("Cross-correlation: $(x)° vs $(y)°","$(x)° time bin","$(y)° time bin",
            transpose(data.correlation_matrix))
    )
    for (j,(title,xlabel,ylabel,matrix)) in enumerate(panels)
        ax = Axis(fig[1,j];aspect=DataAspect(),title,xlabel,ylabel,
            xticks=ticks,yticks=ticks,xgridvisible=false,ygridvisible=false,
            titlesize=fontsize+2,xticklabelsize=fontsize-2,yticklabelsize=fontsize-2)
        heatmap!(ax,edges,edges,_upper_correlation_display(matrix);
            colormap=:jet,colorrange=(0,1),lowclip=:transparent,
            nan_color=:lightgray,interpolate=false)
        limits!(ax,first(edges),last(edges),first(edges),last(edges))
    end
    Colorbar(fig[1,4];colormap=:jet,colorrange=(0,1),label="Pearson correlation",
        ticks=0:.2:1,labelsize=fontsize,ticklabelsize=fontsize-2)
    return (;fig,data...,autocorrelations=auto)
end
end # module
using .RotatingShapeSingleDrift

export initialize_environment, render_frame, frame_at_time, rotation_state,
       step_environment!, plot_frame, plot_rotation_sequence, record_rotation,
       initialize_reservoir, model_step!, run_rotating_model, save_run,
       launch_rotating_dashboard, interactive_result,
       window_spike_pca, plot_orientation_pc1, plot_spike_pc12, plot_spike_pc12_time, spike_hamming_distances, plot_spike_hamming_heatmap, binned_orientation_correlations, plot_binned_orientation_correlations, binned_orientation_cosine_similarity, plot_orientation_cosine_similarity, binned_orientation_cosine_matrix, plot_orientation_cosine_heatmap, binned_orientation_cosines, plot_binned_orientation_cosines, binned_orientation_correlation, plot_orientation_correlation, binned_orientation_correlation_matrix, plot_orientation_correlation_heatmap, plot_orientation_correlation_triptych, load_drift_run
end


# Include to load the runner and checkpoint reader without starting simulations.
using Serialization, TOML, SHA, LinearAlgebra


# Preserve the original ensemble settings by default for checkpoint resumption.
# Opt into the eight-frame sequence with ensemble_environment(frames_per_rotation=8).
function ensemble_environment(;frames_per_rotation=nothing)
    isnothing(frames_per_rotation) ||
        (frames_per_rotation isa Integer && !(frames_per_rotation isa Bool) && frames_per_rotation > 0) ||
        throw(ArgumentError("frames_per_rotation must be a positive integer or nothing"))
    dt = 1/30
    angular_speed = isnothing(frames_per_rotation) ? pi/2 : 2pi/(frames_per_rotation*dt)
    return (;N=64,side_length=2.0,object_radius=.65,angular_speed,dt,theta0=0.,
        reverse_rotation=false,rotations_per_direction=5)
end
ensemble_reservoir() = (;nnodes=250,p_link=.1,sensory_fraction=.1,input_p_link=.1,
    input_amp=1.,leak=.25,lrate_wmat=1.,lrate_targ=.01,targ_min=1.,acts_neg=0,learn_on=true)

_ensemble_seed_path(dir,seed) = joinpath(dir,"seed_"*lpad(string(seed),4,'0')*".jls")

function _ensemble_atomic_write(write_file,path)
    tmp,io = mktemp(dirname(path))
    try
        write_file(io)
        close(io)
        mv(tmp,path)
    finally
        isopen(io) && close(io)
        isfile(tmp) && rm(tmp)
    end
end

# Preserve source fingerprints for existing ensemble manifests.
function _ensemble_sources()
    source = read(@__FILE__, String)
    section(name) = split(split(source, "# BEGIN " * name * "\n"; limit=2)[2],
                          "# END " * name * "\n"; limit=2)[1]
    environment = section("environment")
    language = section("shared_language")
    reservoir = section("reservoir")
    embedded = "# BEGIN shared_language\n" * language * "# END shared_language"
    reservoir = replace(reservoir, embedded =>
        "include(joinpath(@__DIR__,\"..\",\"Language\",\"Language_Functions.jl\"))")
    return Dict("RotatingShapeSingle_Functions.jl"=>bytes2hex(sha256(environment)),
        "RotatingShapeSingle_Reservoir.jl"=>bytes2hex(sha256(reservoir)),
        "Language_Functions.jl"=>bytes2hex(sha256(language)))
end

"""
    ensemble_status(dir)

Return completed and pending seeds for this ensemble. Only published checkpoint
files count as completed; a simulation interrupted before saving remains pending.
"""
function ensemble_status(dir)
    config = TOML.parsefile(joinpath(dir,"ensemble.toml"))
    completed = filter(seed->isfile(_ensemble_seed_path(dir,seed)),config["seeds"])
    pending = setdiff(config["seeds"],completed)
    return (;completed,pending,total=length(config["seeds"]))
end

"""
    load_ensemble_run(dir, seed)

Load a locally generated checkpoint as the same result structure used by the
analysis functions (including result.model, result.ts, and result.start_wmat).
Use the same Julia version and model module definitions as the saved ensemble.
"""
function load_ensemble_run(dir,seed::Integer)
    path = abspath(_ensemble_seed_path(dir,seed))
    isfile(path) || throw(ArgumentError("No completed checkpoint for seed $seed in $dir"))
    saved = open(deserialize,path)
    saved.format_version==1 && saved.seed==seed ||
        throw(ArgumentError("Unexpected checkpoint format or seed in $path"))
    return merge(saved.result,(;output_dir=dirname(path),checkpoint_path=path))
end

"""
    run_seed_ensemble(dir; seeds=1:100, iterations=48000,
        environment=ensemble_environment(), reservoir=ensemble_reservoir(),
        input_seed_offset=100000)

Run one fresh model per seed and save each full result atomically. The sensory
seed is seed + input_seed_offset. Existing completed seeds are skipped if the
directory's configuration and simulation source hashes match. Runs use the
Julia threads configured at launch. Returns completion status, not all models
in memory. An unfinished seed restarts from its initial state on resume.
"""
function run_seed_ensemble(dir;seeds=1:100,iterations=48000,
        environment=ensemble_environment(),reservoir=ensemble_reservoir(),
        input_seed_offset=100000)
    ids = collect(seeds)
    !isempty(ids) && all(s->s isa Integer && !(s isa Bool) && s>=0,ids) && allunique(ids) ||
        throw(ArgumentError("Provide distinct nonnegative integer seeds"))
    iterations isa Integer && iterations>0 || throw(ArgumentError("iterations must be positive"))
    input_seed_offset isa Integer && input_seed_offset>0 ||
        throw(ArgumentError("input_seed_offset must be a positive integer"))
    sensory_ids = [Base.checked_add(seed,input_seed_offset) for seed in ids]
    isempty(intersect(ids,sensory_ids)) || throw(ArgumentError("Recurrent and sensory seed ranges overlap"))
    !hasproperty(reservoir,:seed) && !hasproperty(reservoir,:input_seed) ||
        throw(ArgumentError("Specify seeds through seeds and input_seed_offset, not reservoir"))
    config = Dict("format_version"=>1,"julia_version"=>string(VERSION),
        "iterations"=>iterations,"seeds"=>ids,"input_seed_offset"=>input_seed_offset,
        "environment"=>Dict(string(k)=>v for (k,v) in pairs(environment)),
        "reservoir"=>Dict(string(k)=>v for (k,v) in pairs(reservoir)),
        "source_sha256"=>_ensemble_sources())
    env = RotatingShapeSingleModel.initialize_environment(;environment...)
    dir = abspath(dir)
    mkpath(dir)
    lock_path = joinpath(dir,".running")
    try
        mkdir(lock_path)
    catch
        isdir(lock_path) && error("Ensemble is locked at $lock_path. Check its pid before restarting.")
        rethrow()
    end
    try
        write(joinpath(lock_path,"pid"),string(getpid()))
        manifest = joinpath(dir,"ensemble.toml")
        if isfile(manifest)
            TOML.parsefile(manifest)==config ||
                throw(ArgumentError("Ensemble settings or simulation sources differ; choose a new output directory"))
        else
            any(name->startswith(name,"seed_") && endswith(name,".jls"),readdir(dir)) &&
                error("Checkpoint files exist without an ensemble manifest")
            _ensemble_atomic_write(io->TOML.print(io,config;sorted=true),manifest)
        end
        status = ensemble_status(dir)
        pending = status.pending
        nfinished = Threads.Atomic{Int}(length(status.completed))
        println("Ensemble: $(length(ids)) seeds × $iterations iterations; $(length(pending)) pending; $(Threads.nthreads()) Julia threads")
        println("Output: $dir")
        flush(stdout)
        Threads.@threads for seed in pending
            input_seed = Base.checked_add(seed,input_seed_offset)
            started = time()
            println("Starting seed $seed (sensory seed $input_seed)")
            flush(stdout)
            result = RotatingShapeSingleModel.run_rotating_model(env;reservoir...,seed,input_seed,
                iterations,save=false,progress_every=0)
            finite = all(isfinite,result.ts.mean_abs_error) && all(isfinite,result.model.wmat)
            path = _ensemble_seed_path(dir,seed)
            _ensemble_atomic_write(io->serialize(io,(;format_version=1,seed,input_seed,
                finite,result)),path)
            count = Threads.atomic_add!(nfinished,1)+1
            println("Saved seed $seed | $count/$(length(ids)) complete | $(round(time()-started;digits=1)) s",finite ? "" : " | nonfinite values recorded")
            flush(stdout)
        end
        return ensemble_status(dir)
    finally
        rm(lock_path;recursive=true)
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    1<=length(ARGS)<=2 || error("Usage: julia --threads=4 RotatingShapeSingle_Ensemble.jl OUTPUT_DIR [N]")
    n = length(ARGS)==2 ? parse(Int,ARGS[2]) : 100
    n>0 || error("N must be positive")
    BLAS.set_num_threads(1)
    status = run_seed_ensemble(ARGS[1];seeds=1:n)
    println("Complete: $(length(status.completed))/$(status.total) seeds saved")
end


# Include this file to load the analysis without running it.
using Statistics, DelimitedFiles, CairoMakie, LinearAlgebra


function _summarize_lag_correlations(per_seed)
    dims = (size(per_seed,2),size(per_seed,3))
    averages, se = fill(NaN,dims),fill(NaN,dims)
    counts = zeros(Int,dims)
    for k in axes(per_seed,3), j in axes(per_seed,2)
        values = filter(isfinite,@view per_seed[:,j,k])
        n = length(values)
        counts[j,k] = n
        n>0 && (averages[j,k] = mean(values))
        n>1 && (se[j,k] = std(values)/sqrt(n))
    end
    return (;mean_correlation=averages,se,counts)
end

"""
    ensemble_correlation_lag(batch_dir; warmup=2400, bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.)

Compare the final bin with each earlier bin within each orientation, separately
for every saved simulation. Lag 0 is the final bin compared with itself. Bins
start at warmup + 1; the final partial bin, if any, is retained. Aggregate
Pearson r across seeds at each lag and orientation, excluding undefined values.
SE is sample SD divided by sqrt(number of valid seeds); undefined for n < 2.
The within_bin result compares the two selected orientations in the same bin at each lag.
The between_orientation result compares each orientation's final bin with the
other orientation at each lag. All ordered distinct orientation pairs are recorded and averaged within
seed before computing mean and SE across seeds with complete comparisons.
"""
function ensemble_correlation_lag(batch_dir;warmup=2400,bin_size=1200,orientations=(90.,270.),tolerance_degrees=1.)
    status = ensemble_status(batch_dir)
    isempty(status.pending) ||
        throw(ArgumentError("Ensemble is incomplete: $(length(status.pending)) seeds pending"))
    seeds = status.completed
    isempty(seeds) && throw(ArgumentError("The ensemble has no completed seeds"))
    length(orientations)>=2 && all(isfinite,orientations) ||
        throw(ArgumentError("Provide at least two finite orientations in degrees"))
    orientations = Tuple(mod.(Float64.(orientations),360.))
    allunique(orientations) || throw(ArgumentError("Orientations must be distinct"))
    direction_pairs = [(u,v) for u in eachindex(orientations) for v in eachindex(orientations) if u!=v]
    bins = nothing
    per_seed = Array{Float64,3}(undef,0,0,0)
    cross_per_seed = Matrix{Float64}(undef,0,0)
    directional = Array{Float64,3}(undef,0,0,0)
    for (i,seed) in enumerate(seeds)
        result = load_ensemble_run(batch_dir,seed)
        a = RotatingShapeSingleModel.binned_orientation_correlations(result;
            warmup,bin_size,orientations,tolerance_degrees)
        if i==1
            bins = a.bins
            per_seed = fill(NaN,length(seeds),length(bins.ids),length(orientations))
            cross_per_seed = fill(NaN,length(seeds),length(bins.ids))
            directional = fill(NaN,length(seeds),length(bins.ids),length(direction_pairs))
        elseif a.bins != bins
            throw(ArgumentError("Seed $seed has different bin boundaries"))
        end
        for (k,angle) in enumerate(orientations)
            matrix = a.groups[angle].correlations
            per_seed[i,:,k] .= @view matrix[end,end:-1:1]
        end
        drift = RotatingShapeSingleModel.RotatingShapeSingleDrift
        profiles = [drift._centered_bin_profiles(a.groups[angle]) for angle in orientations]
        cross_per_seed[i,:] .= 0
        for (k,(u,v)) in enumerate(direction_pairs)
            x,sx = profiles[u]
            y,sy = profiles[v]
            for (q,b) in enumerate(reverse(eachindex(bins.ids)))
                if sx[end]==:valid && sy[b]==:valid
                    directional[i,q,k] = clamp(dot(@view(x[end,:]),@view(y[b,:])),-1.,1.)
                end
                if sx[b]==:valid && sy[b]==:valid
                    cross_per_seed[i,q] += clamp(dot(@view(x[b,:]),@view(y[b,:])),-1.,1.)
                else
                    cross_per_seed[i,q] = NaN
                end
            end
        end
        cross_per_seed[i,:] ./= length(direction_pairs)
    end
    summary = _summarize_lag_correlations(per_seed)
    auto_per_seed = dropdims(mean(per_seed;dims=3);dims=3)
    auto_summary = _summarize_lag_correlations(reshape(auto_per_seed,size(auto_per_seed)...,1))
    autocorrelation = (;per_seed=auto_per_seed,mean_correlation=vec(auto_summary.mean_correlation),
        se=vec(auto_summary.se),counts=vec(auto_summary.counts))
    cross_summary = _summarize_lag_correlations(reshape(cross_per_seed,size(cross_per_seed)...,1))
    within_bin = (;per_seed=cross_per_seed,mean_correlation=vec(cross_summary.mean_correlation),
        se=vec(cross_summary.se),counts=vec(cross_summary.counts))
    paired = dropdims(mean(directional;dims=3);dims=3)
    paired_summary = _summarize_lag_correlations(reshape(paired,size(paired)...,1))
    between_orientation = (;directional,direction_pairs,per_seed=paired,
        mean_correlation=vec(paired_summary.mean_correlation),
        se=vec(paired_summary.se),counts=vec(paired_summary.counts),
        directional_counts=vec(sum(isfinite.(directional);dims=(1,3))))
    return (;batch_dir=abspath(batch_dir),seeds,orientations,bins,
        lags=collect(0:length(bins.ids)-1),per_seed,summary...,autocorrelation,within_bin,between_orientation,
        warmup,bin_size,tolerance_degrees)
end

function _save_ensemble_lag(a,fig,output_dir)
    dir = RotatingShapeSingleModel.RotatingShapeSingleReservoir.new_run_directory(output_dir)
    open(joinpath(dir,"lag_summary.csv"),"w") do io
        println(io,"orientation_degrees,lag,comparison_bin_start,comparison_bin_end,mean_pearson_r,se,n_seeds")
        for (k,angle) in enumerate(a.orientations), (j,lag) in enumerate(a.lags)
            b = length(a.bins.ids)-lag
            println(io,join((angle,lag,a.bins.starts[b],a.bins.ends[b],
                a.mean_correlation[j,k],a.se[j,k],a.counts[j,k]),','))
        end
    end
    open(joinpath(dir,"lag_by_seed.csv"),"w") do io
        println(io,"seed,orientation_degrees,lag,pearson_r")
        for (i,seed) in enumerate(a.seeds), (k,angle) in enumerate(a.orientations),
                (j,lag) in enumerate(a.lags)
            println(io,join((seed,angle,lag,a.per_seed[i,j,k]),','))
        end
    end
    cross = a.within_bin
    auto = a.autocorrelation
    open(joinpath(dir,"autocorrelation_summary.csv"),"w") do io
        println(io,"lag,comparison_bin_start,comparison_bin_end,mean_pearson_r,paired_se,n_seed_pairs")
        for (j,lag) in enumerate(a.lags)
            b = length(a.bins.ids)-lag
            println(io,join((lag,a.bins.starts[b],a.bins.ends[b],
                auto.mean_correlation[j],auto.se[j],auto.counts[j]),','))
        end
    end
    open(joinpath(dir,"within_bin_summary.csv"),"w") do io
        println(io,"lag,bin_start,bin_end,mean_pearson_r,se,n_seeds")
        for (j,lag) in enumerate(a.lags)
            b = length(a.bins.ids)-lag
            println(io,join((lag,a.bins.starts[b],a.bins.ends[b],
                cross.mean_correlation[j],cross.se[j],cross.counts[j]),','))
        end
    end
    open(joinpath(dir,"within_bin_by_seed.csv"),"w") do io
        println(io,"seed,lag,pearson_r")
        for (i,seed) in enumerate(a.seeds), (j,lag) in enumerate(a.lags)
            println(io,join((seed,lag,cross.per_seed[i,j]),','))
        end
    end
    between = a.between_orientation
    open(joinpath(dir,"between_orientation_summary.csv"),"w") do io
        println(io,"lag,comparison_bin_start,comparison_bin_end,mean_pearson_r,paired_se,n_seed_pairs,n_directional_values")
        for (j,lag) in enumerate(a.lags)
            b = length(a.bins.ids)-lag
            println(io,join((lag,a.bins.starts[b],a.bins.ends[b],between.mean_correlation[j],
                between.se[j],between.counts[j],between.directional_counts[j]),','))
        end
    end
    open(joinpath(dir,"between_orientation_by_direction.csv"),"w") do io
        println(io,"seed,lag,focal_orientation_degrees,comparison_orientation_degrees,pearson_r")
        for (i,seed) in enumerate(a.seeds), (j,lag) in enumerate(a.lags), (k,(u,v)) in enumerate(between.direction_pairs)
            println(io,join((seed,lag,a.orientations[u],a.orientations[v],
                between.directional[i,j,k]),','))
        end
    end
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:batch_dir,:warmup,:bin_size,:orientations,:tolerance_degrees)
            println(io,key," = ",getproperty(a,key))
        end
        println(io,"reference_bin = ",last(a.bins.ids))
        println(io,"reference_iterations = ",last(a.bins.starts),":",last(a.bins.ends))
        println(io,"autocorrelation = average all same-orientation final-bin comparisons within each seed")
        println(io,"within_bin = Pearson correlation between the selected orientations in the bin at each lag")
        println(io,"between_orientation = final bin of each orientation versus every distinct orientation at each lag; average complete sets of directions within seed before computing SE")
        println(io,"summary = arithmetic mean and sample SD / sqrt(n) across valid seeds")
    end
    CairoMakie.save(joinpath(dir,"Ensemble_correlation_lag.png"),fig)
    return dir
end

"""
    plot_ensemble_correlation_lag(batch_dir; warmup=2400, bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1., save_outputs=true, output_dir=...)

Plot the ensemble mean Pearson r and ±1 SE for the averaged same-orientation final-bin comparison
and the within-bin and across-lag cross-orientation comparisons. The x-axis counts bins backward
from the final bin. Returns (;fig, analysis, output_dir).
"""
function plot_ensemble_correlation_lag(batch_dir;warmup=2400,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(batch_dir,"Analysis","CorrelationLag"))
    a = ensemble_correlation_lag(batch_dir;warmup,bin_size,orientations,tolerance_degrees)
    CairoMakie.activate!()
    fig = Figure(resolution=(1000,570))
    ax = Axis(fig[1,1];title="Spike-pattern correlations by bin age ($(length(a.orientations)) poses: $(join(round.(Int,a.orientations), ", "))°)",
        xlabel="Lag (bins)",ylabel="Pearson correlation")
    auto = a.autocorrelation
    band!(ax,a.lags,auto.mean_correlation.-auto.se,auto.mean_correlation.+auto.se;
        color=(:royalblue,.22))
    lines!(ax,a.lags,auto.mean_correlation;color=:royalblue,linewidth=2.5,
        label="Lagged Autocorrelation")
    scatter!(ax,a.lags,auto.mean_correlation;color=:royalblue,markersize=5)
    cross = a.within_bin
    band!(ax,a.lags,cross.mean_correlation.-cross.se,cross.mean_correlation.+cross.se;
        color=(:seagreen,.22))
    lines!(ax,a.lags,cross.mean_correlation;color=:seagreen,linewidth=2.5,
        linestyle=:dash,label="Within-bin cross-correlation")
    scatter!(ax,a.lags,cross.mean_correlation;color=:seagreen,markersize=5)
    between = a.between_orientation
    band!(ax,a.lags,between.mean_correlation.-between.se,between.mean_correlation.+between.se;
        color=(:purple,.18))
    lines!(ax,a.lags,between.mean_correlation;color=:purple,linewidth=2.5,
        linestyle=:dash,label="Lagged cross correlation")
    scatter!(ax,a.lags,between.mean_correlation;color=:purple,markersize=5)
    length(a.lags)>1 && xlims!(ax,0,last(a.lags))
    axislegend(ax;position=:lb)
    Label(fig[2,1],"$(length(a.seeds)) seeds | $(a.bin_size) iterations/bin | Final bin: $(last(a.bins.starts))–$(last(a.bins.ends)) | Ribbon: ±1 SE";
        tellwidth=false)
    dir = save_outputs ? _save_ensemble_lag(a,fig,output_dir) : nothing
    return (;fig,analysis=a,output_dir=dir)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==1 || error("Usage: julia RotatingShapeSingle_EnsembleLag.jl BATCH_DIR")
    p_lag = plot_ensemble_correlation_lag(ARGS[1])
    println("Saved lag analysis to ",p_lag.output_dir)
    println("Lag range: ",first(p_lag.analysis.lags),":",last(p_lag.analysis.lags))
    println("Valid seed counts (final-bin comparisons): ",extrema(p_lag.analysis.counts))
    println("Valid seed counts (within-bin comparison): ",extrema(p_lag.analysis.within_bin.counts))
    println("Valid seed pairs (between-orientation): ",extrema(p_lag.analysis.between_orientation.counts))
    println("Valid directional values: ",extrema(p_lag.analysis.between_orientation.directional_counts))
end


# Load without running the analysis when included in a Julia session.
using LinearAlgebra, Statistics, CairoMakie

# Rows are focal seeds and columns are comparison seeds. Drop a seed from
# both roles when computing each jackknife replicate.
function _seed_pair_summary(pair_matrix)
    n = size(pair_matrix,1)
    row_sums, col_sums = zeros(n), zeros(n)
    row_counts, col_counts = zeros(Int,n), zeros(Int,n)
    for j in 1:n, i in 1:n
        i==j && continue
        r = pair_matrix[i,j]
        isfinite(r) || continue
        row_sums[i] += r
        col_sums[j] += r
        row_counts[i] += 1
        col_counts[j] += 1
    end
    total,count = sum(row_sums),sum(row_counts)
    estimate = count==0 ? NaN : total/count
    leave_one_out = fill(NaN,n)
    for i in 1:n
        remaining = count-row_counts[i]-col_counts[i]
        remaining>0 && (leave_one_out[i]=(total-row_sums[i]-col_sums[i])/remaining)
    end
    se = n>2 && all(isfinite,leave_one_out) ?
        sqrt((n-1)/n * sum(abs2,leave_one_out.-mean(leave_one_out))) : NaN
    return (;mean_correlation=estimate,se,count,focal_counts=row_counts,
        focal_means=[row_counts[i]==0 ? NaN : row_sums[i]/row_counts[i] for i in 1:n])
end

function _across_seed_profiles(a)
    n,nb = length(a.seeds),length(a.bins.ids)
    profiles = Array{Float64,4}(undef,0,0,0,0)
    drift = RotatingShapeSingleModel.RotatingShapeSingleDrift
    for (i,seed) in enumerate(a.seeds)
        result = load_ensemble_run(a.batch_dir,seed)
        binned = drift.binned_orientation_correlations(result;warmup=a.warmup,
            bin_size=a.bin_size,tolerance_degrees=a.tolerance_degrees,orientations=a.orientations)
        binned.bins==a.bins || throw(ArgumentError("Different bin boundaries for seed $seed"))
        if i==1
            profiles = fill(NaN,binned.nneurons,n,nb,length(a.orientations))
        end
        for (k,angle) in enumerate(a.orientations)
            normalized,status = drift._centered_bin_profiles(binned.groups[angle])
            for b in 1:nb
                status[b]==:valid && (profiles[:,i,b,k] .= @view normalized[b,:])
            end
        end
    end
    return profiles
end

"""
    ensemble_correlation_robustness(batch_dir; warmup=2400, bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.)

Compare the original within-seed lag curves to baselines across different seeds.
For each ordered pair of different seeds, average all applicable orientation comparisons
when all are defined, then average the valid pairs. Pearson correlations use
matching neuron indices. SE is a delete-one-seed jackknife, removing the seed
from both the focal and comparison roles.
"""
function ensemble_correlation_robustness(batch_dir;warmup=2400,bin_size=1200,orientations=(90.,270.),tolerance_degrees=1.)
    within = ensemble_correlation_lag(batch_dir;warmup,bin_size,orientations,tolerance_degrees)
    length(within.seeds)>=2 || throw(ArgumentError("At least two simulations are required"))
    z = _across_seed_profiles(within)
    n,nb = length(within.seeds),length(within.lags)
    means,errors = fill(NaN,nb,3),fill(NaN,nb,3)
    counts = zeros(Int,nb,3)
    focal_means,focal_counts = fill(NaN,n,nb,3),zeros(Int,n,nb,3)
    np = length(within.orientations)
    recent = @view z[:,:,nb,:]
    # Sum all pose profiles once: all ordered pairs minus the matched poses
    # gives the distinct-pose comparisons without 56 matrix products per lag.
    recent_sum = dropdims(sum(recent;dims=3);dims=3)
    for (j,lag) in enumerate(within.lags)
        old = @view z[:,:,nb-lag,:]
        old_sum = dropdims(sum(old;dims=3);dims=3)
        same = sum(transpose(@view(recent[:,:,k])) * @view(old[:,:,k]) for k in 1:np)
        old_same = sum(transpose(@view(old[:,:,k])) * @view(old[:,:,k]) for k in 1:np)
        comparisons = (same ./ np,
            (transpose(recent_sum)*old_sum .- same) ./ (np*(np-1)),
            (transpose(old_sum)*old_sum .- old_same) ./ (np*(np-1)))
        for (k,matrix) in enumerate(comparisons)
            s = _seed_pair_summary(clamp!(matrix,-1.,1.))
            means[j,k],errors[j,k],counts[j,k] = s.mean_correlation,s.se,s.count
            focal_means[:,j,k] .= s.focal_means
            focal_counts[:,j,k] .= s.focal_counts
        end
    end
    return (;within,names=("Lagged Autocorrelation","Lagged cross correlation",
        "Within-bin cross-correlation"),mean_correlation=means,se=errors,
        ordered_pair_counts=counts,focal_means,focal_counts)
end

function _save_seed_robustness(a,fig,output_dir)
    dir = RotatingShapeSingleModel.RotatingShapeSingleReservoir.new_run_directory(output_dir)
    within = a.within
    original = (within.autocorrelation,within.between_orientation,within.within_bin)
    open(joinpath(dir,"across_seed_summary.csv"),"w") do io
        println(io,"comparison,lag,bin_start,bin_end,within_seed_mean,within_seed_se,n_within_seeds,across_seed_mean,across_seed_jackknife_se,n_ordered_seed_pairs,n_directional_correlations")
        for (k,name) in enumerate(a.names), (j,lag) in enumerate(within.lags)
            b = length(within.bins.ids)-lag
            count = a.ordered_pair_counts[j,k]
            println(io,join((name,lag,within.bins.starts[b],within.bins.ends[b],
                original[k].mean_correlation[j],original[k].se[j],original[k].counts[j],
                a.mean_correlation[j,k],a.se[j,k],count,count*(k==1 ? length(within.orientations) : length(within.orientations)*(length(within.orientations)-1))),','))
        end
    end
    open(joinpath(dir,"across_seed_by_focal.csv"),"w") do io
        println(io,"comparison,lag,focal_seed,mean_pearson_r,n_comparison_seeds")
        for (k,name) in enumerate(a.names), (j,lag) in enumerate(within.lags), (i,seed) in enumerate(within.seeds)
            println(io,join((name,lag,seed,a.focal_means[i,j,k],a.focal_counts[i,j,k]),','))
        end
    end
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:batch_dir,:warmup,:bin_size,:orientations,:tolerance_degrees)
            println(io,key," = ",getproperty(within,key))
        end
        println(io,"across_seed = ordered pairs of distinct seeds; average complete sets of orientation comparisons")
        println(io,"across_seed_se = delete-one-seed jackknife, removing both focal and comparison roles")
        println(io,"neuron_matching = same neuron indices across independent initializations")
    end
    CairoMakie.save(joinpath(dir,"Ensemble_across_seed_robustness.png"),fig)
    return dir
end

function _seed_control_figure(a;comparison_label="Between simulations")
    within = a.within
    original = (within.autocorrelation,within.between_orientation,within.within_bin)
    CairoMakie.activate!()
    fig = Figure(resolution=(1100,650))
    ax = Axis(fig[1,1];title="Spike-pattern correlations within and across simulations ($(length(within.orientations)) poses: $(join(round.(Int,within.orientations), ", "))°)",
        xlabel="Lag (bins)",ylabel="Pearson correlation")
    colors = (:royalblue,:purple,:seagreen)
    hlines!(ax,[0.];color=(:gray,.45),linestyle=:dot)
    for (k,color) in enumerate(colors)
        for (values,errors,style) in (
                (original[k].mean_correlation,original[k].se,:solid),
                (a.mean_correlation[:,k],a.se[:,k],:dash))
            band!(ax,within.lags,values.-errors,values.+errors;color=(color,.17))
            lines!(ax,within.lags,values;color,linestyle=style,linewidth=2.5)
        end
    end
    length(within.lags)>1 && xlims!(ax,0,last(within.lags))
    keys = GridLayout()
    fig[2,1] = keys
    Legend(keys[1,1],[LineElement(color=c,linewidth=2.5) for c in colors],
        collect(a.names);orientation=:horizontal,framevisible=false,labelsize=21)
    Legend(keys[2,1],[LineElement(color=:black,linewidth=2.5,linestyle=style)
        for style in (:solid,:dash)],["Within simulation",comparison_label];
        orientation=:horizontal,framevisible=false,labelsize=21)
    return fig
end

function plot_ensemble_correlation_robustness(batch_dir;warmup=2400,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(batch_dir,"Analysis","AcrossSeedRobustness"))
    a = ensemble_correlation_robustness(batch_dir;warmup,bin_size,orientations,tolerance_degrees)
    fig = _seed_control_figure(a)
    dir = save_outputs ? _save_seed_robustness(a,fig,output_dir) : nothing
    return (;fig,analysis=a,output_dir=dir)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==1 || error("Usage: julia RotatingShapeSingle_EnsembleRobustness.jl BATCH_DIR")
    p = plot_ensemble_correlation_robustness(ARGS[1])
    println("Saved robustness analysis to ",p.output_dir)
    println("Ordered pair counts: ",extrema(p.analysis.ordered_pair_counts))
end


using Random, LinearAlgebra, CairoMakie

# Average both focal runs over the complete set of orientation comparisons.
function _random_pair_correlation(z,i,j,b,comparison)
    recent = size(z,3)
    np = size(z,4)
    total = 0.
    count = 0
    for (focal,other) in ((i,j),(j,i)), u in 1:np, v in 1:np
        (comparison==1 ? u==v : u!=v) || continue
        ref_bin = comparison==3 ? b : recent
        r = dot(@view(z[:,focal,ref_bin,u]),@view(z[:,other,b,v]))
        isfinite(r) || return NaN
        total += clamp(r,-1.,1.)
        count += 1
    end
    return total/count
end

"""
    ensemble_random_pair_control(batch_dir; pairing_seed=1, warmup=2400,
        bin_size=1200, orientations=(90.,270.),tolerance_degrees=1.)

Randomly partition the ensemble into disjoint pairs. Use the same pairing
for all lags and measures. Average both focal-run and orientation directions
within each pair. Estimate the mean and SE across complete pair values.
"""
function ensemble_random_pair_control(batch_dir;pairing_seed=1,warmup=2400,
        bin_size=1200,orientations=(90.,270.),tolerance_degrees=1.)
    status = ensemble_status(batch_dir)
    n = status.total
    n>=2 && iseven(n) || throw(ArgumentError("Random pairing requires a positive even number of runs"))
    within = ensemble_correlation_lag(batch_dir;warmup,bin_size,orientations,tolerance_degrees)
    perm = randperm(MersenneTwister(pairing_seed),n)
    pairs = [(perm[i],perm[i+1]) for i in 1:2:n]
    z = _across_seed_profiles(within)
    nb = length(within.lags)
    pair_values = fill(NaN,length(pairs),nb,3)
    for (p,(i,j)) in enumerate(pairs), (q,lag) in enumerate(within.lags), k in 1:3
        pair_values[p,q,k] = _random_pair_correlation(z,i,j,nb-lag,k)
    end
    summary = _summarize_lag_correlations(pair_values)
    seed_pairs = [(within.seeds[i],within.seeds[j]) for (i,j) in pairs]
    return (;within,names=("Lagged Autocorrelation","Lagged cross correlation",
        "Within-bin cross-correlation"),seed_pairs,pairing_seed,pair_values,summary...)
end

function _save_random_pair_control(a,fig,output_dir)
    dir = RotatingShapeSingleModel.RotatingShapeSingleReservoir.new_run_directory(output_dir)
    within = a.within
    original = (within.autocorrelation,within.between_orientation,within.within_bin)
    open(joinpath(dir,"seed_pairs.csv"),"w") do io
        println(io,"pair,seed_a,seed_b")
        for (p,(i,j)) in enumerate(a.seed_pairs)
            println(io,join((p,i,j),','))
        end
    end
    open(joinpath(dir,"random_pair_summary.csv"),"w") do io
        println(io,"comparison,lag,bin_start,bin_end,within_seed_mean,within_seed_se,n_within_seeds,random_pair_mean,random_pair_se,n_pairs")
        for (k,name) in enumerate(a.names), (j,lag) in enumerate(within.lags)
            b = length(within.bins.ids)-lag
            println(io,join((name,lag,within.bins.starts[b],within.bins.ends[b],
                original[k].mean_correlation[j],original[k].se[j],original[k].counts[j],
                a.mean_correlation[j,k],a.se[j,k],a.counts[j,k]),','))
        end
    end
    open(joinpath(dir,"random_pair_values.csv"),"w") do io
        println(io,"comparison,lag,pair,seed_a,seed_b,pearson_r")
        for (k,name) in enumerate(a.names), (j,lag) in enumerate(within.lags),
                (p,(i,h)) in enumerate(a.seed_pairs)
            println(io,join((name,lag,p,i,h,a.pair_values[p,j,k]),','))
        end
    end
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:batch_dir,:warmup,:bin_size,:orientations,:tolerance_degrees)
            println(io,key," = ",getproperty(within,key))
        end
        println(io,"pairing_seed = ",a.pairing_seed)
        println(io,"pairing = one random permutation, grouped into consecutive pairs")
        println(io,"pair_value = mean of both focal-run and orientation directions")
        println(io,"se = sample SD / sqrt(number of valid pairs)")
        println(io,"neuron_matching = same neuron indices across independent initializations")
    end
    CairoMakie.save(joinpath(dir,"Ensemble_random_pair_control.png"),fig)
    return dir
end

function plot_ensemble_random_pair_control(batch_dir;pairing_seed=1,warmup=2400,bin_size=1200,
        orientations=(90.,270.),tolerance_degrees=1.,save_outputs=true,
        output_dir=joinpath(batch_dir,"Analysis","RandomPairControl"))
    a = ensemble_random_pair_control(batch_dir;pairing_seed,warmup,bin_size,orientations,tolerance_degrees)
    fig = _seed_control_figure(a;comparison_label="Between simulations (random pairs)")
    dir = save_outputs ? _save_random_pair_control(a,fig,output_dir) : nothing
    return (;fig,analysis=a,output_dir=dir)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    1<=length(ARGS)<=2 || error("Usage: julia RotatingShapeSingle_EnsembleRandomPairs.jl BATCH_DIR [PAIRING_SEED]")
    pairing_seed = length(ARGS)==2 ? parse(Int,ARGS[2]) : 1
    p = plot_ensemble_random_pair_control(ARGS[1];pairing_seed)
    println("Saved random-pair control to ",p.output_dir)
    println("Valid pair counts: ",extrema(p.analysis.counts))
end


# Adaptation of analyses.Rmd. Including this file does not launch simulations.
using CairoMakie, Statistics, LinearAlgebra, Random

function _rmd_mean(x)
    s = 0.
    n = 0
    for v in x
        if isfinite(v)
            s += v
            n += 1
        end
    end
    return n==0 ? NaN : s/n
end

_rmd_reduce(x,d) = dropdims(mapslices(_rmd_mean,x;dims=d);dims=d)

function _rmd_normalize(x;center)
    z = fill(NaN,size(x))
    for j in axes(x,2)
        v = @view x[:,j]
        all(isfinite,v) || continue
        center && all(==(v[1]),v) && continue
        u = center ? v .- mean(v) : v
        d = norm(u)
        d>0 && (z[:,j] .= u ./ d)
    end
    return z
end

# Columns are windows; rows are relationships. Use pairwise complete edges.
function _rmd_window_cor(x)
    if all(isfinite,x)
        z = _rmd_normalize(x;center=true)
        return clamp!(transpose(z)*z,-1.,1.)
    end
    n = size(x,2)
    out = fill(NaN,n,n)
    for j in 1:n, i in 1:j
        u,v = @view(x[:,i]),@view(x[:,j])
        keep = isfinite.(u) .& isfinite.(v)
        if count(keep)>=2
            out[i,j] = out[j,i] = cor(u[keep],v[keep])
        end
    end
    return out
end

_rmd_lags(c) = [_rmd_mean(diag(c,k)) for k in 1:size(c,1)-1]

function _rmd_embeddings(batch_dir;warmup,bin_size,orientations,tolerance_degrees)
    status = ensemble_status(batch_dir)
    isempty(status.pending) || throw(ArgumentError("Complete the ensemble before running this analysis"))
    seeds = status.completed
    length(seeds)>=3 || throw(ArgumentError("At least three runs are required for the cross-run control"))
    bins,angles = nothing,nothing
    counts = Array{Int,3}(undef,0,0,0)
    means = Array{Float64,4}(undef,0,0,0,0)
    for (i,seed) in enumerate(seeds)
        result = load_ensemble_run(batch_dir,seed)
        a = RotatingShapeSingleModel.binned_orientation_correlations(result;
            warmup,bin_size,orientations,tolerance_degrees)
        if i==1
            bins,angles = a.bins,a.orientations
            length(bins.ids)>=2 || throw(ArgumentError("At least two windows are required"))
            counts = zeros(Int,length(bins.ids),length(angles),length(seeds))
            means = fill(NaN,a.nneurons,size(counts)...)
        elseif a.bins!=bins
            throw(ArgumentError("Seed $seed has different window boundaries"))
        end
        for (k,angle) in enumerate(angles)
            counts[:,k,i] .= a.groups[angle].counts
            means[:,:,k,i] .= transpose(a.groups[angle].mean_spikes)
        end
    end
    return (;seeds,bins,angles,counts,means)
end

function _rmd_geometry_nulls(geometry,edges,cosines,n_perm,rng)
    ne,nw,ns = size(geometry)
    np = size(cosines,1)
    matrices = fill(NaN,nw,nw,n_perm,2)
    # Hold only one permutation's seed matrices in memory.
    shuffled_seeds = Array{Float64,3}(undef,nw,nw,ns)
    shuffled_labels = similar(shuffled_seeds)
    x,y = zeros(ne,nw),zeros(ne,nw)
    for p in 1:n_perm
        order = [randperm(rng,ns) for w in 1:nw]
        for i in 1:ns
            for w in 1:nw
                x[:,w] .= @view geometry[:,w,order[w][i]]
                labels = randperm(rng,np)
                for (k,(u,v)) in enumerate(edges)
                    y[k,w] = cosines[labels[u],labels[v],w,i]
                end
            end
            shuffled_seeds[:,:,i] .= _rmd_window_cor(x)
            shuffled_labels[:,:,i] .= _rmd_window_cor(y)
        end
        matrices[:,:,p,1] .= _rmd_reduce(shuffled_seeds,3)
        matrices[:,:,p,2] .= _rmd_reduce(shuffled_labels,3)
    end
    return matrices
end

"""
    colleague_drift_analyses(batch_dir; warmup=3000, bin_size=1800,
        orientations=0:30:330, tolerance_degrees=1., n_perm=100, permutation_seed=123)

Adapt the language notebook to angle-conditioned embeddings from saved simulations.
The default 48,000-iteration run yields 25 windows. Lag means use every window
pair at the requested positive lag. Cross-orientation comparisons follow the
order supplied in orientations, with the first angle at the earlier window.
"""
function colleague_drift_analyses(batch_dir;warmup=3000,bin_size=1800,
        orientations=0:30:330,tolerance_degrees=1.,n_perm=100,permutation_seed=123)
    n_perm isa Integer && n_perm>=2 || throw(ArgumentError("n_perm must be at least 2"))
    length(orientations)>=3 || throw(ArgumentError("Geometry requires at least three orientations"))
    e = _rmd_embeddings(batch_dir;warmup,bin_size,orientations,tolerance_degrees)
    nn,nw,np,ns = size(e.means)
    edges = [(i,j) for i in 1:np for j in i+1:np]
    ne = length(edges)
    same = fill(NaN,nw,nw,np,ns)
    cross = fill(NaN,nw,nw,ne,ns)
    cosines = fill(NaN,np,np,nw,ns)
    geometry = fill(NaN,ne,nw,ns)
    raw = similar(e.means)
    for i in 1:ns
        z = [_rmd_normalize(@view(e.means[:,:,k,i]);center=true) for k in 1:np]
        for k in 1:np
            same[:,:,k,i] .= clamp!(transpose(z[k])*z[k],-1.,1.)
        end
        for (k,(u,v)) in enumerate(edges)
            cross[:,:,k,i] .= clamp!(transpose(z[u])*z[v],-1.,1.)
        end
        for w in 1:nw
            zraw = _rmd_normalize(@view(e.means[:,w,:,i]);center=false)
            raw[:,w,:,i] .= zraw
            g = transpose(zraw)*zraw
            cosines[:,:,w,i] .= g
            for (k,(u,v)) in enumerate(edges)
                geometry[k,w,i] = g[u,v]
            end
        end
    end
    stable = fill(NaN,nw,nw,ns)
    cross_run = similar(stable)
    cross_run_cosines = similar(cosines)
    for i in 1:ns
        stable[:,:,i] .= _rmd_window_cor(@view geometry[:,:,i])
        j = mod1(i+1,ns)
        # Faithful to the R notebook: cross-run cosine matrices use all entries.
        profiles = Matrix{Float64}(undef,np^2,nw)
        for w in 1:nw
            g = transpose(@view(raw[:,w,:,i])) * @view(raw[:,w,:,j])
            cross_run_cosines[:,:,w,i] .= g
            profiles[:,w] .= vec(g)
        end
        cross_run[:,:,i] .= _rmd_window_cor(profiles)
    end
    nulls = _rmd_geometry_nulls(geometry,edges,cosines,n_perm,MersenneTwister(permutation_seed))
    return (;seeds=e.seeds,bins=e.bins,angles=e.angles,counts=e.counts,edges,
        same,cross,cosines,stable,cross_run,cross_run_cosines,nulls,
        warmup,bin_size,tolerance_degrees,n_perm,permutation_seed)
end

function _rmd_seed_lag_summary(matrices;cyclic=false)
    # Each column is a seed (or one cyclic pair of neighboring seeds).
    values = hcat([_rmd_lags(@view matrices[:,:,s]) for s in axes(matrices,3)]...)
    n = size(values,2)
    m,se,window_se = fill(NaN,size(values,1)),fill(NaN,size(values,1)),fill(NaN,size(values,1))
    counts = zeros(Int,length(m))
    avg_matrix = _rmd_reduce(matrices,3)
    for l in eachindex(m)
        row = @view values[l,:]
        valid = filter(isfinite,row)
        counts[l] = length(valid)
        m[l] = _rmd_mean(valid)
        if cyclic
            loo = [_rmd_mean([row[j] for j in 1:n if j!=i && mod1(j+1,n)!=i]) for i in 1:n]
            all(isfinite,loo) && (se[l]=sqrt((n-1)/n*sum(abs2,loo.-mean(loo))))
        elseif length(valid)>1
            se[l] = std(valid)/sqrt(length(valid))
        end
        d = filter(isfinite,diag(avg_matrix,l))
        length(d)>1 && (window_se[l]=std(d)/sqrt(length(d)))
    end
    return (;m,se,window_se,counts,values)
end

function _rmd_null_summary(mats)
    draws = hcat([_rmd_lags(@view mats[:,:,p]) for p in axes(mats,3)]...)
    m = [_rmd_mean(row) for row in eachrow(draws)]
    limits = map(eachrow(draws)) do row
        v = filter(isfinite,row)
        isempty(v) ? (NaN,NaN) : Tuple(quantile(v,[.025,.975]))
    end
    return (;m,lower=first.(limits),upper=last.(limits),draws)
end

function _rmd_slope(y)
    ids = findall(isfinite,y)
    length(ids)>=3 || return (NaN,NaN,length(ids))
    x,v = Float64.(ids),y[ids]
    dx,dv = x.-mean(x),v.-mean(v)
    sxx = sum(abs2,dx)
    b = dot(dx,dv)/sxx
    return (b,sqrt(sum(abs2,dv.-b.*dx)/(length(ids)-2)/sxx),length(ids))
end

function _rmd_angle_ticks(angles)
    ids = collect(1:cld(length(angles),4):length(angles))
    labels = [string(isinteger(angles[i]) ? Int(angles[i]) : angles[i],"°") for i in ids]
    return ids,labels
end

function _rmd_heatmaps(mats,titles;cols=4,axislabel="Window",ticks=nothing,scale_label="Pearson r")
    n = length(titles)
    rows = cld(n,cols)
    fig = Figure(resolution=(275cols+105,245rows))
    for i in 1:n
        row,col = fldmod1(i,cols)
        ax = Axis(fig[row,col];title=titles[i],xlabel=axislabel,ylabel=axislabel,aspect=DataAspect())
        ticks===nothing || (ax.xticks[]=ticks; ax.yticks[]=ticks)
        heatmap!(ax,mats[i];colormap=:jet,colorrange=(0,1),nan_color=:lightgray,interpolate=false)
    end
    Colorbar(fig[1:rows,cols+1];colormap=:jet,colorrange=(0,1),label=scale_label)
    return fig
end

function _rmd_curve_plot(lags,series;title,ylabel="Pearson correlation",nulls=false,se_only=false)
    fig = Figure(resolution=(1080,560))
    ax = Axis(fig[1,1];title,xlabel="Lag (windows)",ylabel)
    colors = (:royalblue,:purple,:seagreen,:darkorange)
    for (k,(name,s)) in enumerate(series)
        color=colors[k]
        if !se_only
            lo,hi = hasproperty(s,:lower) ? (s.lower,s.upper) : (s.m.-s.se,s.m.+s.se)
            band!(ax,lags,lo,hi;color=(color,.18))
        end
        lines!(ax,lags,se_only ? s.se : s.m;label=name,color,linewidth=2.5)
    end
    axislegend(ax;position=:rt)
    Label(fig[2,1],se_only ? "SE across seeds; cross-run: delete-one-seed jackknife" :
        nulls ? "Observed: ±1 seed SE | Shuffles: central 95% of permutation means" :
        "Ribbons: ±1 SE | Cross-run geometry uses a delete-one-seed jackknife";tellwidth=false)
    return fig
end

function _rmd_write_tables(dir,a,series,same_curves,cross_curves,null_curves)
    lag_rows = vcat(collect(series),
        ["Orientation $(a.angles[k])°"=>s for (k,s) in enumerate(same_curves)],
        ["$(a.angles[u])° x $(a.angles[v])°"=>cross_curves[k] for (k,(u,v)) in enumerate(a.edges)])
    open(joinpath(dir,"lag_curves.csv"),"w") do io
        println(io,"comparison,lag,mean_r,seed_se,n_seed_or_pair_values,window_pair_se")
        for (name,s) in lag_rows,l in eachindex(s.m)
            println(io,join((name,l,s.m[l],s.se[l],s.counts[l],s.window_se[l]),','))
        end
    end
    open(joinpath(dir,"lag_slopes.csv"),"w") do io
        println(io,"comparison,slope_per_window,descriptive_ols_se,n_lags")
        for (name,s) in lag_rows
            println(io,join((name,_rmd_slope(s.m)...),','))
        end
    end
    open(joinpath(dir,"permutation_lags.csv"),"w") do io
        println(io,"control,permutation,lag,mean_r")
        for (k,s) in enumerate(null_curves),p in axes(s.draws,2),l in axes(s.draws,1)
            println(io,join((k==1 ? "seed labels" : "orientation labels",p,l,s.draws[l,p]),','))
        end
    end
    open(joinpath(dir,"sample_counts.csv"),"w") do io
        println(io,"seed,window,iteration_start,iteration_end,orientation_degrees,n_samples")
        for (i,seed) in enumerate(a.seeds),w in a.bins.ids,(k,angle) in enumerate(a.angles)
            println(io,join((seed,w,a.bins.starts[w],a.bins.ends[w],angle,a.counts[w,k,i]),','))
        end
    end
    open(joinpath(dir,"mean_window_matrices.csv"),"w") do io
        println(io,"comparison,window1,window2,mean_r")
        matrices = vcat(["Orientation $(a.angles[k])°"=>_rmd_reduce(@view(a.same[:,:,k,:]),3)
            for k in eachindex(a.angles)],
            ["$(a.angles[u])° x $(a.angles[v])°"=>_rmd_reduce(@view(a.cross[:,:,k,:]),3)
                for (k,(u,v)) in enumerate(a.edges)],
            ["Geometry"=>_rmd_reduce(a.stable,3),"Cross-run geometry"=>_rmd_reduce(a.cross_run,3),
             "Shuffled seeds"=>_rmd_reduce(@view(a.nulls[:,:,:,1]),3),
             "Shuffled orientations"=>_rmd_reduce(@view(a.nulls[:,:,:,2]),3)])
        for (name,c) in matrices,j in axes(c,2),i in axes(c,1)
            println(io,join((name,i,j,c[i,j]),','))
        end
    end
    open(joinpath(dir,"mean_geometry_cosines.csv"),"w") do io
        println(io,"comparison,window,orientation1,orientation2,mean_cosine")
        for (label,matrices) in (("Within seed",a.cosines),("Cyclic cross-run",a.cross_run_cosines))
            g=_rmd_reduce(matrices,4)
            for w in a.bins.ids,j in eachindex(a.angles),i in eachindex(a.angles)
                println(io,join((label,w,a.angles[i],a.angles[j],g[i,j,w]),','))
            end
        end
    end
    open(joinpath(dir,"analysis_parameters.txt"),"w") do io
        for key in (:warmup,:bin_size,:tolerance_degrees,:n_perm,:permutation_seed,:angles,:seeds)
            println(io,key," = ",getproperty(a,key))
        end
        println(io,"lag = all window pairs separated by lag; not anchored to the final window")
        println(io,"cross_orientation = first angle at earlier window, second angle at later window")
        println(io,"cross_run = cyclic next seed; matching neuron indices; correlate full cosine profiles")
        println(io,"uncertainty = seed SE; cyclic cross-run delete-one-seed SE; null central 95%")
        println(io,"window_pair_se and descriptive_ols_se are notebook-style descriptive statistics")
        println(io,"heatmap display = 0–1; signed values retained in CSVs")
    end
end

function run_colleague_analyses(batch_dir;warmup=3000,bin_size=1800,orientations=0:30:330,
        tolerance_degrees=1.,n_perm=100,permutation_seed=123,
        output_dir=joinpath(batch_dir,"Analysis","ColleagueAnalyses"))
    a = colleague_drift_analyses(batch_dir;warmup,bin_size,orientations,tolerance_degrees,n_perm,permutation_seed)
    np,nw = length(a.angles),length(a.bins.ids)
    same_curves = [_rmd_seed_lag_summary(@view a.same[:,:,k,:]) for k in 1:np]
    cross_curves = [_rmd_seed_lag_summary(@view a.cross[:,:,k,:]) for k in eachindex(a.edges)]
    series = ("Same orientation"=>_rmd_seed_lag_summary(_rmd_reduce(a.same,3)),
        "Cross orientation"=>_rmd_seed_lag_summary(_rmd_reduce(a.cross,3)),
        "Within-run geometry"=>_rmd_seed_lag_summary(a.stable),
        "Cross-run cosine profiles"=>_rmd_seed_lag_summary(a.cross_run;cyclic=true))
    null_curves = [_rmd_null_summary(@view a.nulls[:,:,:,k]) for k in 1:2]
    lags = 1:nw-1
    CairoMakie.activate!()
    dir = RotatingShapeSingleModel.RotatingShapeSingleReservoir.new_run_directory(output_dir)
    figures = Dict{String,Any}()
    figures["lag_comparison"] = _rmd_curve_plot(lags,series;title="Pattern and geometry stability across all window pairs")
    figures["geometry_controls"] = _rmd_curve_plot(lags,
        (series[3],"Shuffled seeds"=>null_curves[1],"Shuffled orientations"=>null_curves[2]);
        title="Geometry stability and permutation controls",nulls=true)
    figures["lag_se"] = _rmd_curve_plot(lags,series;title="Standard errors by lag",ylabel="Standard error",se_only=true)
    figures["orientation_heatmaps"] = _rmd_heatmaps(
        [_rmd_reduce(@view(a.same[:,:,k,:]),3) for k in 1:np],["$(a.angles[k])°" for k in 1:np])
    figures["geometry_stability_heatmaps"] = _rmd_heatmaps(
        [_rmd_reduce(a.stable,3),_rmd_reduce(a.cross_run,3),
         _rmd_reduce(@view(a.nulls[:,:,:,1]),3),_rmd_reduce(@view(a.nulls[:,:,:,2]),3)],
        ["Within-run geometry","Cross-run cosine profiles","Shuffled seeds","Shuffled orientations"];cols=2)
    angle_ticks = _rmd_angle_ticks(a.angles)
    for (label,source) in (("geometry_cosines",a.cosines),("cross_run_cosines",a.cross_run_cosines))
        g = _rmd_reduce(source,4)
        figures[label] = _rmd_heatmaps([g[:,:,w] for w in 1:nw],["Window $w" for w in 1:nw];
            cols=5,axislabel="Orientation",ticks=angle_ticks,scale_label="Cosine similarity")
    end
    counts_fig = Figure(resolution=(1050,235cld(np,4)))
    lag_fig = Figure(resolution=(1150,235cld(np,4)))
    for k in 1:np
        r,c = fldmod1(k,4)
        values = vec(a.counts[:,k,:])
        xs = sort(unique(values))
        ax = Axis(counts_fig[r,c];title="$(a.angles[k])°",xlabel="Samples per window",ylabel="Count")
        barplot!(ax,xs,[count(==(x),values) for x in xs];width=.8)
        ax2 = Axis(lag_fig[r,c];title="$(a.angles[k])°",xlabel="Lag (windows)",ylabel="Pearson r")
        s = same_curves[k]
        band!(ax2,lags,s.m.-s.se,s.m.+s.se;color=(:royalblue,.2))
        lines!(ax2,lags,s.m;color=:royalblue)
    end
    figures["sample_counts"] = counts_fig
    figures["orientation_lag"] = lag_fig
    cross_fig = Figure(resolution=(1080,560))
    ax = Axis(cross_fig[1,1];title="Cross-orientation drift",xlabel="Lag (windows)",ylabel="Pearson correlation")
    for s in cross_curves
        lines!(ax,lags,s.m;color=(:gray,.23))
    end
    s=series[2].second
    band!(ax,lags,s.m.-s.se,s.m.+s.se;color=(:purple,.18))
    lines!(ax,lags,s.m;color=:purple,linewidth=3,label="Mean across orientation pairs")
    axislegend(ax;position=:rt)
    figures["cross_orientation_lag"] = cross_fig
    for (name,fig) in figures
        CairoMakie.save(joinpath(dir,name*".png"),fig)
    end
    _rmd_write_tables(dir,a,series,same_curves,cross_curves,null_curves)
    return (;analysis=a,series,same_curves,cross_curves,null_curves,figures,output_dir=dir)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==1 || error("Usage: julia RotatingShapeSingle_ColleagueAnalyses.jl BATCH_DIR")
    BLAS.set_num_threads(1)
    p = run_colleague_analyses(ARGS[1])
    println("Saved notebook adaptation to ",p.output_dir)
    println("Seeds: ",length(p.analysis.seeds)," | windows: ",length(p.analysis.bins.ids),
        " | orientations: ",length(p.analysis.angles))
end


function export_ensemble_spikes(batch_dir, output_path)
    status = ensemble_status(batch_dir)
    isempty(status.pending) || error("Some runs are still incomplete.")
    isempty(status.completed) && error("No completed runs found.")

    output_path = abspath(output_path)
    ispath(output_path) && error("Output already exists: $output_path")
    mkpath(dirname(output_path))

    total_rows = 0
    nnodes = 0

    # Provided by the ensemble script. Publishes the CSV after writing finishes.
    _ensemble_atomic_write(output_path) do io
        chunk = IOBuffer()

        for seed in status.completed
            result = load_ensemble_run(batch_dir, seed)
            ts = result.ts
            nsteps, k = size(ts.spikes)

            length(ts.angle) == nsteps ||
                error("Orientation rows do not match spikes for run $seed.")
            all(x -> x == 0 || x == 1, ts.spikes) ||
                error("Nonbinary spikes in run $seed.")

            if nnodes == 0
                nnodes = k
                println(io, join(["node_$j" for j in 1:nnodes], ','),
                    ",iteration,orientation_degrees,run")
            end
            k == nnodes || error("Run $seed has a different number of nodes.")

            # Format rows in chunks to keep memory use low.
            spikes = permutedims(ts.spikes)
            row = repeat(UInt8[0x30, 0x2c], nnodes)  # "0,0,...,"

            for i in 1:nsteps
                @inbounds for j in 1:nnodes
                    row[2j - 1] = 0x30 + UInt8(spikes[j, i])
                end

                # Round to integer degrees, wrapping a rounded 360 back to 0.
                orientation = mod(round(Int, rad2deg(mod(ts.angle[i], 2pi))), 360)
                write(chunk, row)
                println(chunk, i, ',', orientation, ',', seed)

                i % 4096 == 0 && write(io, take!(chunk))
            end

            write(io, take!(chunk))
            total_rows += nsteps
            println("Exported run $seed ($nsteps rows)")
        end
    end

    return (; path=output_path, rows=total_rows, nnodes)
end

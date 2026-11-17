module RotatingShapeReservoir

using Random, SparseArrays, LinearAlgebra, Statistics, DelimitedFiles
using ..RotatingShapeEnvironment
# Reuse the language model's actual initialization and equations. Keep its module
# nested here to avoid replacing any language model already loaded in Main.
include(joinpath(@__DIR__,"..","Language","Language_Functions.jl"))
export initialize_reservoir, model_step!, run_rotating_model, save_run

mutable struct VisualReservoir{P}
    env::RotatingShapeEnvironment.VisualEnvironment
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
    initialize_reservoir(env; nnodes=100, p_link=.1, input_p_link=.1, sensory_fraction=.1,
        input_amp=5., subnetworks=1, n_subnet_edges=10, seed=1, input_seed=seed+1, ...)

Reuse language recurrent topology, initialization and learning parameters. N²
pixel inputs are flattened column-first: pixel index = row + (column-1)*N.
Left-image pixels project only to nodes 1:nnodes÷2; right-image pixels project
only to the other half. Select round(Int, sensory_fraction * nnodes/2) distinct
sensory nodes independently in each half. Only these nodes receive input.
Each allowed directed input edge is sampled independently
with probability input_p_link and has weight input_amp. Forbidden edges are zero.
"""
function initialize_reservoir(env; nnodes=100,p_link=.1,input_p_link=.1,sensory_fraction=.1,
        leak=.25,lrate_wmat=1.,lrate_targ=.01,targ_min=1.,input_amp=5.,
        learn_on=true,acts_neg=0,seed=1,input_seed=seed+1,
        subnetworks=1,n_subnet_edges=10)
    nnodes isa Integer && nnodes >= 2 && iseven(nnodes) ||
        throw(ArgumentError("nnodes must be an even integer >= 2"))
    0 <= input_p_link <= 1 || throw(ArgumentError("input_p_link must be in [0,1]"))
    0 <= sensory_fraction <= 1 || throw(ArgumentError("sensory_fraction must be in [0,1]"))
    isfinite(input_amp) && input_amp >= 0 || throw(ArgumentError("input_amp must be finite and nonnegative"))
    core = AllostaticLanguage.initialize_model(;nnodes,p_link,leak,lrate_wmat,
        lrate_targ,targ_min,input_amp,learn_on,acts_neg,seed,subnetworks,n_subnet_edges)
    rng = MersenneTwister(input_seed)
    npixels,half_pixels,half_nodes = env.N^2,env.N^2÷2,nnodes÷2
    n_sensory = round(Int,sensory_fraction * half_nodes)
    sensory_nodes = sort!(vcat(randperm(rng,half_nodes)[1:n_sensory],
        half_nodes .+ randperm(rng,half_nodes)[1:n_sensory]))
    rows,cols,weights = Int[],Int[],Float64[]
    for node in sensory_nodes
        pixels = node <= half_nodes ? (1:half_pixels) : (half_pixels+1:npixels)
        for pixel in pixels
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
        mean_abs_error_A=zeros(iterations),mean_abs_error_B=zeros(iterations),
        black_pixels=zeros(Int,iterations),black_left=zeros(Int,iterations),black_right=zeros(Int,iterations))
end

function _record_step!(ts,i,model,frame)
        ts.spikes[i,:] .= model.spikes
        ts.time[i],ts.angle[i] = frame.time,frame.angle
        ts.mean_act[i],ts.mean_error[i],ts.mean_target[i] =
            model.mean_act,model.mean_err,model.mean_targ
        errors = abs.(model.acts .- model.targets)
        half = model.params.nnodes ÷ 2
        ts.mean_abs_error[i] = mean(errors)
        ts.mean_abs_error_A[i] = mean(@view errors[1:half])
        ts.mean_abs_error_B[i] = mean(@view errors[half+1:end])
        ts.black_left[i] = count(@view frame.object_mask[:,1:model.env.N÷2])
        ts.black_right[i] = count(@view frame.object_mask[:,model.env.N÷2+1:end])
        ts.black_pixels[i] = ts.black_left[i]+ts.black_right[i]
    return ts
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
    result = (;model,ts,start_wmat,iterations,start_step,output_dir=save ? abspath(output_dir) : nothing)
    save && save_run(result,output_dir)
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
        ["iteration","time","angle_radians","angle_degrees_wrapped","black_pixels","black_left","black_right"],
        hcat(1:result.iterations,ts.time,ts.angle,rad2deg.(mod.(ts.angle,2pi)),
            ts.black_pixels,ts.black_left,ts.black_right))
    write_csv(joinpath(dir,"reservoir_summary.csv"),
        ["mean_activation","mean_signed_error","mean_target","spike_count","mean_absolute_error","mean_absolute_error_A","mean_absolute_error_B"],
        hcat(ts.mean_act,ts.mean_error,ts.mean_target,sum(ts.spikes;dims=2),ts.mean_abs_error,ts.mean_abs_error_A,ts.mean_abs_error_B))
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
        for key in (:side_length,:object_radius,:angular_speed,:dt,:theta0)
            println(io,key," = ",getproperty(model.env,key))
        end
        println(io,"input_encoding = black:1 white:0")
        println(io,"pixel_index = row + (column - 1) * N")
        println(io,"subnetwork_A = nodes 1:",model.params.nnodes÷2)
        println(io,"subnetwork_B = nodes ",model.params.nnodes÷2+1,":",model.params.nnodes)
    end
    return abspath(dir)
end
end # module

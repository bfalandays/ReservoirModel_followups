module RotatingShapeInteractive

using CairoMakie
using ..RotatingShapeEnvironment
using ..RotatingShapeReservoir
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

function _new_run_directory(root)
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

function _interactive_state(env; iterations=40000,output_dir,kwargs...)
    iterations isa Integer && iterations > 0 || throw(ArgumentError("iterations must be positive"))
    model = initialize_reservoir(deepcopy(env);kwargs...)
    ts = RotatingShapeReservoir._spike_history(iterations,model.params.nnodes)
    return InteractiveRun(model,deepcopy(model),ts,render_frame(model.env),iterations,
        copy(model.wmat),model.env.step,abspath(output_dir),_new_run_directory(output_dir),0)
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
        frame = RotatingShapeReservoir.model_step!(state.model)
        RotatingShapeReservoir._record_step!(state.ts,state.model.n,state.model,frame)
        state.last_frame = frame
    end
    return count
end

function _reset_state!(state)
    # Keep the model object itself, so the main script's `model` binding stays live.
    for name in fieldnames(typeof(state.model))
        setproperty!(state.model,name,deepcopy(getproperty(state.initial_model,name)))
    end
    state.ts = RotatingShapeReservoir._spike_history(state.iterations,state.model.params.nnodes)
    state.last_frame = render_frame(state.model.env)
    state.saved_iterations = 0
    state.run_dir = _new_run_directory(state.output_root)
    return state
end

function _network_geometry(model)
    n = model.params.nnodes
    split = model.params.subnetworks == 1
    half = n ÷ 2
    points = Point2f[]
    for node in 1:n
        local_node = split ? mod(node-1,half)+1 : node
        group_size = split ? half : n
        angle = 2pi*(local_node-1)/group_size
        cx = split ? (node <= half ? -1.5 : 1.5) : 0.
        push!(points,Point2f(cx+cos(angle),sin(angle)))
    end
    within_points,cross_points = Point2f[],Point2f[]
    within_dirs,cross_dirs = Vec2f[],Vec2f[]
    for src in 1:n,dst in 1:n
        (src==dst || iszero(model.link_mat[src,dst])) && continue
        p,q = points[src],points[dst]
        dx,dy = q[1]-p[1],q[2]-p[2]
        length = hypot(dx,dy)
        ux,uy = dx/length,dy/length
        cross = split && ((src<=half)!=(dst<=half))
        push!(cross ? cross_points : within_points,Point2f(p[1]+.035ux,p[2]+.035uy))
        push!(cross ? cross_dirs : within_dirs,Vec2f(dx-.07ux,dy-.07uy))
    end
    return (;points,within_points,cross_points,within_dirs,cross_dirs,split)
end

function _node_colors(model)
    half = model.params.nnodes ÷ 2
    return [model.spikes[i]>0 ? :yellow :
        (model.params.subnetworks==1 && i>half ? :seagreen : :royalblue)
        for i in eachindex(model.spikes)]
end

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
    image_data = Observable(RotatingShapeEnvironment.plot_pixels(state.last_frame))
    absolute_errors = Observable(abs.(model.acts .- model.targets))
    error_history = Observable(Point2f[])
    error_history_A = Observable(Point2f[])
    error_history_B = Observable(Point2f[])
    target_values = Observable(copy(model.targets))
    half = model.params.nnodes ÷ 2
    targets_A = lift(values -> values[1:half],target_values)
    targets_B = lift(values -> values[half+1:end],target_values)
    # Identical bin edges make the overlaid counts directly comparable.
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
    if !isempty(geometry.cross_points)
        arrows!(ax_network,geometry.cross_points,geometry.cross_dirs;
            color=(:darkorange,.8),linewidth=1.4,arrowsize=9)
    end
    scatter!(ax_network,geometry.points;color=colors,markersize=15,
        strokecolor=:black,strokewidth=.6)
    limits!(ax_network,geometry.split ? -2.8 : -1.35,geometry.split ? 2.8 : 1.35,-1.35,1.35)
    Label(fig[3,1],"Yellow: spiking | Blue: A, silent | Green: B, silent\nGray: within-subnetwork edges | Orange: cross-subnetwork edges";tellwidth=false)

    ax_stimulus = Axis(fig[2,2];aspect=DataAspect(),title="Stimulus — frame driving current spikes",xlabel="x",ylabel="y")
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    heatmap!(ax_stimulus,edges,edges,image_data;colormap=[:black,:white],
        colorrange=(0,1),interpolate=false)
    RotatingShapeEnvironment.decorate_axis!(ax_stimulus,env;show_split=true)
    Label(fig[3,2],"$(env.N) × $(env.N) pixels | Black = sensory 1, white = sensory 0\nLeft field → A | Right field → B";tellwidth=false)

    ax_errors = Axis(fig[4,1];title="Mean absolute error over time",
        xlabel="Model iteration",ylabel="Mean absolute error")
    lines!(ax_errors,error_history_A;color=:royalblue,linewidth=2,label="A")
    lines!(ax_errors,error_history_B;color=:seagreen,linewidth=2,label="B")
    axislegend(ax_errors;position=:rt)
    Label(fig[5,1],"mean(abs(activation − target))\nMean within each subnetwork, per iteration";tellwidth=false)

    ax_targets = Axis(fig[4,2];title="Target value distribution",
        xlabel="Target value",ylabel="Number of nodes")
    hist!(ax_targets,targets_A;bins=target_bins,color=(:royalblue,.5),
        strokecolor=(:royalblue,.7),strokewidth=.5,label="A")
    hist!(ax_targets,targets_B;bins=target_bins,color=(:seagreen,.5),
        strokecolor=(:seagreen,.7),strokewidth=.5,label="B")
    axislegend(ax_targets;position=:rt)
    Label(fig[5,2],"Current targets | Blue: A, green: B\nShared bins; transparent bars";tellwidth=false)

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
        image_data[] = RotatingShapeEnvironment.plot_pixels(state.last_frame)
        # Recompute using current targets, which learning may have updated.
        absolute_errors[] = abs.(state.model.acts .- state.model.targets)
        # All completed steps are recorded, including advance(nsteps) calls.
        n = state.model.n
        for (history,values) in ((error_history,state.ts.mean_abs_error),
                (error_history_A,state.ts.mean_abs_error_A),
                (error_history_B,state.ts.mean_abs_error_B))
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
        status[] = "Iteration $(state.model.n) / $iterations  |  Input t=$(round(frame.time;digits=3)) s  |  Angle $(angle)°  |  Spikes $(Int(sum(state.model.spikes))) / $(state.model.params.nnodes)"
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
            @error "RotatingShape interactive action failed" exception=(err,catch_backtrace())
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
    screen = show ? display(fig;backend=backend_module,inline=false,title="RotatingShape — live reservoir") : nothing
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
    return (;fig,screen,state,running,node_colors=colors,image_data,absolute_errors,error_history,error_history_A,error_history_B,target_values,targets_A,targets_B,target_bins,status,message,
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

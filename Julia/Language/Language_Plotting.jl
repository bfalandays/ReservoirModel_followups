using CairoMakie

function plot_raster_plot(spikes, stream, path; training_steps=size(spikes,1))
    values = copy(spikes)
    if training_steps < size(values,1)
        # Red = silence during input; yellow = spike; blue = silence after cue.
        tail = @view values[training_steps+1:end,:]
        tail[tail .== 0] .= -1
    end
    n, nnodes = size(values)
    fig = Figure(resolution=(1200,800))
    ax = Axis(fig[1,1]; xlabel="Token / time step",ylabel="Reservoir node",
        xticks=(1:n,stream),xticklabelrotation=pi/4)
    heatmap!(ax, .5:1:(n+.5), .5:1:(nnodes+.5), values;
        colormap=[:blue,:red,:yellow],colorrange=(-1,1))
    save(path,fig)
    return fig
end

function plot_cor_matrix(spikes, stream, path)
    cm = correlation_matrix(spikes)
    n = length(stream)
    fig = Figure(resolution=(1400,1100))
    ax = Axis(fig[1,1]; xticks=(1:n,stream),yticks=(1:n,stream),
        xticklabelrotation=pi/4,yreversed=true,aspect=DataAspect())
    # Makie first matrix dimension is x; correlation rows are y.
    hm = heatmap!(ax,1:n,1:n,permutedims(cm);
        colormap=:RdYlBu,colorrange=(-1,1),nan_color=:transparent)
    for i in 1:n, j in 1:i
        label = isnan(cm[i,j]) ? "NA" : string(round(cm[i,j];digits=2))
        text!(ax,j,i;text=label,align=(:center,:center),fontsize=13)
    end
    Colorbar(fig[1,2],hm;label="Pearson r across nodes")
    save(path,fig)
    return fig
end

function plot_results(result, probes; output_dir=joinpath(@__DIR__,"Figures"))
    mkpath(output_dir)
    plot_raster_plot(result.start.spikes,result.start.stream,joinpath(output_dir,"Fig1.png"))
    plot_cor_matrix(result.start.spikes,result.start.stream,joinpath(output_dir,"Fig2.png"))
    for probe in probes
        spikes = vcat(result.finish.spikes[end-11:end,:],probe.ts.spikes)
        stream = vcat(result.finish.stream[end-11:end],probe.ts.stream)
        # Color only absent-input silence blue (not cue-present probe silence).
        prefix = "$(AllostaticLanguage.input_keys[probe.first_input]).$(probe.input_num)"
        plot_raster_plot(spikes,stream,joinpath(output_dir,"Fig3.$prefix.png");
            training_steps=12+probe.input_num)
        plot_cor_matrix(spikes,stream,joinpath(output_dir,"Fig4.$prefix.png"))
    end
    return output_dir
end

"""
    plot_network_structure(model; label_nodes=false, show_adjacency=true,
                           show_input=true, input_edge_alpha=.3,
                           output_path=...)

Plot structural recurrent connections from `model.link_mat`, with directed
arrows. Two-subnetwork models use side-by-side circular layouts: A is blue,
B is green, and cross-subnetwork edges are orange. Optionally show the full
source-by-destination recurrent adjacency matrix. Input channels are purple
squares above the reservoir, with purple directed edges from `input_wmat`.
Purple reservoir-node outlines identify nodes receiving input connections.
Returns the figure and saves a PNG.
"""
function plot_network_structure(model; label_nodes=false, show_adjacency=true,
        show_input=true, input_edge_alpha=.3,
        output_path=joinpath(@__DIR__,"Figures","Network_structure.png"))
    0 <= input_edge_alpha <= 1 || throw(ArgumentError("input_edge_alpha must be in [0,1]"))
    links = model.link_mat
    n = size(links,1)
    n > 0 && size(links,2) == n || throw(ArgumentError("link_mat must be nonempty and square"))
    split = hasproperty(model.params,:subnetworks) && model.params.subnetworks == 1
    split && !iseven(n) && throw(ArgumentError("Two subnetworks require even nnodes"))
    size(model.input_wmat,2) == n || throw(DimensionMismatch("input_wmat must have one column per reservoir node"))
    half = n ÷ 2
    positions = Point2f[]
    for node in 1:n
        size_group = split ? half : n
        local_node = split ? mod(node-1,half)+1 : node
        theta = 2pi*(local_node-1)/size_group
        center_x = split ? (node <= half ? -1.5 : 1.5) : 0.
        push!(positions,Point2f(center_x+cos(theta),sin(theta)))
    end
    within_starts, cross_starts = Point2f[],Point2f[]
    within_dirs, cross_dirs = Vec2f[],Vec2f[]
    for source in 1:n, destination in 1:n
        (source == destination || iszero(links[source,destination])) && continue
        p,q = positions[source],positions[destination]
        dx,dy = q[1]-p[1],q[2]-p[2]
        distance = hypot(dx,dy)
        ux,uy = dx/distance,dy/distance
        # End the arrow just before the target node so its head stays visible.
        start = Point2f(p[1]+.035ux,p[2]+.035uy)
        direction = Vec2f(dx-.07ux,dy-.07uy)
        cross = split && ((source <= half) != (destination <= half))
        push!(cross ? cross_starts : within_starts,start)
        push!(cross ? cross_dirs : within_dirs,direction)
    end
    input_positions = Point2f[]
    input_starts, input_dirs = Point2f[],Vec2f[]
    nchannels = size(model.input_wmat,1)
    input_labels = nchannels == 5 ? ["man","dog","walks","bites","_"] :
                   ["Input $i" for i in 1:nchannels]
    if show_input
        width = split ? 2.1 : 1.15
        xs = nchannels == 1 ? [0.] : collect(range(-width,width;length=nchannels))
        for x in xs
            push!(input_positions,Point2f(x,2.05))
        end
        for channel in 1:nchannels, node in 1:n
            iszero(model.input_wmat[channel,node]) && continue
            p,q = input_positions[channel],positions[node]
            dx,dy = q[1]-p[1],q[2]-p[2]
            distance = hypot(dx,dy)
            ux,uy = dx/distance,dy/distance
            push!(input_starts,Point2f(p[1]+.06ux,p[2]+.06uy))
            push!(input_dirs,Vec2f(dx-.095ux,dy-.095uy))
        end
    end
    fig = Figure(resolution=show_adjacency ? (1500,850) : (1000,800))
    ax = Axis(fig[1,1];aspect=DataAspect(),title=split ? "Two-subnetwork reservoir" : "Reservoir network")
    hidedecorations!(ax)
    hidespines!(ax)
    if !isempty(input_starts)
        arrows!(ax,input_starts,input_dirs;color=(:mediumpurple,input_edge_alpha),
            linewidth=1,arrowsize=8)
    end
    if !isempty(within_starts)
        arrows!(ax,within_starts,within_dirs;color=(:slategray,.2),linewidth=.6,arrowsize=6)
    end
    if !isempty(cross_starts)
        arrows!(ax,cross_starts,cross_dirs;color=(:darkorange,.9),linewidth=1.8,arrowsize=11)
    end
    colors = [split && node > half ? :seagreen : :royalblue for node in 1:n]
    sensory_nodes = vec(any(model.input_wmat .!= 0;dims=1))
    outlines = [show_input && sensory_nodes[i] ? :mediumpurple : :white for i in 1:n]
    scatter!(ax,positions;color=colors,markersize=11,strokecolor=outlines,strokewidth=1.3)
    if show_input
        scatter!(ax,input_positions;color=:mediumpurple,marker=:rect,markersize=16,
            strokecolor=:white,strokewidth=1)
        for channel in 1:nchannels
            p = input_positions[channel]
            text!(ax,p[1],p[2]+.13;text=input_labels[channel],align=(:center,:bottom),fontsize=16)
        end
        text!(ax,0,2.55;text="Input layer",align=(:center,:center),fontsize=18)
    end
    if label_nodes
        for node in 1:n
            p = positions[node]
            text!(ax,p[1],p[2]+.075;text=string(node),align=(:center,:bottom),fontsize=10)
        end
    end
    if split
        xlims!(ax,-2.8,2.8)
        ylims!(ax,-1.35,show_input ? 2.8 : 1.35)
    else
        limits!(ax,show_input ? -1.6 : -1.35,show_input ? 1.6 : 1.35,
            -1.35,show_input ? 2.8 : 1.35)
    end
    edgecount = count(!iszero,links)
    caption = split ? "A: nodes 1–$half (blue) | B: nodes $(half+1)–$n (green)\n$(length(cross_starts)) directed cross edges (orange); $edgecount total recurrent edges" :
                      "$n nodes; $edgecount directed recurrent edges"
    if show_input
        caption *= "\nPurple squares/arrows: input layer and $(length(input_starts)) input edges; purple outlines: sensory nodes"
    end
    Label(fig[2,1],caption;tellwidth=false)
    if show_adjacency
        axmat = Axis(fig[1,2];title="Recurrent adjacency",xlabel="Destination node",
            ylabel="Source node",aspect=DataAspect())
        edges = .5:1:(n+.5)
        heatmap!(axmat,edges,edges,permutedims(Float64.(links .!= 0));
            colormap=[:white,:black],colorrange=(0,1),interpolate=false)
        limits!(axmat,.5,n+.5,n+.5,.5)
        if split
            vlines!(axmat,[half+.5];color=:darkorange,linestyle=:dash)
            hlines!(axmat,[half+.5];color=:darkorange,linestyle=:dash)
        end
        Label(fig[2,2],"Black: connection present | White: no connection";tellwidth=false)
    end
    if output_path !== nothing
        mkpath(dirname(output_path))
        save(output_path,fig)
    end
    return fig
end


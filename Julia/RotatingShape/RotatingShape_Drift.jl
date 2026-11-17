module RotatingShapeDrift

using CairoMakie, LinearAlgebra, Statistics, DelimitedFiles
export window_spike_pca, plot_orientation_pc1, plot_spike_pc12, plot_subnetwork_pc12, plot_subnetwork_pc1_time, plot_horizontal_subnetwork_pc12, load_drift_run

"""Read spike vectors and their matching input orientations from a saved run directory."""
function load_drift_run(dir::AbstractString)
    spikes = readdlm(joinpath(dir,"spikes.csv"),',',Float64;skipstart=1)
    metadata,header = readdlm(joinpath(dir,"timesteps.csv"),',',Float64;header=true)
    angle_col = findfirst(==("angle_radians"),vec(header))
    angle_col === nothing && throw(ArgumentError("timesteps.csv lacks angle_radians"))
    size(spikes,1)==size(metadata,1) || throw(ArgumentError("Spike and timestep row counts differ"))
    metadata[:,1]==collect(1:size(spikes,1)) || throw(ArgumentError("Expected consecutive iteration rows starting at 1"))
    return (;ts=(;spikes,angle=metadata[:,angle_col]))
end

"""
    window_spike_pca(result; window=5000:6000, nodes=:A, ncomponents=1)

Fit PCA to the selected inclusive iteration window. Rows are observations and
columns are neurons. Center each neuron's binary spikes without variance scaling.
`nodes` may be :A (first half), :B (second half), :all, or a vector of node indices.
Returns PC1 scores, loadings, the training mean and explained variance, as well as
the selected spike matrix and matching wrapped input orientations. Use ncomponents
to retain additional PCs in component_scores, component_loadings and
explained_variances. The original scores/loadings fields always refer to PC1.
No model is run.
"""
function window_spike_pca(result;window=5000:6000,nodes=:A,ncomponents=1)
    ts = result.ts
    n,nneurons = size(ts.spikes)
    length(ts.angle)==n || throw(ArgumentError("One orientation is required per spike row"))
    window isa AbstractUnitRange{<:Integer} || throw(ArgumentError("window must be a consecutive integer range"))
    length(window)>=2 || throw(ArgumentError("PCA requires at least two iterations"))
    first(window)>=1 && last(window)<=n || throw(ArgumentError("Window must lie within completed iterations 1:$n"))
    if nodes in (:A,:B)
        iseven(nneurons) || throw(ArgumentError("Subnetwork analysis requires an even node count"))
        ids = nodes==:A ? collect(1:nneurons÷2) : collect(nneurons÷2+1:nneurons)
    elseif nodes===:all
        ids = collect(1:nneurons)
    elseif nodes isa AbstractVector{<:Integer}
        ids = collect(nodes)
    else
        throw(ArgumentError("nodes must be :A, :B, :all or a vector of indices"))
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
        groups=(:A,:B), savepath=nothing)

Static time-colored PC1 versus orientation trajectories. Fit each group's PCA on
the requested window only. Lines follow chronological order; connections across
0/360 degrees are omitted. Returns (;fig, analyses). For later cross-window
comparisons, reuse a fixed PCA basis rather than comparing independent axes.
"""
function plot_orientation_pc1(source;window=5000:6000,groups=(:A,:B),savepath=nothing)
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
    plot_subnetwork_pc12(result_or_directory; window=5000:6000, savepath=nothing)

Two static PC1-PC2 trajectories, fitted separately to A (first half of neurons)
and B (second half). Both use the same iteration window and color scale. Each
neuron is centered without scaling. Returns (;fig, analyses), ordered A then B.
Independent PCA coordinates are not aligned between subnetworks or windows.
"""
function plot_subnetwork_pc12(source;window=5000:6000,savepath=nothing)
    result = source isa AbstractString ? load_drift_run(source) : source
    analyses = [window_spike_pca(result;window,nodes=group,ncomponents=2)
        for group in (:A,:B)]
    CairoMakie.activate!()
    fig = Figure(resolution=(1350,650))
    colorrange = (Float64(first(window)),Float64(last(window)))
    for (j,a) in enumerate(analyses)
        group = j==1 ? "A" : "B"
        pct = round.(100 .* a.explained_variances;digits=1)
        ax = Axis(fig[1,j];aspect=DataAspect(),title="Subnetwork $group",
            xlabel="PC1 ($(pct[1])% variance)",ylabel="PC2 ($(pct[2])% variance)")
        lines!(ax,a.component_scores[:,1],a.component_scores[:,2];
            color=a.iterations,colormap=:viridis,colorrange,linewidth=1.5)
        scatter!(ax,a.component_scores[:,1],a.component_scores[:,2];
            color=a.iterations,colormap=:viridis,colorrange,markersize=3)
    end
    Colorbar(fig[1,3];colormap=:viridis,colorrange,label="Model iteration")
    Label(fig[0,1:2],"Iterations $(first(window))–$(last(window)) | centered spikes | PCA fitted separately per subnetwork";tellwidth=false)
    if savepath !== nothing
        mkpath(dirname(abspath(savepath)))
        save(savepath,fig)
    end
    return (;fig,analyses)
end
"""
    plot_subnetwork_pc1_time(result_or_directory; window=5000:6000,
        savepath=nothing, azimuth=1.55pi, elevation=pi/8)

3D trajectory: model iteration on x, PC1 of A on y, and PC1 of B on z.
Set interactive=true to open a rotatable GLMakie window with zoom/reset controls.
The default remains a static CairoMakie figure. backend=:cairo, show=false can
be used to check interactive controls offscreen.
Fit each subnetwork's PCA independently within the same window, using centered,
unscaled spikes. Connect paired scores from matching iterations and color by
iteration. Returns (;fig, analyses, coordinates), analyses ordered A then B and
coordinates containing columns [iteration, PC1_A, PC1_B]. Time is measured in
model iterations. Independent PCA signs do not establish synchrony by themselves.
"""
function plot_subnetwork_pc1_time(source;window=5000:6000,savepath=nothing,
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
    return Base.invokelatest(_build_subnetwork_pc1_time,source;window,savepath,
        azimuth,elevation,interactive,show,backend_module)
end

function _build_subnetwork_pc1_time(source;window,savepath,azimuth,elevation,
        interactive,show,backend_module)
    result = source isa AbstractString ? load_drift_run(source) : source
    analyses = [window_spike_pca(result;window,nodes=group) for group in (:A,:B)]
    a,b = analyses
    coordinates = hcat(a.iterations,a.scores,b.scores)
    fig = Figure(resolution=(1000,850))
    pct = round.([100*a.explained_variance,100*b.explained_variance];digits=1)
    ax = Axis3(fig[1,1];title="Subnetwork PC1 trajectories through time",
        xlabel="Model iteration",
        ylabel="A: PC1 ($(pct[1])% variance)",zlabel="B: PC1 ($(pct[2])% variance)",
        azimuth,elevation,aspect=(1.6,1,1))
    colorrange = (Float64(first(window)),Float64(last(window)))
    lines!(ax,a.iterations,a.scores,b.scores;color=a.iterations,
        colormap=:viridis,colorrange,linewidth=2)
    scatter!(ax,a.iterations,a.scores,b.scores;color=a.iterations,
        colormap=:viridis,colorrange,markersize=3)
    Colorbar(fig[1,2];colormap=:viridis,colorrange,label="Model iteration")
    colgap!(fig.layout,1,90) # Leave room for the 3D axis labels beside the colorbar.
    Label(fig[0,1],"Iterations $(first(window))–$(last(window)) | centered spikes | separate PCA fits for A and B";tellwidth=false)
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
        title="Subnetwork PC1 trajectories through time") : nothing
    return (;fig,analyses,coordinates,axis=ax,screen,zoom_slider,reset_button)
end
"""
    plot_horizontal_subnetwork_pc12(result_or_directory; window=5000:6000,
        target_angle_degrees=90., tolerance_degrees=1., savepath=nothing)

Select spike rows by their matching input orientation, then fit two PCs separately
for each subnetwork using ONLY those rows. With the current default L, 90 degrees
puts the long arm along the bottom with its short arm pointing up. Angle matching
uses circular distance with inclusive tolerance. Use window=nothing for all
completed iterations. At least three selected rows with spike variance are needed.
Returns (;fig, analyses, selected_iterations, target_angle_degrees,
tolerance_degrees). Analyses retain the original iteration indices. Lines connect
successive selected observations, not successive simulation timesteps.
"""
function plot_horizontal_subnetwork_pc12(source;window=5000:6000,
        target_angle_degrees=90.,tolerance_degrees=1.,savepath=nothing)
    result = source isa AbstractString ? load_drift_run(source) : source
    ts = result.ts
    n = size(ts.spikes,1)
    length(ts.angle)==n || throw(ArgumentError("One orientation is required per spike row"))
    window = window===nothing ? (1:n) : window
    window isa AbstractUnitRange{<:Integer} && !isempty(window) &&
        first(window)>=1 && last(window)<=n ||
        throw(ArgumentError("window must be a nonempty consecutive range within completed iterations 1:$n"))
    isfinite(target_angle_degrees) || throw(ArgumentError("Target angle must be finite"))
    isfinite(tolerance_degrees) && 0<=tolerance_degrees<=180 ||
        throw(ArgumentError("Angular tolerance must be in [0,180] degrees"))
    target_angle_degrees = mod(Float64(target_angle_degrees),360.)
    angles = rad2deg.(mod.(Float64.(ts.angle[window]),2pi))
    all(isfinite,angles) || throw(ArgumentError("Orientations must be finite"))
    distance = abs.(mod.(angles .- target_angle_degrees .+ 180.,360.) .- 180.)
    # Small numerical allowance includes nominally exact angles after many turns.
    selected_iterations = collect(window)[distance .<= tolerance_degrees+1e-8]
    length(selected_iterations)>=3 || throw(ArgumentError(
        "Only $(length(selected_iterations)) matching frames; select a longer window or larger angular tolerance (at least 3 required)"))
    selected = (;ts=(;spikes=ts.spikes[selected_iterations,:],angle=ts.angle[selected_iterations]))
    analyses = [merge(window_spike_pca(selected;window=1:length(selected_iterations),
        nodes=group,ncomponents=2),(;iterations=copy(selected_iterations))) for group in (:A,:B)]
    CairoMakie.activate!()
    fig = Figure(resolution=(1350,650))
    colorrange = (Float64(first(selected_iterations)),Float64(last(selected_iterations)))
    for (j,a) in enumerate(analyses)
        group = j==1 ? "A" : "B"
        pct = round.(100 .* a.explained_variances;digits=1)
        ax = Axis(fig[1,j];aspect=DataAspect(),title="Subnetwork $group — selected orientations",
            xlabel="PC1 ($(pct[1])% variance)",ylabel="PC2 ($(pct[2])% variance)")
        lines!(ax,a.component_scores[:,1],a.component_scores[:,2];
            color=a.iterations,colormap=:viridis,colorrange,linewidth=1.5)
        scatter!(ax,a.component_scores[:,1],a.component_scores[:,2];
            color=a.iterations,colormap=:viridis,colorrange,markersize=7)
    end
    Colorbar(fig[1,3];colormap=:viridis,colorrange,label="Model iteration")
    Label(fig[0,1:2],"$(target_angle_degrees)° ± $(tolerance_degrees)° | $(length(selected_iterations)) selected frames | iterations $(first(window))–$(last(window))";tellwidth=false)
    if savepath !== nothing
        mkpath(dirname(abspath(savepath)))
        save(savepath,fig)
    end
    return (;fig,analyses,selected_iterations,target_angle_degrees,tolerance_degrees)
end
end # module

function interactive_plot(model)
    #### plotting
    adata = [:pos, :heading, :inputs, :acts, :targets, :spikes, :inhibitory_nodes, :output_acts, :wmat, :collisions]
    mdata = [:n, :nnodes, :p_link, :leak, :leaktype, :lrate_wmat, :lrate_targ, :targ_min, :wheel_radius, :input_amp, :sens_angles, :learn_on]

    fig, mainAx, abmobs = Agents.abmplot(model; 
        dummystep, model_step!,
        add_controls = true, enable_inspection = true,
        adata, mdata, figure = (; resolution = (1800,600))
    )

    ## PLOTTING AGENTS IN SPACE
    # get data for plotting as observables
    positionData = @lift begin
        # the output has a row for every agent, and each row has a circle for plotting the body, and the positions of the left and right sensor, respectively, for plotting as points
        positions = $(abmobs.adf).pos[end-1:end]
        headings = $(abmobs.adf).heading[end-1:end]
        get_leftSensPos(pos, heading) = (pos[1] + (model.agent_radius * cos(heading + deg2rad(model.wall_sens_angles[1]))), pos[2] + (model.agent_radius * sin(heading + deg2rad(model.wall_sens_angles[1]))))
        get_rightSensPos(pos, heading) = (pos[1] + (model.agent_radius * cos(heading - deg2rad(model.wall_sens_angles[1]))), pos[2] + (model.agent_radius * sin(heading - deg2rad(model.wall_sens_angles[1]))))
        a = [[gb.Circle(GLMakie.Point2f(positions[i]), model.agent_radius), get_leftSensPos(positions[i], headings[i]), get_rightSensPos(positions[i], headings[i])] for i in 1:2]
        b = reduce(hcat, a)
    end
    poly!(mainAx, @lift([$positionData[1,1], $positionData[1,2]]), color = [:gray50, :black])
    scatter!(mainAx, @lift([$positionData[2,1], $positionData[2,2]]) ; markersize = 10, color = :red)
    scatter!(mainAx, @lift([$positionData[3,1], $positionData[3,2]]) ; markersize = 10, color = :blue)
    # scatterlines!(ax, positions; color = :black, markersize = 5, markeralpha = .01)

    ### Set up subplots
    subLayout = fig[1, 2:3] = GridLayout(2,5)

    ## PLOTTING SENSORS
    sensors_layout = subLayout[1:2,1]
    ax_sensors = [Axis(sensors_layout[i,1]; ylabel = i == 1 ? "Grey" : "Black") for i in 1:2]
    #[xlims!(ax_sensors[i], (0,1)) for i in 1:2]
    #[xlims!(ax_sensors[i], (-90,90)) for i in 1:2]
    [ylims!(ax_sensors[i], (0,1.2)) for i in 1:2]
    sensData = @lift begin
        $(abmobs.adf).inputs[end-1:end]
    end
    sensColors=reduce(vcat,[:red, :blue, repeat([:black],length(unique(model.sens_angles)))])
    #sensColors=reduce(vcat,[:red, :blue, repeat([:black],length(model.sens_anglesL)*2)])
    #sensColors= [:red, :blue, :black]
    #sensColors = :black
    barplot!(ax_sensors[1], @lift($sensData[1]); color = sensColors, strokecolor = sensColors, strokewidth = 1)
    barplot!(ax_sensors[2], @lift($sensData[2]); color = sensColors, strokecolor = sensColors, strokewidth = 1)

    ## PLOTTING EFFECTORS
    effectors_layout = subLayout[1:2,2]
    ax_effectors = [Axis(effectors_layout[i,1]; ylabel = "Value", xticks = (1:2, ["Left", "Right"])) for i in 1:2]
    [xlims!(ax_effectors[i], (0,3)) for i in 1:2]
    [ylims!(ax_effectors[i], (0,1)) for i in 1:2]
    effectorData = @lift begin
        $(abmobs.adf).output_acts[end-1:end]
    end
    barplot!(ax_effectors[1], @lift($effectorData[1]); color = [:red, :blue], strokecolor = :black, strokewidth = 1)
    barplot!(ax_effectors[2], @lift($effectorData[2]); color = [:red, :blue], strokecolor = :black, strokewidth = 1)

    ## PLOTTING SUMMARY
    summary_layout = subLayout[1:2,3]
    ax_summary = [Axis(summary_layout[i,1]; ylabel = "Value", xticks = (1:3, ["meanActs","meanErrs", "meanTargs"]), xticklabelrotation = pi/8) for i in 1:2]
    [xlims!(ax_summary[i], (0,4)) for i in 1:2]
    [ylims!(ax_summary[i], (-3,3)) for i in 1:2]
    summaryData = @lift begin
        meanActs = mean.($(abmobs.adf).acts[end-1:end])
        meanTargs = mean.($(abmobs.adf).targets[end-1:end])
        meanErrs = meanActs .- meanTargs
        return reduce(hcat,[meanActs, meanErrs, meanTargs])
    end
    barplot!(ax_summary[1], @lift($summaryData[1,1:3]); color = :black, strokecolor = :black, strokewidth = 1)
    barplot!(ax_summary[2], @lift($summaryData[2,1:3]); color = :black, strokecolor = :black, strokewidth = 1)

    # ## PLOTTING NETWORK
    # network_layout = subLayout[1:2,4]
    # ax_network = [Axis(network_layout[i,1]) for i in 1:2]
    # networkData = @lift begin
    #     spikes = $(abmobs.adf).spikes[end-1:end]
    #     inhibitory_nodes = $(abmobs.adf).inhibitory_nodes[end-1:end]
    #     #colors = [[x == 0 ? :blue : :yellow for x in spikes[i]] for i in 1:2]
    #     colors = [[spikes[i][j] == 0 ? (inhibitory_nodes[i][j] == 1 ? :blue4 : :red4) : (inhibitory_nodes[i][j] == 1 ? :steelblue2 : :red1)
    #         for j in 1:length(spikes[i])] 
    #             for i in 1:2]
    # end
    # G = [SimpleWeightedDiGraph(ag.link_mat) for ag in allagents(model)]
    # pos_x, pos_y = GraphPlot.spring_layout.(G)
    # # Create plot points
    # edges = []
    # for i in 1:2
    #     for edge in Graphs.edges(G[i])
    #         push!(edges, [pos_x[i][Graphs.dst(edge)],pos_y[i][Graphs.dst(edge)], i])
    #     end
    # end
    # edges = reduce(hcat, edges)'
    # lines!(ax_network[1], edges[edges[:,3] .== 1, 1], edges[edges[:,3] .== 1, 2], color=(:black, .1))
    # lines!(ax_network[2], edges[edges[:,3] .== 2, 1], edges[edges[:,3] .== 2, 2], color=(:black, .1))
    # scatter!(ax_network[1], pos_x[1], pos_y[1]; markersize = 10, color = @lift($networkData[1]))
    # scatter!(ax_network[2], pos_x[2], pos_y[2]; markersize = 10, color = @lift($networkData[2]))

    # ## PLOTTING WEIGHTS
    # weights_layout = subLayout[1:2,5]
    # ax_weights = [Axis(weights_layout[i,1]; backgroundcolor = :lightgrey) for i in 1:2]
    # weightsData = @lift begin
    #     wmats = $(abmobs.adf).wmat[end-1:end]
    #     wmats_2 = [reduce(vcat, wmat) for wmat in wmats]
    #     wmats_3 = [wmat[wmat .!= 0] for wmat in wmats_2]
    # end
    # hist!(ax_weights[1], @lift($weightsData[1]))
    # hist!(ax_weights[2], @lift($weightsData[2]))

    ## PLOTTING TIMESTEP
    text_layout = fig[2, end-1]
    ax_text = Axis(text_layout[1,1])
    xlims!(ax_text, (-1,1))
    ylims!(ax_text, (-1,1))
    step_ = @lift begin
        a = $(abmobs.mdf).step[end]
        string(a)
    end
    text!(ax_text, step_, position=(0,0),align = (:center, :center),fontsize=100)

    ## PLOTTING COLLISIONS
    collision_layout = fig[2,end]
    ax_collisions = Axis(collision_layout[1,1])
    collisions = @lift begin
        a = $(abmobs.adf)
        ids = a.id
        colors = [id == 1 ? :grey : :black for id in ids]
        step = a.step .+ 1
        collisions = a.collisions

        #b = reduce(hcat,[ids, step, collisions])
        b = DataFrame(id = ids, colors = colors, step = step, collisions = collisions)
        c = unique(b)

        greyData = c[c.id .==1,:]
        blackData = c[c.id .==2,:]
        return greyData, blackData #unique(b,dims=1)
    end
    scatterlines!(ax_collisions, @lift($collisions[1].collisions), color = :gray50, markersize = 5, markeralpha = .01)
    scatterlines!(ax_collisions, @lift($collisions[2].collisions), color = :black, markersize = 5, markeralpha = .01)

    on(abmobs.model) do m
        #[autolimits!(ax_sensors[i]) for i in 1:2]
        #[autolimits!(ax_weights[i]) for i in 1:2]
        [autolimits!(ax_summary[i]) for i in 1:2]
        autolimits!(ax_collisions)
    end
    ## interactive fig
    fig
end
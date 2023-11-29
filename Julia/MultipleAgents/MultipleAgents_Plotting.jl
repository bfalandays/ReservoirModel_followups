function interactive_plot(model)
    #### plotting
    adata = [:pos, :speed, :heading, :tStep]
    mdata = [:n]

    fig, mainAx, abmobs = Agents.abmplot(model; 
        dummystep, model_step!,
        add_controls = true, enable_inspection = true,
        adata, mdata, figure = (; resolution = (1400,600))
    )

    ## PLOTTING AGENTS IN SPACE
    # get data for plotting as observables
    positionData = @lift begin
        # the output has a row for every agent, and each row has a circle for plotting the body, and the positions of the left and right sensor, respectively, for plotting as points
        # a = $(abmobs.adf)
        # # filter the data to only the latest step
        # b = $(a)[a.step .== maximum(a.step), :]

        positions = $(abmobs.adf).pos[end-(model.n_agents-1):end]
        headings = $(abmobs.adf).heading[end-(model.n_agents-1):end]

        # positions = a.pos
        # headings = a.heading
        get_leftSensPos(pos, heading) = (pos[1] + (model.agent_radius * cos(heading + deg2rad(model.wall_sens_angles[1]))), pos[2] + (model.agent_radius * sin(heading + deg2rad(model.wall_sens_angles[1]))))
        get_rightSensPos(pos, heading) = (pos[1] + (model.agent_radius * cos(heading - deg2rad(model.wall_sens_angles[1]))), pos[2] + (model.agent_radius * sin(heading - deg2rad(model.wall_sens_angles[1]))))
        a = [[gb.Circle(GLMakie.Point2f(positions[i]), model.agent_radius), get_leftSensPos(positions[i], headings[i]), get_rightSensPos(positions[i], headings[i])] for i in 1:nagents(model)]
        b = reduce(hcat, a)
    end

    step_ = @lift begin
        a = round($(abmobs.mdf).step[end] * model.dt, digits = 1)
        string(a)
    end
    text!(mainAx, step_, position=(model.space.extent[1]/2,model.space.extent[2]/2),align = (:center, :center),fontsize=20)

    poly!(mainAx, @lift([$positionData[1,i] for i in 1:nagents(model)]), color = :black) 
    scatter!(mainAx, @lift([$positionData[2,i] for i in 1:nagents(model)]) ; markersize = 10 * 15/model.space.extent[1], color = :red) 
    scatter!(mainAx, @lift([$positionData[3,i] for i in 1:nagents(model)]) ; markersize = 10 * 15/model.space.extent[1], color = :blue) 

    headings_layout = fig[1,end+1] = GridLayout()
    speeds_layout = fig[1,end+1] = GridLayout()

    headings = @lift(gb.Point2f.($(abmobs.adf).tStep * model.dt, rad2deg.(rem2pi.($(abmobs.adf).heading .- π/2, RoundNearest)) ))
    speeds = @lift(gb.Point2f.($(abmobs.adf).tStep * model.dt, $(abmobs.adf).speed ))

    ids = @lift($(abmobs.adf).id)

    # create an axis to plot into and style it to our liking
    ax_headings = Axis(headings_layout[1,1];backgroundcolor = :lightgrey, ylabel = "Headings")
    ylims!(ax_headings, (-180,180))

    ax_speeds = Axis(speeds_layout[1,1];backgroundcolor = :lightgrey, ylabel = "Speeds")
    ylims!(ax_speeds, (0,1))

    scatter!(ax_headings, headings; markersize = 5, color = ids, label = "Headings")
    scatter!(ax_speeds, speeds; markersize = 5, color = ids, label = "Speeds")


    on(abmobs.model) do m
        autolimits!(ax_headings)
        ylims!(ax_headings, (-180,180))
        
        autolimits!(ax_speeds)
        #ylims!(ax_speeds, (0,1))
    end
    
    fig
end
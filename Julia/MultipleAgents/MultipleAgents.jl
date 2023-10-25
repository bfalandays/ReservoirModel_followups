# TO DO: 
# - make a center-surround effect whereby sensory nodes activate their neighbors one over and inhibit their neighbors 2 over, in order to make the other agent pop out from the background.
# - make elastic collisions better
# - try 4 or more neighbors
# - try going back to L/R sensor arrays, then padding each with 0s to make them the same length as the total unique angles, then averaging at each position. This might diminish the activation at the edges, and emphasize the region that is covered by both eyes. Then maybe kernal or mexican hat on this?
# - try separating the two wall sensors and an array of sensors that only respond to the other agent(s)
# - try 3 effector nodes for left, right, and forward, so that speed can be controlled independently of turning

using SimpleWeightedGraphs, Graphs
import GraphPlot
include("./Julia/MultipleAgents/MultipleAgents_Functions.jl");

params = Dict(
    :n => 0,
    :nnodes => 500,
    :p_link => .1,
    :leak => .25,
    :leaktype => 1,
    :lrate_wmat => .10,# 1.0,
    :lrate_targ => .01, #.01
    :targ_min => 1.0,
    :wheel_radius => .25,
    :input_amp => 10,
    :wall_sens_angles => [-45,45],
    :sens_angles => vcat(sort(collect(range(start=-90,stop=90,step=4)),rev=true)),#vcat(sort(collect(range(start=-60,stop=60,step=4)).+30,rev=true), sort(collect(range(start=-60,stop=60,step=4)).-30,rev=true)),#vcat(sort(collect(range(start=-90,stop=90,step=1)),rev=true)),
    :sens_anglesL => vcat(sort(collect(range(start=-90,stop=90,step=4)),rev=true))[1:23],#vcat(sort(collect(range(start=-60,stop=60,step=4)).+30,rev=true)),
    :sens_anglesR => vcat(sort(collect(range(start=-90,stop=90,step=4)),rev=true))[24:end],#vcat(sort(collect(range(start=-60,stop=60,step=4)).-30,rev=true)),
    :learn_on => 1,
    :noise => 0.0,
    :agent_radius => .5,
    :acts_neg => 1,
    :visual_coupling => 1,
    :physical_coupling => 1
    )

####

model = ABM(Agent, 
ContinuousSpace((15,15), 
periodic = false); 
properties = params,
scheduler = Schedulers.fastest
)

box_faces = [
        Segment((0.0,0.0),(model.space.extent[1], 0.0)),
        Segment((model.space.extent[1],0.0),(model.space.extent[1], model.space.extent[1])),
        Segment((model.space.extent[1],model.space.extent[1]),(0.0, model.space.extent[1])),
        Segment((0.0,model.space.extent[1]),(0.0, 0.0)),
    ]
maxDist = sqrt(model.space.extent[1]^2 + model.space.extent[1]^2)
if string(model.space)[1:8] != "periodic"
    maxDist /= 2
end

n_agents = 3
locs=[]
for i in 1:n_agents
    newPos = (rand(Uniform(model.agent_radius + .2 ,model.space.extent[1] - model.agent_radius - .2)), rand(Uniform(model.agent_radius + .2,model.space.extent[2] - model.agent_radius - .2)))
    if i == 1
        push!(locs, newPos)
    else
        minDist = minimum([sqrt((newPos[1] - locs[j][1])^2 + (newPos[2] - locs[j][2])^2) for j in 1:length(locs)])
        while minDist < model.agent_radius + .2
            newPos = (rand(Uniform(model.agent_radius + .2 ,model.space.extent[1] - model.agent_radius - .2)), rand(Uniform(model.agent_radius + .2,model.space.extent[2] - model.agent_radius - .2)))
            minDist = minimum([sqrt((newPos[1] - locs[j][1])^2 + (newPos[2] - locs[j][2])^2) for j in 1:length(locs)])
        end
        push!(locs, newPos)
    end
    add_agent_pos!(
        init_agent(
            model,
            nextid(model),
            locs[i],
            rand(Uniform(0, 2pi))
        ),
        model
    )
end

#### plotting
adata = [:pos, :heading, :inputs, :acts, :targets, :spikes, :inhibitory_nodes, :output_acts, :wmat, :collisions]
mdata = [:n, :nnodes, :p_link, :leak, :leaktype, :lrate_wmat, :lrate_targ, :targ_min, :wheel_radius, :input_amp, :sens_angles, :learn_on]

fig, mainAx, abmobs = Agents.abmplot(model; 
    dummystep, model_step!,
    add_controls = true, enable_inspection = true,
    adata, mdata, figure = (; resolution = (600,600))
)

## PLOTTING AGENTS IN SPACE
# get data for plotting as observables
positionData = @lift begin
    # the output has a row for every agent, and each row has a circle for plotting the body, and the positions of the left and right sensor, respectively, for plotting as points
    # a = $(abmobs.adf)
    # # filter the data to only the latest step
    # b = $(a)[a.step .== maximum(a.step), :]

    positions = $(abmobs.adf).pos[end-(n_agents-1):end]
    headings = $(abmobs.adf).heading[end-(n_agents-1):end]

    # positions = a.pos
    # headings = a.heading
    get_leftSensPos(pos, heading) = (pos[1] + (model.agent_radius * cos(heading + deg2rad(model.wall_sens_angles[1]))), pos[2] + (model.agent_radius * sin(heading + deg2rad(model.wall_sens_angles[1]))))
    get_rightSensPos(pos, heading) = (pos[1] + (model.agent_radius * cos(heading - deg2rad(model.wall_sens_angles[1]))), pos[2] + (model.agent_radius * sin(heading - deg2rad(model.wall_sens_angles[1]))))
    a = [[gb.Circle(GLMakie.Point2f(positions[i]), model.agent_radius), get_leftSensPos(positions[i], headings[i]), get_rightSensPos(positions[i], headings[i])] for i in 1:nagents(model)]
    b = reduce(hcat, a)
end

[poly!(mainAx, @lift($positionData[1,i]), color = :black) for i in 1:nagents(model)]
[scatter!(mainAx, @lift($positionData[2,i]) ; markersize = 10, color = :red) for i in 1:nagents(model)]
[scatter!(mainAx, @lift($positionData[3,i]) ; markersize = 10, color = :blue) for i in 1:nagents(model)]

# scatterlines!(ax, positions; color = :black, markersize = 5, markeralpha = .01)


## interactive fig
fig

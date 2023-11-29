# TO DO: 
# - in the VENstyle model, try giving agents another effector for slowing down.
# - check on how I'm implementing spatiality again
# - figure out why activation values don't decay even when agents are only getting minimal input from noise. 

# NOTES:

using SimpleWeightedGraphs, Graphs
import GraphPlot
include("./Julia/MultipleAgents/MultipleAgents_Functions.jl");

seed = 123
set.seed(seed)

props = Dict(
    :n => 0,

    :agent_radius => .5,
    :wheel_radius => .25,
    :sens_angles => [vcat(sort(collect(range(start=-60,stop=60,step=4)).+30,rev=true)), vcat(sort(collect(range(start=-60,stop=60,step=4)).-30,rev=true))],
    #:sens_angles => [vcat(sort(collect(range(start=-90,stop=90,step=4)).+45,rev=true)), vcat(sort(collect(range(start=-90,stop=90,step=4)).-45,rev=true))],
    #:sens_angles => [vcat(sort(collect(range(start=-20,stop=20,step=.5)),rev=true))],
    :wall_sens_angles => [-45,45],

    :nnodes => 250,
    :num_effectors => 2, # 2 for left/right, 3 for left/right/forward
    :p_link => .1,
    :lrate_wmat => .10,# .10,
    :lrate_targ => .01, #.01
    :targ_min => 1.0,
    :learn_on => 1,
    :acts_neg => 1,
    :leak => .25,
    :leaktype => 1,

    :input_type => "random", # "random" or "spatial"
    :res_type => "smallworld", # "random", "spatial", "smallworld" or "scalefree"
    :output_type => "random", # "random" or "spatial"

    :visual_coupling => 1,
    :physical_coupling => 1,
    :collision_heading_change => 1,
    :sens_agent_dist => 0,
    :sens_walls => 1,
    :sensory_scaling => true,
    :network_noise => 0.0,
    :sensory_noise => 0.1,
    
    :VENstyle => true,
    :topSpeed => .2, #2.0,
    :accelTime => 5, #2.0,
    :topHeadingRate => π/8,
    :HaccelTime => 5, #2.0,
    
    :dt => 1,

    :periodic => true,
    :space_size => 15,
    :n_agents => 5,

    )
    if props[:sens_walls] == 1 && props[:periodic] == false && props[:visual_coupling] == 1
        n_inputs = length(reduce(vcat,props[:sens_angles])) + 2
    elseif props[:sens_walls] == 0
        n_inputs = length(reduce(vcat,props[:sens_angles]))
    else
        n_inputs = 2
    end
    props[:input_amp] = props[:nnodes]*props[:p_link]/2 #* n_inputs/64

####

model = ABM(Agent, 
ContinuousSpace((props[:space_size],props[:space_size]), 
periodic = props[:periodic]); 
properties = props,
rng = rng = Random.MersenneTwister(seed),
scheduler = Schedulers.fastest
)

box_faces = [
        Segment((0.0,0.0),(model.space.extent[1], 0.0)),
        Segment((model.space.extent[1],0.0),(model.space.extent[1], model.space.extent[1])),
        Segment((model.space.extent[1],model.space.extent[1]),(0.0, model.space.extent[1])),
        Segment((0.0,model.space.extent[1]),(0.0, 0.0)),
    ]
maxDist = sqrt(model.space.extent[1]^2 + model.space.extent[1]^2)
if model.periodic == true
    maxDist /= 2
end

locs=[]
for i in 1:model.n_agents
    newPos = (rand(abmrng(model),Uniform(model.agent_radius + .2 ,model.space.extent[1] - model.agent_radius - .2)), rand(abmrng(model),Uniform(model.agent_radius + .2,model.space.extent[2] - model.agent_radius - .2)))
    if i == 1
        push!(locs, newPos)
    else
        minDist = minimum([sqrt((newPos[1] - locs[j][1])^2 + (newPos[2] - locs[j][2])^2) for j in 1:length(locs)])
        while minDist < 2*model.agent_radius + .2
            newPos = (rand(abmrng(model),Uniform(model.agent_radius + .2 ,model.space.extent[1] - model.agent_radius - .2)), rand(abmrng(model),Uniform(model.agent_radius + .2,model.space.extent[2] - model.agent_radius - .2)))
            minDist = minimum([sqrt((newPos[1] - locs[j][1])^2 + (newPos[2] - locs[j][2])^2) for j in 1:length(locs)])
        end
        push!(locs, newPos)
    end
    add_agent_pos!(
        init_agent(
            model,
            nextid(model),
            locs[i],
            rand(abmrng(model),Uniform(0, 2pi))
        ),
        model
    )
end

if model.n_agents == 2
    include("./Julia/MultipleAgents/TwoAgents_Plotting.jl");
else
    include("./Julia/MultipleAgents/MultipleAgents_Plotting.jl");
end

interactive_plot(model)

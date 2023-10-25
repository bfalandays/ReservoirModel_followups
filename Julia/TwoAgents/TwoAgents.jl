# TO DO: 
# - make a center-surround effect whereby sensory nodes activate their neighbors one over and inhibit their neighbors 2 over, in order to make the other agent pop out from the background.
# - make elastic collisions better
# - try 4 or more neighbors
# - try going back to L/R sensor arrays, then padding each with 0s to make them the same length as the total unique angles, then averaging at each position. This might diminish the activation at the edges, and emphasize the region that is covered by both eyes. Then maybe kernal or mexican hat on this?
# - try separating the two wall sensors and an array of sensors that only respond to the other agent(s)
# - try 3 effector nodes for left, right, and forward, so that speed can be controlled independently of turning

using SimpleWeightedGraphs, Graphs
import GraphPlot
include("./Julia/TwoAgents/TwoAgents_Functions.jl");

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
    :noise => 0.1,
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

add_agent_pos!(
    init_agent(
        model,
        nextid(model),
        (10,5),#(11.25,3.75),
        deg2rad(135)#pi#3π/4 + deg2rad(30)
    ),
    model
)

add_agent_pos!(
    init_agent(
        model,
        nextid(model),
        (5,10),#(3.75,11.25),
        -π/4 #+ deg2rad(30)
    ),
    model
)

include("./Julia/TwoAgents/TwoAgents_Plotting.jl");

interactive_plot(model)




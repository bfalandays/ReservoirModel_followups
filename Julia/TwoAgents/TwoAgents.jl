# TO DO: 
# - make a plot for headings and speeds
# - try again to have outputs control acceleration and headingRate, maybe by adding some friction and only letting accel increase? maybe momentum is needed too?
# - check on how I'm implementing spatiality again
# - figure out why activation values don't decay even when agents are only getting minimal input from noise. 
# - try scaling the total input at each moment so that the sum is always the same, but the distribution changes. 
#       And then try setting the input amp as a constant ratio of the network size? 
#       Or just a constant for all network sizes, since the number of input->reservoir connections scales with network size?
# - sensory scaling seems to be making the detection of other agents too small relative to the walls. Need to figure out why that is.

# TO DO: 
# - try doing a more VENlab-esque movement model, where effectors control change in speed and change in direction, rather than speed and direction directly. 
#   - best way to do this is probably by making values below .5 decrease, and above .5 increase
# - make visual_coupling an agent-level variable, so that I can shut it off for only one of the two
# - try to add more weight to wall sensors. Maybe just add more of them?

using SimpleWeightedGraphs, Graphs
import GraphPlot
include("./Julia/TwoAgents/TwoAgents_Functions.jl");

seed = 1234
set.seed(seed)

props = Dict(
    :n => 0,

    :agent_radius => .5,
    :wheel_radius => .25,
    :sens_angles => [vcat(sort(collect(range(start=-60,stop=60,step=4)).+30,rev=true)), vcat(sort(collect(range(start=-60,stop=60,step=4)).-30,rev=true))],
    #:sens_angles => [vcat(sort(collect(range(start=-20,stop=20,step=1)),rev=true))],
    #:sens_angles => [vcat(sort(collect(range(start=-90,stop=90,step=2)),rev=true))],

    :wall_sens_angles => [-45,45],

    :nnodes => 200,
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
    :res_type => "random", # "random", "spatial", "smallworld" or "scalefree"
    :output_type => "random", # "random" or "spatial"

    :visual_coupling => 1,
    :physical_coupling => 1,
    :collision_heading_change => 1,
    :sens_agent_dist => 0,
    :sens_walls => 1,
    :sensory_scaling => true,
    :network_noise => 0.0,
    :sensory_noise => 0.1,
    
    :periodic => false,

    :VENstyle => false,
    :topSpeed => 2.0,
    :accelTime => 1.0,
    :topHeadingRate => π,
    :HaccelTime => 1.0,

    :dt => 1,

    # :friction => .1,
    # :max_accel => .1,
    # :Hfriction => .1/(pi/4),
    # :max_Haccel => .1,

    )
    if props[:sens_walls] == 1 && props[:periodic] == false && props[:visual_coupling] == 1
        n_inputs = length(reduce(vcat,props[:sens_angles])) + 2
    elseif props[:sens_walls] == 0
        n_inputs = length(reduce(vcat,props[:sens_angles]))
    else
        n_inputs = 2
    end
    props[:input_amp] = props[:nnodes]*props[:p_link] #* n_inputs/64



####

model = ABM(Agent, 
ContinuousSpace((15,15), 
periodic = props[:periodic]); 
properties = props,
rng = Random.MersenneTwister(seed),
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

add_agent_pos!(
    init_agent(
        model,
        nextid(model),
        (5,2.5),#(11.25,3.75),
        deg2rad(135)#pi#3π/4 + deg2rad(30)
    ),
    model
)

add_agent_pos!(
    init_agent(
        model,
        nextid(model),
        (2.5,5),#(3.75,11.25),
        -π/4 #+ deg2rad(30)
    ),
    model
)

include("./Julia/TwoAgents/TwoAgents_Plotting.jl");

interactive_plot(model)


# using Plots
# Plots.histogram(rand(Exponential(model.nnodes/4),1000))
# Plots.histogram(vec(sum(model[1].link_mat, dims=1)))
# Plots.histogram(edge_lengths)

# test = barabasi_albert(model.nnodes, 2,2, is_directed=true)
# test2 = degree_histogram(test)
# Plots.histogram(test2)

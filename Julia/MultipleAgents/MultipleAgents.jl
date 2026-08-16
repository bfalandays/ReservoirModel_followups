# TO DO: 
# - in the VENstyle model, try giving agents another effector for slowing down.
# - check on how I'm implementing spatiality again
# - figure out why activation values don't decay even when agents are only getting minimal input from noise. 

#= NOTES: (8/15/26)
    The "VENstyle" model is based on how Bill Warren's lab (the VEN lab) models visual control of locomotion. 
    I'm not sure if I ever landed on super reliable settings for that version, so I'd consider it "experimental",
    but it does seem to produce coordinated motion sometimes. However, the published version of the Braitenberg model 
    (:num_effectors = 2, :VENstyle = false) is simpler and more robustly produces coordinated motion, so that's 
    probably the best starting point.
    
    But I'm just going to describe some of the idea behind the VENstyle version, because it could be cool if that worked eventually:
    The VENstyle model forces there to be 3 output effectors (overriding the num_effectors setting). 
    One output controls acceleration in the heading direction. The difference between the other two outputs
    controls angular acceleration. Both fight against friction, which is in turn a function of a friction coefficient
    and the current linear/angular speed.
    
    I don't think the current version totally makes sense, because I have "topSpeed" and "topHeadingRate"
    parameters in addition to friction, which seems redundant. I have a vague memory now, but I think I was
    aiming to eliminate the topSpeed and topHeadingRate params, because they're kinda arbitrary, and I wanted to see if adding
    friction could prevent the agents from potentially going to insane speeds. 

    (side note: I also messed around a version that was almost the same as the published Braitenberg-style agent, but which
    used 3 effectors instead of 2. In the original 2-effector version, each output controls the speed of one wheel. Differences 
    in speed between the two wheels causes rotation, and the average of the two wheel speeds determines forward speed.
    In the 3-effector version, one output controls forward speed and the other two outputs do rotation. The 3-effector
    version also works seems to work fine.)
=#

#= Notes on some other params that probably need explaining:
    - :visual_coupling determines whether agents sense each other (1) or only see the walls, but not other agents (0)
    - :physical_coupling determines whether agents collide with each other (1) the same way as they do with walls, or call pass through each other (0)
    - :collision_heading_change determines whether, if agents collide, their headings get mirrored in the same way as when they hit a wall (1), or whether their heading is unaffected by agent-to-agent collisions (0)
    - :sens_agent_dist determines whether a sensory nodes are sensitive to distance (1), meaning it is less-strongly activated the further away the sensed-object (another agent or the wall) is, or whether the node just outputs a binary response when another that node's ray intersects with anything (0). I don't think the distance version makes sense, I think I was just messing around 
    - :sensory_scaling normalizes the sensory nodes' activity (1) or applies no transformation (0). I think the rationale here was to work in conjunction with :sens_agent_dist = 1, because when distance matters, sometimes there was very little sensory input and agents would freeze.
    - :sensory_noise and :network_noise add noise to the sensory nodes or the reservoir nodes, respectively. Noise also helps prevent freezing, but sensory noise seems to be enough.
=#

#= Couple other miscellaneous notes:
    - This model actually has 2 different kinds of sensor arrays active simultaneously. In the paper,
      I used different sensor styles for the object-tracking model and the braitenberg model. 
      For the object-tracking agent, each "eye" was an array of sensors pointing out at different angles, 
      like a simple retina. For the braitenberg agent, each eye was a single sensor.
      It seems like I just mashed them together here, using the object-tracking-style array for sensing other agents,
      and the 2-sensor braitenberg-style array for sensing walls. That seems kinda weird, but there are issues if you
      just use the object-tracking style array, because the sensors' rays are constantly intersecting with a wall, so if 
      an agent passes in front (and the sensor output is binary), then there's no change in the sensor. So basically, other
      agents become mostly invisible to each other. To deal with that, you have to make sensors distance-sensitive, such that
      when agent A passes between agent B and the wall, the sensor becomes more strongly activated. But of course, the distance
      sensitivity thing adds further complications. So just wanted to explain that idiosyncracy of this model.
    
    - I never tested this systematically, but from visual inspection, I remember thinking that using a small-world
      network structure for the reservoir resulted in coordinated motion more consistently and more quickly. And from running it a 
      few times now, it's still looking that way to me. So that could be something interesting to explore!
=#

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
    :num_effectors => 3, # 2 for left/right, 3 for left/right/forward
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
    
    :VENstyle => false, # the params below in this block only matter if :VENstyle = true
    :topSpeed => .2, #2.0,
    :accelTime => 5, #2.0,
    :topHeadingRate => π/8,
    :HaccelTime => 5, #2.0,
    
    :dt => 1,

    :periodic => false, # when periodic is true, the space is a torus and there are no walls to bounce off
    :space_size => 15,
    :n_agents => 3,

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

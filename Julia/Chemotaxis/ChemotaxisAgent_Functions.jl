using Agents, LinearAlgebra, InteractiveDynamics, GLMakie, Statistics, Meshes, Distributions, DataFrames, StatsBase, SimpleWeightedGraphs, Graphs, PDMats
import GeometryBasics as gb
import GraphPlot

mutable struct Agent <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    type::Symbol # :agent, :stim
    complexPos::ComplexF64
    speed::Float64
    heading::Float64
    sens_angles::Vector{Float64}
    sens_Cpos::NTuple{2, ComplexF64}
    degree::Int64
    tStep::Int64
    prev_inputs::Vector{NTuple{2,Float64}}
end

global learn_on = 1 #learning is on by default
global acts_neg = 1 #whether or not activations can be < 0. true by default -- activations can take on any value
global agent_radius = .5 # the radius of the agent

global ts_pos = [] #vector of agent positions over time for analysis
global ts_spikes = [] # array of spikes over time for analysis. Will be timesteps (rows) X nnodes (cols)
global ts_acts = [] #array of reservoir activations over time
global ts_heading = [] #array of agent headings over time
global ts_sens = [] #array of sensor activations over time
global ts_eff = [] #array of effector activations over time

#create a shortcut for initializing the agent. On the right of the "=", we create the full struct with all 10 variables, supplying default values for ones that don't change. Variables that we do change, we give a variable name (e.g. "pos" rather than supply a position)
# on the left side of the "=", we give a name to the shortcut ("_Agent") and give it just the variables that we will input every time we create an agent -- id, position, heading, and the angles of the sensors
_Agent(id, pos, heading, sens_angles) = Agent(id, pos, :agent, pos[1] + pos[2]im, 0, heading, sens_angles, ( agent_radius * ( cos(heading+deg2rad(sens_angles[1])) + sin(heading+deg2rad(sens_angles[1]))im), agent_radius * ( cos(heading-deg2rad(sens_angles[1])) + sin(heading-deg2rad(sens_angles[1]))im)), 0, 0, [(0.0,0.0)])

### functions
global xmean = 25.0
global ymean = 25.0
global spread = 15.0
#defining the chemical gradient
function gradient(x,y)
    exp(-( (x-xmean)^2 + (y-ymean)^2 )/(2*spread^2))
end

function dx_gradient(x,y)
    -((x-xmean)*exp((-(x-xmean)^2 - (y-ymean)^2) /(2*spread^2)))/(spread^2)
end

function dy_gradient(x,y)
    -((y-ymean)*exp((-(x-xmean)^2 - (y-ymean)^2) /(2*spread^2)))/(spread^2)
end

function move_agent(output_acts, agent,model)

    # translation
    vel_centre = (output_acts[1]+output_acts[2])/2 #get the mean of the two outputs, which gives the center of velocity vector
    pos_centre = vel_centre .* [cos(agent.heading), sin(agent.heading)] .* movement_amp # multiply the center of velocity by the cos and sin of heading to get the updated x- and y- position
    
    # rotation
    omega = (output_acts[2]-output_acts[1])/(2*agent_radius) #rotation is determined by the relative speed of the two wheels and the radius of the agent, which gives the distance of the wheels from the center
    agent.heading += omega * movement_amp

    #move and rotate the agent -- agents.jl takes care of the torus situation
    move_agent!(agent, (agent.pos[1] + pos_centre[1], agent.pos[2] + pos_centre[2]),model)

end


function get_input_acts(agent)

    # get position and heading of agent
    pos = agent.pos
    heading = agent.heading

    #take the angles of the sensors and the radius of the agent to find out the position of the sensors
    sL_pos = (agent_radius * cos(heading+deg2rad(agent.sens_angles[1])) + pos[1], agent_radius * sin(heading+deg2rad(agent.sens_angles[1])) + pos[2])
    sR_pos = (agent_radius * cos(heading-deg2rad(agent.sens_angles[1])) + pos[1], agent_radius * sin(heading-deg2rad(agent.sens_angles[1])) + pos[2])

    #get the concentration of the gradient function (a 2D gaussian function) at the location of each sensor
    sL_p = gradient(sL_pos[1],sL_pos[2])
    sL_dx = dx_gradient(sL_pos[1],sL_pos[2])
    sL_dy = dy_gradient(sL_pos[1],sL_pos[2]) 
    sL_d = cos(heading)*sL_dx + sin(heading)*sL_dy

    sR_p = gradient(sR_pos[1],sR_pos[2])
    sR_dx = dx_gradient(sR_pos[1],sR_pos[2])
    sR_dy = dy_gradient(sR_pos[1],sR_pos[2])
    sR_d = cos(heading)*sR_dx + sin(heading)*sR_dy

    # #get the activations
    # if noise > 0 #if there is some noise in the activation function...
    #     sL_act = sL_p + rand(Uniform(-noise,noise))
    #     sR_act = sR_p + rand(Uniform(-noise,noise))
    # else
    #     #if we aren't adding noise, the sensor activation is just the normalized probability density of the gradient function at the position of the sensor
    #     sL_act = sL_p
    #     sR_act = sR_p
    # end
    sL_act = sL_d*20 #sL_p +  #sL_d #*10 + sL_p
    sR_act = sR_d*20 #sR_p +  #*10 + sR_p

    #put the left and right sensor activations into an array
    sens_acts = [sL_act, sR_act]
        
    #output
    sens_acts
end

function get_acts(acts,leak,spikes,wmat,input,input_wmat,targets)
    
            # leak                 # sum input from sensors          # sum input from spikes in the reservoir
    acts = (acts .* (1-leak) .+ sum(input .* input_wmat, dims = 1)' .+ sum(spikes[] .* wmat[], dims = 1)')[:,1] #.+ rand(Normal(0,nodeNoise),nnodes)

    #set a floor of activation at 0 if this is turned on (it's off by default)
    if acts_neg == 0
        acts[acts .< 0] .= 0
    end
    
    #get the list of current spikes
    prev_spikes = copy(spikes[])

    #get the list of current spiking thresholds -- just double the current target value of the node
    thresholds = targets .* 2

    #if the current activation of a node is above the threshold, it spikes and we set the corresponding value of the spike array to 1
    spikes[][acts .>= thresholds] .= 1
    #for nodes that didn't spike, we set the corresponding value of the spike array to 0
    spikes[][acts .< thresholds] .= 0
    
    #if the node spikes, then we subtract the threshold value of that node from the current activation -- energy dissipation
    acts[spikes[] .== 1] .-= thresholds[spikes[] .== 1]
    
    #calculate the error for each node (current activation minus target value)
    errors = acts .- targets
    
    # output
    [acts, spikes[], errors, prev_spikes]
end


function learning(learn_on,link_mat,spikes,prev_spikes, errors,wmat,targets)
    
    active_neighbors=abs.(link_mat) #start with the link matrix (anatomical connectivity) so that nodes can't attribute error to any non-neighbors (meaning other nodes that aren't connected to the focal node)
    active_neighbors[prev_spikes .== 0,:] .= 0 # for any node that didn't spike, we set all the outgoing connections in that row to 0
    d_wmat = copy(active_neighbors) # set up a matrix of weight changes. "d" is for "delta" i.e. this will become a matrix of changes to the current weight matrix (of the same size as the weight matrix)
    active_neighbors = sum(active_neighbors,dims=1) # for every node, get the number of incoming spikes (how many of my neighbors were active). dims=1 means we sum along columns
    
    if learn_on==1 # assuming learning is turned on
        if sum(active_neighbors) >0 # and there was at least one spike
            d_wmat = (errors' .* d_wmat) # multiply errors by the matrix of weight changes. 
            #The result is that every column (incoming connections to a node) will contain a 0 for rows corresponding to non-neighbors OR neighbors that didn't spike on the last iteration
            #or the column will contain the error value of the current node for rows corresponding to neighbors that did spike. 
            #in other words, the TOTAL error of each node will be assigned to every incoming connection along which a spike was propagated

            d_wmat=(d_wmat ./ active_neighbors).*lrate_wmat # normalize the error for each node -- distributes error evenly across all incoming connections
            #in the last step, the total error for each node was represented multiple times (e.g. if 10 neighbors spiked, 10 rows contained the total error)
            #so here we need to divide each row by the number of neighbors that spiked, so that the sum of the column will be the total error for the focal node

            d_wmat[isinf.(d_wmat)] .= 0 #deal with infinites and nans, since we may be dividing by zeros
            d_wmat[isnan.(d_wmat)] .= 0 

            wmat[] .-= d_wmat # update the weight matrix by subtracting d_wmat (change in weight matrix)
            #this means that if error was positive (activation was ABOVE target), nodes try to lower the weights
            #if activation was negative, nodes will try to increase weights
        end
        
        # for updating targets, take the error for each node and multiply by a learning rate
        #targets are updated in the opposite direction from weights:
        #if error was positive (activation > target), nodes increase the target
        #if error was negative, nodes decrease the target
        targets .+= (errors .* lrate_targ) 

        #in the last step, some targets may have gone below the minimum target value
        #the assumption built in here is that all neurons need SOME positive input to survive -- they can't "prefer" to have 0 or negative input
        #so we check for any targets that have gone below the floor, and reset that value to the floor.
        targets[targets .< targ_min] .= targ_min
    end
          
    #output
    [wmat[], targets]
end


global prev_input = [0.0,0.0]
global input_diff = [0.0,0.0]

## the main stepping function that advances the model, calling on the functions defined above
function model_step!(model)

    #declare variables as global so they are available wherever we need them
    global acts
    global spikes
    global targets
    global errors
    global prev_spikes
    global wmat
    global input_wmat
    global direction
    global input
    global diffs

    #loop through all agents in the model. allagents() is an agents.jl function. Here there is only one agent so this doesn't matter much
    for agent in allagents(model)
        
        #increment the timestep
        agent.tStep += 1

        input = get_input_acts(agent) #.* 20
        input = [x <= 0.0 ? rand(Uniform(0,noise)) : x for x in input]

        # if agent.tStep == 1
        #     agent.prev_inputs[1] = Tuple(get_input_acts(agent))
        #     output_acts = [rand(Uniform(0,.2)), rand(Uniform(0,.2))]
        #     move_agent(output_acts, agent,model)
        # end

        # #get the next sensor input
        # if length(agent.prev_inputs) < 20
        #     pushfirst!(agent.prev_inputs, Tuple(get_input_acts(agent)))
        # else
        #     pop!(agent.prev_inputs)
        #     pushfirst!(agent.prev_inputs, Tuple(get_input_acts(agent)))
        # end
        # input = collect(agent.prev_inputs[1])

        # for i in 1:(length(agent.prev_inputs)-1)
        #     diff_ = [agent.prev_inputs[i][1] - agent.prev_inputs[i+1][1];agent.prev_inputs[i][2] - agent.prev_inputs[i+1][2]]
        #     if i == 1
        #         diffs = transpose(diff_)
        #     else
        #         diffs=vcat(diffs, transpose(diff_))
        #     end
        # end
        # diffs = mean(diffs, dims=1)[1,:]
        
        # input_diff = diffs .* 1000 #.+ [rand(Normal(0,noise)),rand(Normal(0,noise))]
        # input_diff = [x <= 0.05 ? rand(Uniform(0,noise)) : x for x in input_diff]

        ##NOTE: I think the problem is that I'm calculating the derivative without accounting for distance moved, so if the agent stays still, it goes to 0
        # alternatively, we could try to directly calculate the slope of the gradient along the heading direction of the agent

        #save that input in the array that is stored within the model, for later analysis
        #current version below is giving the difference between input(t) and input(t-1) to the sensors -- a rough approximation of a derivative sensor
        #to go back to just giving the raw concentrations, just change "input_diff" to "input" below, and in the next chunk make sure we are feeding "input" to the get_acts() function, rather than "input_diff"
        model.inputs = input #input_diff #

        #for reservoir nodes, get the current activation array, current spike array, current error array, and the spike array from the previous timestep 
        acts, spikes[], errors, prev_spikes = get_acts(acts,leak,spikes,wmat,input,input_wmat,targets)
        
        #if we are trying to use the time derivative of sensor values, we feed the input_diff (difference between current and previous input) to the function below
        #acts, spikes[], errors, prev_spikes = get_acts(acts,leak,spikes,wmat,input_diff,input_wmat,targets)

        #propogate the spikes of the reservoir to the output nodes
        output_acts = (sum(spikes[] .* output_wmat,dims = 1) ./ sum(output_wmat, dims=1))[1,:]
        #the value of the output is normalized by the number of incoming connections to each output node, such that output is always in the range [0,1]
        #the main purpose of this is to deal with potential imbalances in incoming connections to effectors, given that the links are assigned probabilistically
        #e.g. one effector might have 22 incoming connections from the reservoir, while the other could have 19, and this might make arbitrary biases in movement that we want to avoid

        #save the output acts in the array stored in the model, for later analysis
        model.outputs = output_acts

        #saving data
        push!(ts_spikes, Tuple(Int64(x) for x in spikes[]))
        push!(ts_acts, Tuple(Float64(x) for x in acts))
        push!(ts_pos, agent.pos)

        push!(ts_heading, (agent.heading))
        push!(ts_sens, Tuple(Float64(x) for x in input))
        push!(ts_eff, Tuple(Float64(x) for x in output_acts))

        #perform learning on the reservoir weight matrix and targets
        wmat[], targets = learning(learn_on,link_mat,spikes,prev_spikes, errors,wmat,targets)

        # move the agent
        move_agent(output_acts, agent,model)

    end

    #store the mean activations, errors, and targets of the reservoir nodes for plotting
    model.mean_act = mean(acts)
    model.mean_err = mean(errors)
    model.mean_targ = mean(targets)

    #increment the timestep variable that is stored in the model (above we did this for a variable stored in each agent)
    #we don't really need to have both, but before I delete stuff I'll have to look at which one I'm actually using for plotting, lolz
    model.n += 1
end
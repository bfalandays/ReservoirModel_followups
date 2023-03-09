# load function file
include("./Chemotaxis/ChemotaxisAgent_Functions.jl");

## params
global nnodes = 200 #number of nodes in the reservoir
global p_link = .1 # probability of a link between input node-reservoir, reservoir-reservoir, and reservoir-output
global leak = .25 #leak rate of nodes
global lrate_wmat = 1.0 # learning rate for reservoir weight matrix
global lrate_targ = .01 # learning rate for targets
global targ_min = 1.0 # the minimum target value
global movement_amp = .5 # an arbitrary gain on output to make sure agent doesn't move too slow. E.g. if output is 1, agent moves 1*movement_amp units forward
global input_amp = 5 # an arbitrary gain on input to make sure the agent gets moving. 
#The gradient concentration is very low in the corners, so if we start the agent there, they don't get enough input to cause any spikes in the network, and the agent just stays put.
#hence the input_amp that you choose will depend upon how we set up the gradient

global learn_on = 1 # learning is turned on
global acts_neg = 1 #activations can take on negative values
global noise = 2#.005 # how much noise is added to sensor values
global nodeNoise = 0#.01 #how much noise is added to reservoir node activations

#parameters for the 2D gaussian chemical gradient
global xmean = 25.0
global ymean = 25.0
global spread = 15.0

#getting a heatmap to plot the gradient
global xs = collect(0:.1:50) #initialize a list of x values to sample from 0 to 50 (the range of the space)
global ys = collect(0:.1:50) # list of y values
global zs = zeros(length(xs), length(ys)) #matrix of z values for the heatmap -- initially zeros

#iterate through the x and y values, get the probability density of the gradient function at each point, and fill in the z matrix
for col in 1:length(xs)
    for row in 1:length(ys)
        #p = pdf(gradient, [xs[col],ys[row]])
        p = gradient(xs[col],ys[row])
        zs[row,col]=p
    end
end
#zs= zs ./ maximum(zs) # normalize the z matrix so that the max value is 1

#### 

#angle of the two sensors relative to the current heading of agent. We supply it in degrees, to make it more intuitive, but the functions convert everything to radians
sens_degrees = [45, -45]

#assign sensor colors for plotting
sensColors= [:red, :blue]

### INPUT LAYER ###
#set up an input weight matrix
global input_wmat=zeros((length(sens_degrees),nnodes))
#loop through the input matrix
for row in range(1,size(input_wmat)[1]) #rows correspond to each sensor
    for col in range(1,nnodes) #columns correspond to reservoir nodes
        input_wmat[row,col]=StatsBase.sample([0,input_amp],Weights([1-p_link,p_link]))
        #link from sensor to a node in the matrix is 0 with probability 1-p_link, or input_amp with probability p_link
    end
end

### RESERVOIR LAYER ###
# create a FIXED anatomical connectivity matrix inside the reservoir
global link_mat = zeros((nnodes,nnodes))
for row in range(1,size(link_mat)[1]) #rows correspond to OUTGOING connections from one node to all others
    for col in range(1,size(link_mat)[2]) #columns correspond to INCOMING connections to a node
        if row == col
            continue # don't allow any self-connections, so if we're on the diagonal we skip to the next iteration of the loop
        else
            link_mat[row,col] = StatsBase.sample([0,1],Weights([1-p_link,p_link]))
            #from one reservoir node to another, form no connection with probabiltiy 1-p_link, or form a connection with probability p_link
        end
    end
end
        
# initialize a VARIABLE weight matrix for the reservoir based on the link matrix
global wmat= Observable(zeros((nnodes,nnodes)))
for row in range(1,size(wmat[])[1]) 
    for col in range(1,size(wmat[])[2]) 
        if link_mat[row,col] == 1 #if there is an anatomical link...
            # if StatsBase.sample([-1,1],Weights([.25,.75])) == -1
            #     wmat[][row,col] = rand(Normal(-2.0,.1))
            # else
            #     wmat[][row,col] = rand(Normal(2.0,.1))
            # end
            wmat[][row,col] = rand(Normal(2.0,.1))
            #initialize a weight by drawing from a normal distribution with mean 1 and sd .1
        end
    end
end
startWmat = copy(wmat[]) #save the initial weight matrix just in case we want to see how it changed from start to end

### OUTPUT LAYER ###
# set up the weight matrix from reservoir nodes to effector nodes
effector_nodes=[] #initialize an array to store a list of reservoir nodes that are connected to effectors
global output_wmat=zeros((nnodes,2)) #initialize a nnodes (reservoir) X 2 (output) weight matrix, initially zeros
#loop through the output weight matrix
for row in range(1,size(output_wmat)[1])#rows correspond to reservoir nodes
    for col in range(1,size(output_wmat)[2])#columns correspond to effector nodes
        output_wmat[row,col]=StatsBase.sample([0,1],Weights([1-p_link,p_link]))
        #form a weight from reservoir node to output node with probability p_link
        if output_wmat[row,col] != 0
            push!(effector_nodes,row) #if we formed a link, store the reservoir node in the list we initialized above
        end
    end
end

#set up some dataframes for saving activity over time
global spikes = Observable(zeros(nnodes)) 
global targets = repeat([targ_min],nnodes)
global acts = zeros(nnodes)
global inputs = zeros(length(sens_degrees))
global input_diff = zeros(length(sens_degrees))
global output_acts = zeros(2)

# set up initial values for the timestep, mean activations, errors, and targets of the reservoir nodes
global n = 0
global mean_act = 0.0
global mean_err = 0.0
global mean_targ = mean(targets)

#set up the model
    #name of agent struct         #50 x 50 torus (periodic=true)      #list of properties fed to the model
model = ABM(Agent, ContinuousSpace((50,50), periodic = true); properties = Dict(
    :n => n,
    :mean_act => mean_act,
    :mean_err => mean_err,
    :mean_targ => mean_targ,
    :inputs => inputs,
    :input_diff => input_diff,
    :outputs => output_acts,
    :weights => wmat,
),
scheduler = Schedulers.fastest #scheduler for looping through agents. Since there is only one agent, this doesn't matter
)

# add an agent to the model
add_agent_pos!(
    _Agent(
        nextid(model), #give it the next available ID in the model (here it's 1 because nothing else was added first)
        (5,5), #put it at position 5,5 (note: coordinates start in bottom left)
        π/2, # initial heading is 90 degrees -- straight up. East is 0 and headings move counterclockwise
        sens_degrees, #give it the angles of the two sensors relative to the agent's heading

    ),
    model
)

#### PLOTTING ###
adata = [:pos, :heading] #these are agent-level variables that we want to store for plotting
mdata = [:mean_act, :mean_err, :mean_targ, :inputs, :outputs] # these are model-level variables (defined in the properties Dict) that we want to store for plotting

#the interactive plotting function needs a Dict of parameters. I'm not doing anything with it, so we just give it an empty Dict
params = Dict(   
)

#create a interactive plot using the abmplot function of agents.jl
fig, ax, abmobs = abmplot(model; 
    dummystep, model_step!,
    add_controls = true, enable_inspection = true,
    adata, mdata, figure = (; resolution = (2400,1600))
)

#add a layout to the plot for sensor activations
sensors_layout = fig[1,end+1] = GridLayout()

#add a layout for mean activations, targets, and errors of reservoir nodes
acts_layout = fig[1,end+1] = GridLayout()

#add a layout containing a text box for putting the timestep
text_layout = fig[2, end-1]
ax_text = Axis(text_layout[1,1])
xlims!(ax_text, (-1,1))
ylims!(ax_text, (-1,1))

#add a layout for plotting the network and spikes
network_layout = fig[3, end-2] = GridLayout()

#add a layout for plotting the effector (output) values
effectors_layout = fig[3,end-1] = GridLayout()

#add a layout for plotting the weights of the network
weights_layout = fig[3,end] = GridLayout()

#define the axis within the sensor layout
ax_sensors = Axis(sensors_layout[1,1];backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:length(sens_degrees)))
xlims!(ax_sensors, (0,length(sens_degrees)+1))
ylims!(ax_sensors, (-2,2))

#define the axis within with mean activations layout
ax_acts = Axis(acts_layout[1,1]; backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:3, ["meanActs","meanErrs", "meanTargs"]), xticklabelrotation = pi/8)
xlims!(ax_acts, (0,4))
ylims!(ax_acts, (-3,3))

#define the axis for the network layout
ax_network = Axis(network_layout[1,1])

#define the axis for the effector layout
ax_effectors = Axis(effectors_layout[1,1];backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:2, ["Left Turn", "Right Turn"]))
xlims!(ax_effectors, (0,3))
ylims!(ax_effectors, (0,1))

#define the axis for the weights layout
ax_weights = Axis(weights_layout[1,1]; backgroundcolor = :lightgrey)

####
#get position of left sensor as an observable for plotting
sL = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    heading = getindex(b.heading)[1]
    x = getindex(b.pos[1])
    y = getindex(b.pos[2])
    sL = (agent_radius * cos(heading+deg2rad(sens_degrees[1])) + x, agent_radius * sin(heading+deg2rad(sens_degrees[1])) + y)
    [sL]
end

#and the right sensor
sR = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    heading = getindex(b.heading)[1]
    x = getindex(b.pos[1])
    y = getindex(b.pos[2])
    sR = (agent_radius * cos(heading-deg2rad(sens_degrees[1])) + x, agent_radius * sin(heading-deg2rad(sens_degrees[1])) + y)
    [sR]
end

#and the position of the agent
agentLoc = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    pos = b.pos
    Circle(GLMakie.Point2f(pos), agent_radius)
end

#get a list of the agent positions over time, for plotting where it has been
positions = @lift begin
    a = $(abmobs.adf)
    b = unique(filter(:id => n -> n == 1, a))
    b.pos
end

#get the mean activations of the reservoir nodes
meanAct = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_act]
end

#and the mean errors
meanErr = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_err]
end

#and the mean targets
meanTarg = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_targ]
end

#get the activation of the sensors
sensActs = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    b.inputs
end

#get the timestep
step_ = @lift begin
    a = $(abmobs.adf)
    b = last(a)
    string(b.step[1])
end

#get the effector activations
effectorActs = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    b.outputs
end

#get the colors of each node in the network plot -- blue if not spiking, yellow if spiking
colors = @lift begin
    [x == 0 ? :blue : :yellow for x in $spikes]
end

#get the weights of the reservoir 
_weights2 = @lift begin
    x =reduce(vcat, $(wmat))
    x[x .!= 0]
end

####
#plot the gradient concentration as a heatmap
heatmap!(ax, xs, ys, zs)

#plot the timestep
text!(ax_text, step_, position=(0,0),align = (:center, :center),fontsize=100)

# #plot the agent position over time
# scatterlines!(ax, positions; color = :black, markersize = 5, markeralpha = .01)

#plot the agent
poly!(ax, agentLoc, color = :pink)

#plot the agent's sensors
scatter!(ax, sL ; markersize = 10, color = :red)
scatter!(ax, sR ; markersize = 10, color = :blue)

#plot the sensor activations
barplot!(ax_sensors, sensActs; color = sensColors, strokecolor = :black, strokewidth = 1)

#plot the mean activations, errors, and targets of the reservoir nodes
barplot!(ax_acts, [1], meanAct; color = :black, strokecolor = :black, strokewidth = 1)
barplot!(ax_acts, [2], meanErr; color = :black, strokecolor = :black, strokewidth = 1)
barplot!(ax_acts, [3], meanTarg; color = :black, strokecolor = :black, strokewidth = 1)

#plot the effector activations
barplot!(ax_effectors, effectorActs; color = :black, strokecolor = :black, strokewidth = 1)

#plot the weights
hist!(ax_weights, _weights2)

#adjust the limits of the weights plot as needed
on(abmobs.model) do m
    autolimits!(ax_weights)
end

############

# create a network object from the reservoir link matrix using the SimpleWeightedGraphs package
G = SimpleWeightedDiGraph(link_mat)
#get the positions of each node in the network plot as a spring layout
pos_x, pos_y = GraphPlot.spring_layout(G)

# Create edges
edge_x = []
edge_y = []

for edge in Graphs.edges(G)
    push!(edge_x, pos_x[Graphs.src(edge)])
    push!(edge_x, pos_x[Graphs.dst(edge)])
    push!(edge_y, pos_y[Graphs.src(edge)])
    push!(edge_y, pos_y[Graphs.dst(edge)])
end

#plot the edges
lines!(ax_network, reduce(vcat,edge_x), reduce(vcat,edge_y), color=(:black, .1))

#plot the nodes with colors based on spiking/not-spiking
scatter!(ax_network, reduce(vcat, pos_x), reduce(vcat,pos_y); markersize = 20, color = colors)

########
## generate the figure
fig

# #save video
# frames = 1:2000
# record(fig, "BraitenbergAgent2.mp4", frames; framerate = 40) do i
#     step!(abmobs, 1)
# end

# #output data
# ts_pos_ = DataFrame(ts_pos)
# ts_spikes_ = DataFrame(ts_spikes)
# ts_acts_ = DataFrame(ts_acts)
# ts_heading_ = DataFrame(heading = ts_heading)
# ts_sens_ = DataFrame(ts_sens)
# ts_eff_ = DataFrame(ts_eff)
# ts_hits_ = DataFrame(hits = ts_hits)

# using CSV
# fill = "_perturb2"
# CSV.write("./WallAvoidance/Data/pos" * fill * ".csv", ts_pos_)
# CSV.write("./WallAvoidance/Data/spikes" * fill * ".csv", ts_spikes_)
# CSV.write("./WallAvoidance/Data/acts" * fill * ".csv", ts_acts_)
# CSV.write("./WallAvoidance/Data/heading" * fill * ".csv", ts_heading_)
# CSV.write("./WallAvoidance/Data/sens" * fill * ".csv", ts_sens_)
# CSV.write("./WallAvoidance/Data/eff" * fill * ".csv", ts_eff_)
# CSV.write("./WallAvoidance/Data/hits" * fill * ".csv", ts_hits_)



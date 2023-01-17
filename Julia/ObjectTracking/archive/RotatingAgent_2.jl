using SimpleWeightedGraphs, Graphs
import GraphPlot
include("./ObjectTracking/RotatingAgent_Functions_2.jl");

## params
plotIter = 3600*2
nnodes = 200
p_link = .1
leak = .25 #
leaktype = 1 #1 = leak a percentage, or 2 = leak a constant rate
lrate_wmat = .01
lrate_targ = .01
targ_min = 1.0 
sens_offset = 30
movement_amp = 10
input_amp = 2.75
stim_speed = 1
global acts_neg = 1

####
sens_degrees = vcat(sort(collect(range(start=-61,stop=63,step=4)).+30,rev=true), sort(collect(range(start=-60,stop=64,step=4)).-30,rev=true))
sensColors=reduce(vcat,[repeat([:red],32), repeat([:blue], 32)])

sensory_nodes = StatsBase.sample(range(1,nnodes),Weights(repeat([1],nnodes)),Int(round(nnodes*p_link)))
global input_wmat=zeros((length(sens_degrees),nnodes))
for row in range(1,size(input_wmat)[1])
    #for col in sensory_nodes
    for col in range(1,nnodes)
        input_wmat[row,col]=StatsBase.sample([0,input_amp],Weights([1-p_link,p_link]))
    end
end

global link_mat = zeros((nnodes,nnodes))
inhibitory_nodes=[]
for row in range(1,size(link_mat)[1])
    inhibitory = StatsBase.sample([0,1],Weights([.75,.25]))
    #inhibitory = StatsBase.sample([0,1],Weights([1,0]))
    if inhibitory == 1
        push!(inhibitory_nodes,row)
    end
    for col in range(1,size(link_mat)[2])
        if row == col
            continue
        elseif inhibitory == 1
            link_mat[row,col] = StatsBase.sample([0,-1],Weights([1-p_link,p_link]))
        else
            link_mat[row,col] = StatsBase.sample([0,1],Weights([1-p_link,p_link]))
        end
    end
end
        
global wmat= Observable(zeros((nnodes,nnodes)))
for row in range(1,size(wmat[])[1])
    for col in range(1,size(wmat[])[2])
        if link_mat[row,col] == 1
            val = rand(Normal(1,.1))
            if val < 0
                val = 0
            end
            wmat[][row,col] = val
        elseif link_mat[row,col] == -1
            val = rand(Normal(-1,.1))
            if val > 0
                val = 0
            end
            wmat[][row,col] = val
        end
    end
end
startWmat = copy(wmat[])

effector_nodes=[]
global output_wmat=zeros((nnodes,2))
for row in range(1,size(output_wmat)[1])
    for col in range(1,size(output_wmat)[2])
        output_wmat[row,col]=StatsBase.sample([0,1],Weights([1-p_link,p_link]))
        if output_wmat[row,col] != 0
            push!(effector_nodes,row)
        end
    end
end


######

global spikes = Observable(zeros(nnodes))
global targets = repeat([targ_min],nnodes)
global acts = zeros(nnodes)
global inputs = zeros(length(sens_degrees))
global output_acts = zeros(2)

global n = 0
global direction = 1
global mean_act = 0.0
global mean_err = 0.0

model = ABM(Agent, 
ContinuousSpace((3,3), 
periodic = true); 
properties = Dict(
    :n => n,
    :direction => direction,
    :mean_act => mean_act,
    :mean_err => mean_err,
    :inputs => inputs,
    :outputs => output_acts,
    :weights => wmat,
),
scheduler = Schedulers.fastest
)

add_agent_pos!(
    _Agent(
        nextid(model),
        (1.5,1.5),
        π/2,
        sens_degrees,

    ),
    model
)

add_agent_pos!(
    Stimulus(
        nextid(model),
        ( stim_radius * cos(0)+1.5, stim_radius * sin(0)+1.5),

    ),
    model
)

#### plotting
adata = [:pos, :heading, :degree]
mdata = [:mean_act, :mean_err, :inputs, :outputs]

params = Dict(   
)

fig, ax, abmobs = abmplot(model; 
    dummystep, model_step!,
    add_controls = true, enable_inspection = true,
    adata, mdata, figure = (; resolution = (2400,1600))
)

sensors_layout = fig[1,end+1] = GridLayout()
acts_layout = fig[1,end+1] = GridLayout()
network_layout = fig[3, end-2] = GridLayout()
effectors_layout = fig[3,end-1] = GridLayout()
weights_layout = fig[3,end] = GridLayout()

ax_sensors = Axis(sensors_layout[1,1];
    backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:length(sens_degrees)))
xlims!(ax_sensors, (0,length(sens_degrees)+1))
ylims!(ax_sensors, (0,1))

ax_acts = Axis(acts_layout[1,1]; backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:2, ["meanActs","meanErrs"]), xticklabelrotation = pi/8)
xlims!(ax_acts, (0,3))
ylims!(ax_acts, (-3,3))

ax_network = Axis(network_layout[1,1])

ax_effectors = Axis(effectors_layout[1,1];
    backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:2, ["Left Turn", "Right Turn"]))
xlims!(ax_effectors, (0,3))
ylims!(ax_effectors, (0,1))

ax_weights = Axis(weights_layout[1,1]; backgroundcolor = :lightgrey)

####
sL = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    heading = getindex(b.heading)[1]
    sL = (agent_radius * cos(heading+deg2rad(30)) + 1.5, agent_radius * sin(heading+deg2rad(30)) + 1.5)#( agent_radius * ( cos(heading+deg2rad(30)) + sin(heading+deg2rad(30))im), agent_radius * ( cos(heading-deg2rad(30)) + sin(heading-deg2rad(30))im))
    [sL]
end

sR = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    heading = getindex(b.heading)[1]
    sR = (agent_radius * cos(heading-deg2rad(30)) + 1.5, agent_radius * sin(heading-deg2rad(30)) + 1.5)
    [sR]
end

stimPos = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 2, a))
    b.pos
end

meanAct = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_act]
end

meanErr = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_err]
end

sensActs = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    b.inputs
end

effectorActs = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    b.outputs
end

colors = @lift begin
    [x == 0 ? :blue : :yellow for x in $spikes]
end

_weights2 = @lift begin
    x =reduce(vcat, $(wmat)[link_mat .!= 0])
    #x[x .!= 0]
end

####
poly!(ax, Circle(GLMakie.Point2f(1.5, 1.5), agent_radius), color = :pink)

scatter!(ax, sL ; markersize = 20, color = :red)
scatter!(ax, sR ; markersize = 20, color = :blue)
scatter!(ax, stimPos ; markersize = 30, color = :green)

barplot!(ax_sensors, sensActs; color = sensColors, strokecolor = :black, strokewidth = 1)

barplot!(ax_acts, [1], meanAct; color = :black, strokecolor = :black, strokewidth = 1)
barplot!(ax_acts, [2], meanErr; color = :black, strokecolor = :black, strokewidth = 1)

barplot!(ax_effectors, effectorActs; color = :black, strokecolor = :black, strokewidth = 1)

hist!(ax_weights, _weights2)

on(abmobs.model) do m
    #autolimits!(ax_weights)
end

############

G = SimpleWeightedDiGraph(link_mat)
pos_x, pos_y = GraphPlot.spring_layout(G)

# Create plot points
edge_x = []
edge_y = []

for edge in Graphs.edges(G)
    push!(edge_x, pos_x[Graphs.src(edge)])
    push!(edge_x, pos_x[Graphs.dst(edge)])
    push!(edge_y, pos_y[Graphs.src(edge)])
    push!(edge_y, pos_y[Graphs.dst(edge)])
end

#  Color node points by the number of connections.
#color_map = [spikes[node] for node in 1:nnodes]

lines!(ax_network, reduce(vcat,edge_x), reduce(vcat,edge_y), color=(:black, .1))

scatter!(ax_network, reduce(vcat, pos_x), reduce(vcat,pos_y); markersize = 20, color = colors)

########
fig

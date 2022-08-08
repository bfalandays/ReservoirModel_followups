using Agents, LinearAlgebra, InteractiveDynamics, GLMakie, Statistics, Meshes, Distributions, DataFrames, StatsBase

mutable struct Agent <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    complexPos::ComplexF64
    speed::Float64
    heading::Float64
end

## params
plotIter = 3600*4
nnodes = 500
p_link = .1
leak = .1 #
leaktype = 1 #1 = leak a percentage, or 2 = leak a constant rate
lrate_wmat = .01
lrate_targ = .01
targ_min = 1 
sens_offset = 30
rot_amp = 20
input_amp = .75
forward_amp = .2
spike_cost = 0#.1
stim_speed = 1
noise_sd = 0#.2
weight_sd_init = 1#5
effector_type = 1 
acts_neg=0

####
sens_degrees = vcat(sort(collect(range(-61,4,63)).+30,rev=true), sort(collect(range(-60,4,64)).-30,rev=true))

input_wmat=zeros((length(sens_degrees),nnodes))
for row in range(1,size(input_wmat)[1])
    for col in range(1,nnodes)
        input_wmat[row,col]=StatsBase.sample([0,input_amp],Weights([1-p_link,p_link]))
    end
end

link_mat = zeros((nnodes,nnodes))
inhibitory_nodes=[]
for row in range(1,size(link_mat)[1])
    inhibitory = StatsBase.sample([0,1],Weights([.75,.25]))
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
        
wmat=zeros((nnodes,nnodes))
for row in range(1,size(wmat)[1])
    for col in range(1,size(wmat)[2])
        if link_mat[row,col] == 1
            wmat[row,col] =  rand(Normal(input_amp,.1))
        elseif link_mat[row,col] == -1
            wmat[row,col] =  rand(Normal(-input_amp,.1))
        end
    end
end

effector_nodes=[]
output_wmat=zeros((nnodes,3))
for row in range(1,size(output_wmat)[1])
    for col in range(1,size(output_wmat)[2])
        output_wmat[row,col]=StatsBase.sample([0,1],Weights([1-p_link,p_link]))
        if output_wmat[row,col] != 0
            push!(effector_nodes,row)
        end
    end
end

### functions

function get_input_acts(agent,stim)
    
    sens_degrees_ = sens_degrees .+ agent.heading
    sens_degrees_[sens_degrees_ > 360] .-= 360
    sens_degrees_[sens_degrees_ <= 0] .+= 360
    
    sens_dists = abs(sens_degrees_ .- degree)
    sens_acts = zeros(length(sens_degrees))
    sens_acts[sens_dists <= 4] = 1 #- sens_dists/60
        
    sens_acts
end

function get_acts(acts,leak,spikes,wmat,input,input_wmat,targets, refractory, refractory_acts,spike_rep)
    
    acts = acts * (1-leak) + dot(input, input_wmat) + dot(spikes, wmat)

    if acts_neg == 0
        acts[acts<0]=0
    end
    
    prev_spikes = spikes
    thresholds=targets*2
    spikes[acts >= thresholds]=1
    spikes[acts < thresholds]=0
    
    acts[spikes==1]-=thresholds[spikes==1]
       
    errors=acts-targets
    
    [acts, spikes, errors, refractory,refractory_acts, prev_spikes, spike_rep]
end

function learning(learn_on,link_mat,spikes,prev_spikes, errors,wmat,targets)
    
    active_neighbors=np.abs(link_mat.copy())
    active_neighbors[prev_spikes==0,:]=0
    d_wmat = active_neighbors.copy()
    active_neighbors=np.sum(active_neighbors,axis=0)#+np.repeat(1,nnodes)
    
    if learn_on==1
        if np.sum(active_neighbors) >0
            d_wmat = errors*d_wmat
            d_wmat=(d_wmat/active_neighbors)
            #d_wmat=(d_wmat/(active_neighbors+1))
            d_wmat=np.nan_to_num(d_wmat)
            wmat-=d_wmat
        end
            
        #targets=targets+((errors/(active_neighbors+1))*lrate_targ)
        targets=targets+(errors*lrate_targ) 
        targets[targets<targ_min]=targ_min
    end
            
    [wmat, targets]
end


###
model = ABM(Agent, ContinuousSpace((10,10), periodic = true); scheduler = Schedulers.fastest)



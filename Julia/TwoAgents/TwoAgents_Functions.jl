using Agents, LinearAlgebra, InteractiveDynamics, GLMakie, Statistics, Meshes, Distributions, DataFrames, StatsBase
import GeometryBasics as gb

function gaussian(x, σ)
    return .5 *exp(-((x - 0)^2) / (2 * σ^2))
end

function DoG(x,σ₁, σ₂)
    A = 1 / (pdf(Normal(0, σ₁),0) - pdf(Normal(0, σ₂),0))

    return A/5 * (pdf(Normal(0, σ₁), x) - pdf(Normal(0, σ₂), x)) 
end


n = 46 #length(unique(agent.sens_angles))
σ_taper = 10.0
tapering_factor(row, n, σ_taper) = exp(-((row - (n+1)/2)^2) / (2 * σ_taper^2))
mex_wmat = zeros(n, n)
for row in 1:n
    for col in 1:n        
        # Calculate the weight using the DoG kernel
        local σ₁ = 7
        local σ₂ = 1.5*σ₁
        local dog_weight = DoG(abs(row - col), σ₁, σ₂) #gaussian(abs(row-col),3) #

        local tapering = tapering_factor(row, n, σ_taper)
        
        # Combine the DoG weight and tapering factor
        local weight = dog_weight #* tapering
        
        # Store the weight in the weight matrix
        mex_wmat[row, col] = weight
    end
end

# using Plots
# # # Plots.plot(agent.sens_anglesL, sens_acts[1])
# # # Plots.plot(agent.sens_anglesR, sens_acts[2])
# Plots.plot(unique(agent.sens_angles), sens_acts)

# Plots.plot(unique(agent.sens_angles), vec(sum(sens_acts .* mex_wmat, dims = 1)))


mutable struct Agent <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    type::Symbol # :agent, :stim
    complexPos::ComplexF64
    speed::Float64
    heading::Float64
    wall_sens_angles::Vector{Float64}
    sens_angles::Vector{Float64}
    sens_anglesL::Vector{Float64}
    sens_anglesR::Vector{Float64}
    sens_Cpos::NTuple{2, ComplexF64}
    tStep::Int64

    sensory_nodes::Vector{Int64}
    inhibitory_nodes::Vector{Int64}
    input_wmat::Matrix{Float64}
    link_mat::Matrix{Float64}
    wmat::Matrix{Float64}
    effector_nodes::Vector{Int64}
    output_wmat::Matrix{Float64}

    spikes::Vector{Float64}
    prev_spikes::Vector{Float64}
    targets::Vector{Float64}
    errors::Vector{Float64}
    acts::Vector{Float64}
    inputs::Vector{Float64}
    output_acts::Vector{Float64}
    newPos::NTuple{2,Float64}
    collisions::Int64
end

function make_mats(model)
    sensory_nodes = StatsBase.sample(range(1,model.nnodes),Weights(repeat([1],model.nnodes)),Int(round(model.nnodes*model.p_link)))
    input_wmat=zeros((length(unique(model.sens_angles))+2,model.nnodes))
    #input_wmat=zeros((length(unique(model.sens_angles)),model.nnodes))
    #input_wmat=zeros((length(agent.sens_anglesL)*2 + 2,model.nnodes))
    for row in range(1,size(input_wmat)[1])
        #for col in sensory_nodes
        for col in range(1,model.nnodes)
            input_wmat[row,col]=StatsBase.sample([0,model.input_amp],Weights([1-model.p_link,model.p_link]))
        end
    end

    link_mat = zeros((model.nnodes,model.nnodes))
    inhibitory_nodes=[]
    for row in range(1,size(link_mat)[1])
        inhibitory = StatsBase.sample([1,-1],Weights([.75,.25]))
        #inhibitory = StatsBase.sample([1,-1],Weights([1,0]))
        push!(inhibitory_nodes,inhibitory)
        for col in range(1,size(link_mat)[2])
            if row == col
                continue
            else
                link_mat[row,col] = StatsBase.sample([0,1],Weights([1-model.p_link,model.p_link]))
            end
        end
    end

    wmat= zeros((model.nnodes,model.nnodes))
    for row in range(1,size(wmat)[1])
        for col in range(1,size(wmat)[2])
            if link_mat[row,col] == 1
                #wmat[row,col] =  rand(Normal(model.input_amp,.1))
                wmat[row,col] =  rand(truncated(Normal(1,.2), lower = 0))
            end
        end
    end

    effector_nodes=[]
    output_wmat=zeros((model.nnodes,2))
    for row in range(1,size(output_wmat)[1])
        for col in range(1,size(output_wmat)[2])
            output_wmat[row,col]=StatsBase.sample([0,1],Weights([1-model.p_link,model.p_link]))
            if output_wmat[row,col] != 0
                push!(effector_nodes,row)
            end
        end
    end

    return sensory_nodes, inhibitory_nodes, input_wmat, link_mat, wmat, effector_nodes, output_wmat

end

function init_agent(model, id, pos, heading)
    
    sensory_nodes, inhibitory_nodes, input_wmat, link_mat, wmat, effector_nodes, output_wmat = make_mats(model)
    wall_sens_angles = model.wall_sens_angles
    sens_angles = model.sens_angles
    sens_anglesL = model.sens_anglesL
    sens_anglesR = model.sens_anglesR

    spikes = zeros(model.nnodes)
    prev_spikes = copy(spikes)
    targets = repeat([model.targ_min],model.nnodes)
    errors = zeros(model.nnodes)
    acts = zeros(model.nnodes)
    inputs = zeros(length(unique(model.sens_angles))+2)
    #inputs = zeros(length(model.sens_anglesL)*2+2)
    output_acts = zeros(2)
    collisions = 0
    
    Agent(id, pos, :agent, pos[1] + pos[2]im, 0, heading, wall_sens_angles,sens_angles, sens_anglesL, sens_anglesR, ( model.agent_radius * ( cos(heading+deg2rad(30)) + sin(heading+deg2rad(30))im), model.agent_radius * ( cos(heading-deg2rad(30)) + sin(heading-deg2rad(30))im)), 0, sensory_nodes, inhibitory_nodes, input_wmat, link_mat, wmat, effector_nodes,output_wmat, spikes, prev_spikes, targets, errors, acts, inputs, output_acts, pos, collisions)
end

### functions
function sync_update(model)
    limits = [[model.agent_radius, model.space.extent[1] - model.agent_radius],[model.agent_radius, model.space.extent[2] - model.agent_radius]]

    function checkWallCollision(agent)
        if agent.newPos[1] > limits[1][2]
            wall_hit = "right"
        elseif agent.newPos[1] < limits[1][1]
            wall_hit = "left"
        elseif agent.newPos[2] > limits[2][2]
            wall_hit = "top"
        elseif agent.newPos[2] < limits[2][1]
            wall_hit = "bottom"
        else
            wall_hit = "none"
        end
        if wall_hit != "none"
            agent.collisions += 1

            unitVec = [cos(agent.heading), sin(agent.heading)]
            if wall_hit == "right"
                agent.heading = atan(unitVec[2], -unitVec[1])
                agent.newPos = (limits[1][2] - (agent.newPos[1] - limits[1][2]), agent.newPos[2])
            elseif wall_hit == "left"
                agent.heading = atan(unitVec[2], -unitVec[1])
                agent.newPos = (limits[1][1] + (limits[1][1] - agent.newPos[1]), agent.newPos[2])
            elseif wall_hit == "top"
                agent.heading = atan(-unitVec[2], unitVec[1])
                agent.newPos = (agent.newPos[1], limits[2][2] - (agent.newPos[2] - limits[2][2]))
            elseif wall_hit == "bottom"
                agent.heading = atan(-unitVec[2], unitVec[1])
                agent.newPos = (agent.newPos[1], limits[2][1] + (limits[2][1] - agent.newPos[2]))
            end
        end
    end

    function checkNeighborCollision(agent)
        for neighborID in collect(allids(model))[allids(model) .!= agent.id]
            neighbor = model[neighborID]
            distanceToNeighbor = sqrt((neighbor.newPos[1] - agent.newPos[1])^2 + (neighbor.newPos[2] - agent.newPos[2])^2)
            if distanceToNeighbor < 2*model.agent_radius
                hit = true
            else
                hit = false
            end
            if hit == true
                agentVel = [agent.newPos[1] - agent.pos[1], agent.newPos[2] - agent.pos[2]]
                neighborVel = [neighbor.newPos[1] - neighbor.pos[1], neighbor.newPos[2] - neighbor.pos[2]]

                xDist = neighbor.pos[1] - agent.pos[1]
                yDist = neighbor.pos[2] - agent.pos[2]
                normalVector = [xDist, yDist]
                normalVector ./= norm(normalVector)
                tangentVector = [-normalVector[2], normalVector[1]]

                agScalarNormal = dot(agentVel, normalVector)
                neighborScalarNormal = dot(neighborVel, normalVector)

                agScalarTangent = dot(agentVel, tangentVector)
                neighborScalarTangent = dot(neighborVel, tangentVector)

                m1 = 1
                m2 = 1
                agScalarNormalAfter = (agScalarNormal * (m1 - m2) + 2 * m2 * neighborScalarNormal) / (m1 + m2)
                neighborScalarNormalAfter = (neighborScalarNormal * (m2-m1) + 2 * m1 * agScalarNormal) / (m1 + m2)

                agScalarNormalAfterVec = agScalarNormalAfter * normalVector
                neighborScalarNormalAfterVec = neighborScalarNormalAfter * normalVector

                agScalarNormalVec = agScalarTangent * tangentVector
                neighborScalarNormalVec = neighborScalarTangent * tangentVector

                agVel = agScalarNormalVec + agScalarNormalAfterVec
                neighborVel = neighborScalarNormalVec + neighborScalarNormalAfterVec

                agent.heading = atan(agVel[2], agVel[1])
                neighbor.heading = atan(neighborVel[2], neighborVel[1])

                midPt = agent.pos .+ (xDist,yDist)./2
                r = model.agent_radius + .2
                agCollisionPt = NTuple{2,Float64}(midPt .+ r .* -normalVector )
                neighborCollisionPt = NTuple{2,Float64}(midPt .+ r .* normalVector )

                agent.newPos = (agCollisionPt[1] + agVel[1], agCollisionPt[2] + agVel[2])
                neighbor.newPos = (neighborCollisionPt[1] + neighborVel[1], neighborCollisionPt[2] + neighborVel[2])
                # agent.newPos = (midPt[1] + agVel[1], midPt[2] + agVel[2])
                # neighbor.newPos = (midPt[1] + neighborVel[1], midPt[2] + neighborVel[2])
                # agent.newPos = agCollisionPt
                # neighbor.newPos = neighborCollisionPt

            end
        end
    end

    for agent in allagents(model)
        if model.physical_coupling == 1
            checkNeighborCollision(agent)
        end
        checkWallCollision(agent)
    end
    for agent in allagents(model)
        move_agent!(agent, (agent.newPos[1],agent.newPos[2]),model)
    end
end

function move_agent(agent,model)

    output_acts = agent.output_acts

    # translation
    vel_centre = (output_acts[1]+output_acts[2])/2 * model.wheel_radius
    pos_centre = vel_centre .* [cos(agent.heading), sin(agent.heading)]
    agent.newPos = (agent.pos[1] + pos_centre[1], agent.pos[2] + pos_centre[2])

    # rotation
    omega = (output_acts[2]-output_acts[1])/(2*model.agent_radius) 
    agent.heading += omega

end

function get_input_acts_combined(agent,model)
    pos = agent.pos
    heading = agent.heading
    #tmp = deg2rad.(agent.sens_angles) .+ heading
    tmp = [deg2rad.(agent.sens_anglesL) .+ heading,deg2rad.(agent.sens_anglesR) .+ heading]
    Rays = [[Meshes.Ray(Meshes.Point2(pos), Meshes.Vec(cos(a),sin(a))) for a in i] for i in tmp]

    neighborAgent = agent.id == 1 ? model[2] : model[1]
    distanceToNeighbor = sqrt((neighborAgent.pos[1] - pos[1])^2 + (neighborAgent.pos[2] - pos[2])^2)
    angleToNeighbor = atan(neighborAgent.pos[2] - pos[2], neighborAgent.pos[1] - pos[1])
    perpendicularAngle = angleToNeighbor > 0 ? angleToNeighbor - deg2rad(90) : angleToNeighbor + deg2rad(90)
    segmentPts = [
        (neighborAgent.pos[1] + model.agent_radius * cos(perpendicularAngle), neighborAgent.pos[2] + model.agent_radius * sin(perpendicularAngle)),
        (neighborAgent.pos[1] - model.agent_radius * cos(perpendicularAngle), neighborAgent.pos[2] - model.agent_radius * sin(perpendicularAngle))
    ]
    neighborSegment = Segment(segmentPts[1], segmentPts[2])

    function f(r)
        intersectDist = maxDist
        for face in box_faces
            if type(intersection(r, face)) != NotIntersecting
                tmp = get(intersection(r, face)) - Meshes.Point2(pos)
                intersectDist = sqrt(tmp[1]^2 + tmp[2]^2)
                break
            end
        end
        if type(intersection(r, neighborSegment)) != NotIntersecting
            intersectDist = distanceToNeighbor
        end
        return intersectDist
    end

    intersections = [[f(r) for r in i] for i in Rays]

    sens_acts = [1 .- i ./ maxDist for i in intersections]

    if model.noise > 0
        sens_acts = [i .+ rand(Uniform(-model.noise,model.noise), length(i)) for i in sens_acts]
        sens_acts = [[j < 0 ? 0 : j for j in i] for i in sens_acts]
    end

    sens_acts = vcat(sens_acts[1],sens_acts[2])
    return sens_acts
end

function get_input_acts_from_agents(agent,model)
    pos = agent.pos
    heading = agent.heading
    #tmp = deg2rad.(agent.sens_angles) .+ heading
    tmp = [deg2rad.(agent.sens_anglesL) .+ heading,deg2rad.(agent.sens_anglesR) .+ heading]
    Rays = [[Meshes.Ray(Meshes.Point2(pos), Meshes.Vec(cos(a),sin(a))) for a in i] for i in tmp]

    neighborAgent = agent.id == 1 ? model[2] : model[1]
    distanceToNeighbor = sqrt((neighborAgent.pos[1] - pos[1])^2 + (neighborAgent.pos[2] - pos[2])^2)
    angleToNeighbor = atan(neighborAgent.pos[2] - pos[2], neighborAgent.pos[1] - pos[1])
    perpendicularAngle = angleToNeighbor > 0 ? angleToNeighbor - deg2rad(90) : angleToNeighbor + deg2rad(90)
    segmentPts = [
        (neighborAgent.pos[1] + model.agent_radius * cos(perpendicularAngle), neighborAgent.pos[2] + model.agent_radius * sin(perpendicularAngle)),
        (neighborAgent.pos[1] - model.agent_radius * cos(perpendicularAngle), neighborAgent.pos[2] - model.agent_radius * sin(perpendicularAngle))
    ]
    neighborSegment = Segment(segmentPts[1], segmentPts[2])

    function f(r)
        intersectDist = maxDist
        if type(intersection(r, neighborSegment)) != NotIntersecting
            intersectDist = distanceToNeighbor
        end
        return intersectDist
    end

    intersections = [[f(r) for r in i] for i in Rays]

    sens_acts = [1 .- i ./ maxDist for i in intersections]

    if model.noise > 0
        sens_acts = [i .+ rand(Uniform(-model.noise,model.noise), length(i)) for i in sens_acts]
        sens_acts = [[j < 0 ? 0 : j for j in i] for i in sens_acts]
    end

    sens_acts = vcat(sens_acts[1],sens_acts[2])
    #sens_acts = vec(sum(sens_acts .* mex_wmat, dims = 1))
    return sens_acts
end

function get_input_acts_from_walls(agent,model)

    pos = agent.pos
    heading = agent.heading

    tmp = deg2rad.(agent.wall_sens_angles) .+ heading
    Rays = [Meshes.Ray(Meshes.Point2(pos), Meshes.Vec(cos(a),sin(a))) for a in tmp]
    function f(r)
        intersectDist = maxDist
        for face in box_faces
            if type(intersection(r, face)) != NotIntersecting
                tmp = get(intersection(r, face)) - Meshes.Point2(pos)
                intersectDist = sqrt(tmp[1]^2 + tmp[2]^2)
                break
            end
        end
        return intersectDist
    end
    intersections = [f(r) for r in Rays] 
    sens_acts = [1 .- i ./ maxDist for i in intersections]

    if model.noise > 0
        sens_acts .+= rand(Uniform(-model.noise,model.noise), length(sens_acts)) 
        sens_acts = [j < 0 ? 0 : j for j in sens_acts]
    end

    return sens_acts
end

function get_input_acts(agent,model)

    if model.visual_coupling == 1
        sens_acts = vcat(get_input_acts_from_walls(agent,model), get_input_acts_from_agents(agent,model))
    else
        sens_acts = vcat(get_input_acts_from_walls(agent,model), repeat([0], length(unique(agent.sens_angles))))
        #sens_acts = vcat(get_input_acts_from_walls(agent,model), repeat([0], length(agent.sens_anglesL)*2))
    end
    
    agent.inputs=sens_acts
end

function get_acts(agent, model)
    
    agent.acts = (agent.acts .* (1-model.leak) .+ sum(agent.inputs .* agent.input_wmat, dims = 1)' .+ sum(agent.spikes .* agent.wmat .* agent.inhibitory_nodes, dims = 1)')[:,1]

    #agent.acts = (agent.acts .* (1-model.leak) .+ sum(agent.inputs .* agent.input_wmat, dims = 1)' .+ sum(agent.spikes .* agent.wmat, dims = 1)')[:,1]

    if model.acts_neg == 0
        agent.acts[agent.acts .< 0] .= 0
    end
    
    agent.prev_spikes = copy(agent.spikes)
    thresholds = agent.targets .* 2
    agent.spikes[agent.acts .>= thresholds] .= 1
    agent.spikes[agent.acts .< thresholds] .= 0
    
    agent.acts[agent.spikes .== 1] .-= thresholds[agent.spikes .== 1]
       
    agent.errors = agent.acts .- agent.targets

    agent.output_acts = (sum(agent.spikes .* agent.output_wmat,dims = 1) ./ sum(agent.output_wmat, dims=1))[1,:]
end

function learning(agent, model)
    
    active_neighbors=abs.(agent.link_mat)
    active_neighbors[agent.prev_spikes .== 0,:] .= 0
    d_wmat = copy(active_neighbors)
    active_neighbors = sum(active_neighbors,dims=1)
    
    if model.learn_on==1
        if sum(active_neighbors) >0
            d_wmat = (agent.errors' .* d_wmat)
            d_wmat=(d_wmat ./ active_neighbors)
            d_wmat[isinf.(d_wmat)] .= 0 
            d_wmat[isnan.(d_wmat)] .= 0 
            d_wmat[agent.inhibitory_nodes .== -1,:] .*= -1
            agent.wmat .-= d_wmat .* model.lrate_wmat
            agent.wmat[agent.wmat .< 0] .= 0
        end
            
        agent.targets .+= (agent.errors .* model.lrate_targ) 
        agent.targets[agent.targets .< model.targ_min] .= model.targ_min
    end
            
end

function model_step!(model)

    for agent in allagents(model)
        
        agent.tStep += 1
        get_input_acts(agent, model)

        get_acts(agent,model)

        learning(agent,model)

        move_agent(agent,model)

    end

    sync_update(model)

    model.n += 1
end
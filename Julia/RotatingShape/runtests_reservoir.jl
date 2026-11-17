using Test, SparseArrays, LinearAlgebra, DelimitedFiles
include(joinpath(@__DIR__,"RotatingShape_Functions.jl"))
using .RotatingShapeEnvironment
include(joinpath(@__DIR__,"RotatingShape_Reservoir.jl"))
using .RotatingShapeReservoir

@testset "Visual reservoir and hemisphere-specific input" begin
    env=initialize_environment(;N=8)
    m=initialize_reservoir(env;nnodes=10,sensory_fraction=1.,input_p_link=1.,n_subnet_edges=7)
    @test size(m.input_wmat)==(64,10)
    @test issparse(m.input_wmat)
    @test all(iszero,m.input_wmat[1:32,6:10])
    @test all(iszero,m.input_wmat[33:64,1:5])
    @test all(m.input_wmat[1:32,1:5] .== 5.)
    @test all(m.input_wmat[33:64,6:10] .== 5.)
    @test nnz(m.input_wmat)==320
    @test sum(m.link_mat[1:5,6:10])+sum(m.link_mat[6:10,1:5])==7
    @test all(iszero,diag(m.link_mat))
    core=RotatingShapeReservoir.AllostaticLanguage.initialize_model(;nnodes=10,
        subnetworks=1,n_subnet_edges=7,lrate_wmat=1.)
    @test m.link_mat==core.link_mat && m.wmat==core.wmat
    f=render_frame(env)
    old_inputs=Float64.(vec(f.object_mask))
    a,s,e,prev=RotatingShapeReservoir.AllostaticLanguage.get_acts(m.acts,m.params.leak,
        m.spikes,m.wmat,old_inputs,m.input_wmat,m.targets)
    before=copy(m.input_wmat)
    presented=RotatingShapeReservoir.model_step!(m;learn_on=false)
    @test m.inputs==old_inputs
    @test reshape(m.inputs,8,8)==Float64.(presented.object_mask)
    @test m.acts==a && m.spikes==s && m.errors==e
    @test m.input_wmat==before
    @test env.step==1
    @test all(iszero,m.inputs[vec(.!presented.object_mask)])
    small=run_rotating_model(initialize_environment(;N=8);iterations=20,
        nnodes=10,n_subnet_edges=7,save=false,progress_every=0)
    same=run_rotating_model(initialize_environment(;N=8);iterations=20,
        nnodes=10,n_subnet_edges=7,save=false,progress_every=0)
    @test small.ts.spikes==same.ts.spikes
    @test size(small.ts.spikes)==(20,10) && eltype(small.ts.spikes)==UInt8
    @test all(x->x==0 || x==1,small.ts.spikes)
    @test small.ts.time≈collect(0:19)./30
    @test small.ts.angle≈small.ts.time.*(pi/2)
    @test all(isfinite,small.ts.mean_error)
    @test all(iszero,small.model.wmat[small.model.link_mat .== 0])
    @test all(small.model.targets .>= 1.)
    empty=initialize_reservoir(initialize_environment(;N=8);nnodes=10,input_p_link=0.)
    @test nnz(empty.input_wmat)==0
    large=initialize_reservoir(initialize_environment(;N=64);input_p_link=.1)
    @test abs(nnz(large.input_wmat)/(4096*5)-.1)<.01
    @test length(large.params.sensory_nodes)==10
    @test count(<=(50),large.params.sensory_nodes)==5
    @test length(unique(large.params.sensory_nodes))==10
    @test all(iszero,large.input_wmat[:,setdiff(1:100,large.params.sensory_nodes)])
    selected=initialize_reservoir(initialize_environment(;N=8);nnodes=100,input_p_link=1.)
    @test nnz(selected.input_wmat)==32*10
    @test findall(vec(sum(selected.input_wmat;dims=1)).>0)==selected.params.sensory_nodes
    repeat_model=initialize_reservoir(initialize_environment(;N=64))
    @test repeat_model.params.sensory_nodes==large.params.sensory_nodes
    @test repeat_model.input_wmat==large.input_wmat
    none=initialize_reservoir(initialize_environment(;N=8);sensory_fraction=0.)
    @test isempty(none.params.sensory_nodes) && nnz(none.input_wmat)==0
    @test_throws ArgumentError initialize_reservoir(env;sensory_fraction=1.1)
    @test_throws ArgumentError initialize_reservoir(env;sensory_fraction=-.1)
    input_row,node,_=findnz(large.input_wmat)
    @test all((input_row .<= 2048).==(node .<= 50))
    untouched=initialize_environment(;N=8)
    run_rotating_model(untouched;iterations=2,nnodes=10,save=false,progress_every=0)
    @test untouched.step==0
    @test_throws ArgumentError initialize_reservoir(untouched;nnodes=9)
    @test_throws ArgumentError initialize_reservoir(untouched;input_p_link=1.1)
    mktempdir() do dir
        save_run(small,dir)
        @test readdlm(joinpath(dir,"spikes.csv"),',',Int;skipstart=1)==small.ts.spikes
        @test length(readlines(joinpath(dir,"timesteps.csv")))==21
        @test isfile(joinpath(dir,"input_edges.csv"))
        @test isfile(joinpath(dir,"sensory_nodes.csv"))
        @test occursin("black:1 white:0",read(joinpath(dir,"run_parameters.txt"),String))
    end
end

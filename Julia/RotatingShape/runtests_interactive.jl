using Test, DelimitedFiles, Statistics
include(joinpath(@__DIR__,"RotatingShape_Functions.jl"))
using .RotatingShapeEnvironment
include(joinpath(@__DIR__,"RotatingShape_Reservoir.jl"))
using .RotatingShapeReservoir
include(joinpath(@__DIR__,"RotatingShape_Interactive.jl"))
using .RotatingShapeInteractive

@testset "Interactive stepping, synchronization and recording" begin
    mktempdir() do dir
        env=initialize_environment(;N=8)
        d=launch_rotating_dashboard(env;iterations=6,nnodes=10,
            backend=:cairo,show=false,start_timer=false,output_dir=dir)
        @test !d.running[] && d.state.model.n==0
        @test d.image_data[]==RotatingShapeEnvironment.plot_pixels(render_frame(env))
        @test d.absolute_errors[]==abs.(d.state.model.acts .- d.state.model.targets)
        @test d.target_values[]==d.state.model.targets
        @test d.target_values[] !== d.state.model.targets
        @test isempty(d.error_history[])
        d.buttons.step.clicks[] += 1
        @test d.state.model.n==1
        @test length(d.error_history[])==1
        @test d.state.ts.mean_abs_error[1]≈mean(abs.(d.state.model.acts .- d.state.model.targets))
        @test d.error_history[][1][2]≈d.state.ts.mean_abs_error[1]
        @test d.absolute_errors[]==abs.(d.state.model.acts .- d.state.model.targets)
        @test d.target_values[]==d.state.model.targets
        @test d.target_values[] !== d.state.model.targets
        @test d.state.last_frame.time==0
        @test reshape(d.state.model.inputs,8,8)==Float64.(d.state.last_frame.object_mask)
        @test d.image_data[]==RotatingShapeEnvironment.plot_pixels(d.state.last_frame)
        @test all((d.node_colors[].==:yellow).==(d.state.model.spikes.>0))
        d.advance(2)
        @test d.state.model.n==3
        @test [p[1] for p in d.error_history[]]==[1,2,3]
        @test [p[2] for p in d.error_history[]]≈d.state.ts.mean_abs_error[1:3]
        @test d.state.last_frame.time≈2/30
        reference=run_rotating_model(env;iterations=3,nnodes=10,save=false,progress_every=0)
        result=interactive_result(d)
        @test result.ts.spikes==reference.ts.spikes
        @test result.ts.time==reference.ts.time
        @test result.ts.mean_abs_error==reference.ts.mean_abs_error
        @test result.model.wmat==reference.model.wmat
        d.buttons.save.clicks[] += 1
        @test d.state.saved_iterations==3
        @test length(readlines(joinpath(d.state.run_dir,"spikes.csv")))==4
        old_dir=d.state.run_dir
        model=d.state.model
        d.buttons.reset.clicks[] += 1
        @test d.state.model===model
        @test d.state.model.n==0 && d.state.run_dir!=old_dir
        @test all(iszero,d.state.model.spikes)
        @test isempty(d.error_history[])
        @test d.absolute_errors[]==abs.(d.state.model.acts .- d.state.model.targets)
        @test d.target_values[]==d.state.model.targets
        @test d.target_values[] !== d.state.model.targets
        @test d.state.model.wmat==d.state.start_wmat
        d.advance(6)
        @test d.state.model.n==6 && !d.running[]
        @test d.state.saved_iterations==6
        @test length(readlines(joinpath(d.state.run_dir,"spikes.csv")))==7
        d.advance(1)
        @test d.state.model.n==6
        @test length(d.error_history[])==6
        d.reset()
        d.advance(2)
        current=d.state.run_dir
        d.stop()
        @test length(readlines(joinpath(current,"spikes.csv")))==3
        @test env.step==0
    end
end

@testset "Playback timer" begin
    mktempdir() do dir
        d=launch_rotating_dashboard(initialize_environment(;N=8);iterations=3,
            nnodes=10,backend=:cairo,show=false,start_timer=true,autoplay=true,
            playback_fps=60,output_dir=dir)
        @test timedwait(()->d.state.model.n==3 && d.state.saved_iterations==3,10.)==:ok
        @test !d.running[]
        @test length(d.error_history[])==3
        @test d.absolute_errors[]==abs.(d.state.model.acts .- d.state.model.targets)
        @test d.target_values[]==d.state.model.targets
        @test d.target_values[] !== d.state.model.targets
        @test d.state.last_frame.time≈2/30
        @test d.image_data[]==RotatingShapeEnvironment.plot_pixels(d.state.last_frame)
        d.stop()
        @test !isopen(d.timer[])
    end
end

@testset "Subnetwork error histories and shared target bins" begin
    mktempdir() do dir
        d=launch_rotating_dashboard(initialize_environment(;N=8);iterations=5,
            nnodes=20,backend=:cairo,show=false,start_timer=false,output_dir=dir)
        d.state.model.targets[1:10] .= 2.
        d.state.model.targets[11:20] .= 5.
        d.advance(3)
        m=d.state.model
        @test d.state.ts.mean_abs_error_A[3]≈mean(abs.(m.acts[1:10].-m.targets[1:10]))
        @test d.state.ts.mean_abs_error_B[3]≈mean(abs.(m.acts[11:20].-m.targets[11:20]))
        @test [p[2] for p in d.error_history_A[]]≈d.state.ts.mean_abs_error_A[1:3]
        @test [p[2] for p in d.error_history_B[]]≈d.state.ts.mean_abs_error_B[1:3]
        @test d.state.ts.mean_abs_error[1:3]≈(d.state.ts.mean_abs_error_A[1:3].+d.state.ts.mean_abs_error_B[1:3])./2
        @test d.targets_A[]==m.targets[1:10]
        @test d.targets_B[]==m.targets[11:20]
        @test first(d.target_bins[])<=minimum(m.targets)
        @test last(d.target_bins[])>=maximum(m.targets)
        @test length(d.target_bins[])==21
        d.reset()
        @test isempty(d.error_history_A[]) && isempty(d.error_history_B[])
        @test d.targets_A[]==d.state.model.targets[1:10]
        d.stop()
    end
end

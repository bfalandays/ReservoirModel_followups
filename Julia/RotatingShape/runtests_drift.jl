using Test, LinearAlgebra, Statistics, DelimitedFiles
include("RotatingShape_Drift.jl")
using .RotatingShapeDrift
@testset "Window PCA, orientation alignment and static figure" begin
    spikes = UInt8.([mod(i*j+i÷3,2) for i in 1:80,j in 1:8])
    r = (;ts=(;spikes,angle=collect(0:79).*pi/6))
    a=window_spike_pca(r;window=11:60,nodes=:A)
    @test a.spikes==Float64.(spikes[11:60,1:4])
    @test a.angles==r.ts.angle[11:60]
    @test all(0 .<= a.orientation_degrees .< 360)
    @test a.iterations==collect(11:60)
    @test a.scores≈(a.spikes.-transpose(a.center))*a.loadings
    @test norm(a.loadings)≈1
    @test abs(mean(a.scores))<1e-12
    values=eigvals(Symmetric(cov(a.spikes)))
    @test a.explained_variance≈maximum(values)/sum(values)
    @test a.loadings[argmax(abs.(a.loadings))]>0
    @test window_spike_pca(r;window=11:60,nodes=:B).nodes==collect(5:8)
    @test length(window_spike_pca(r;window=11:60,nodes=:all).loadings)==8
    @test_throws ArgumentError window_spike_pca(r;window=0:10)
    @test_throws ArgumentError window_spike_pca(r;window=70:90)
    @test_throws ArgumentError window_spike_pca(r;window=1:1)
    @test_throws ArgumentError window_spike_pca(r;window=1:20,nodes=[1,1])
    @test_throws ArgumentError window_spike_pca((;ts=(;spikes=zeros(20,8),angle=zeros(20)));window=1:20)
    mktempdir() do dir
        path=joinpath(dir,"pca.png")
        p=plot_orientation_pc1(r;window=11:60,savepath=path)
        @test isfile(path) && filesize(path)>1000
        @test length(p.analyses)==2
        open(joinpath(dir,"spikes.csv"),"w") do io
            writedlm(io,permutedims(["n$i" for i in 1:8]),',');writedlm(io,spikes,',')
        end
        open(joinpath(dir,"timesteps.csv"),"w") do io
            writedlm(io,["iteration" "angle_radians"],',');writedlm(io,hcat(1:80,r.ts.angle),',')
        end
        loaded=load_drift_run(dir)
        @test loaded.ts.spikes==spikes
        @test loaded.ts.angle≈r.ts.angle
        @test plot_orientation_pc1(dir;window=11:60).analyses[1].scores≈a.scores
    end
end

@testset "Joint full-network PC1-PC2 trajectory" begin
    spikes=Float64.([mod(i*j+i÷(j+1),2) for i in 1:90,j in 1:12])
    r=(;ts=(;spikes,angle=collect(0:89).*pi/6))
    a=window_spike_pca(r;window=11:70,nodes=:all,ncomponents=2)
    @test a.nodes==collect(1:12)
    @test size(a.component_scores)==(60,2)
    @test size(a.component_loadings)==(12,2)
    @test a.component_scores≈(spikes[11:70,:].-transpose(a.center))*a.component_loadings
    @test a.component_loadings'*a.component_loadings≈Matrix{Float64}(I,2,2)
    ev=reverse(eigvals(Symmetric(cov(spikes[11:70,:]))))
    @test a.explained_variances≈ev[1:2]./sum(ev)
    @test a.scores==a.component_scores[:,1]
    @test a.explained_variances[1]>=a.explained_variances[2]
    @test_throws ArgumentError window_spike_pca(r;window=1:2,ncomponents=2)
    @test_throws ArgumentError window_spike_pca(r;window=1:20,ncomponents=0)
    mktempdir() do dir
        p=plot_spike_pc12(r;window=11:70,savepath=joinpath(dir,"pc12.png"))
        @test filesize(joinpath(dir,"pc12.png"))>1000
        @test p.analysis.component_scores≈a.component_scores
    end
end

@testset "Separate subnetwork PC1-PC2 trajectories" begin
    spikes=Float64.([mod(i*j+i÷(j+1),2) for i in 1:90,j in 1:12])
    r=(;ts=(;spikes,angle=collect(0:89).*pi/6))
    mktempdir() do dir
        p=plot_subnetwork_pc12(r;window=11:70,savepath=joinpath(dir,"subnets.png"))
        @test filesize(joinpath(dir,"subnets.png"))>1000
        @test length(p.analyses)==2
        @test p.analyses[1].nodes==collect(1:6)
        @test p.analyses[2].nodes==collect(7:12)
        for (a,group) in zip(p.analyses,(:A,:B))
            reference=window_spike_pca(r;window=11:70,nodes=group,ncomponents=2)
            @test size(a.component_scores)==(60,2)
            @test a.component_scores≈reference.component_scores
            @test a.spikes==spikes[11:70,a.nodes]
            @test a.iterations==collect(11:70)
        end
    end
end

@testset "Paired subnetwork PC1 and time in 3D" begin
    spikes=Float64.([mod(i*j+i÷(j+1),2) for i in 1:90,j in 1:12])
    r=(;ts=(;spikes,angle=collect(0:89).*pi/6))
    mktempdir() do dir
        p=plot_subnetwork_pc1_time(r;window=11:70,savepath=joinpath(dir,"pc1_time.png"))
        @test filesize(joinpath(dir,"pc1_time.png"))>1000
        @test size(p.coordinates)==(60,3)
        @test p.coordinates[:,1]==collect(11:70)
        for (j,group) in enumerate((:A,:B))
            reference=window_spike_pca(r;window=11:70,nodes=group)
            @test p.coordinates[:,j+1]≈reference.scores
            @test p.analyses[j].nodes==reference.nodes
        end
        open(joinpath(dir,"spikes.csv"),"w") do io
            writedlm(io,permutedims(["n$i" for i in 1:12]),',');writedlm(io,spikes,',')
        end
        open(joinpath(dir,"timesteps.csv"),"w") do io
            writedlm(io,["iteration" "angle_radians"],',');writedlm(io,hcat(1:90,r.ts.angle),',')
        end
        @test plot_subnetwork_pc1_time(dir;window=11:70).coordinates≈p.coordinates
    end
end

@testset "3D exploration controls" begin
    spikes=Float64.([mod(i*j+i÷(j+1),2) for i in 1:90,j in 1:12])
    r=(;ts=(;spikes,angle=collect(0:89).*pi/6))
    p=plot_subnetwork_pc1_time(r;window=11:70,interactive=true,backend=:cairo,show=false)
    reference=plot_subnetwork_pc1_time(r;window=11:70)
    @test p.coordinates==reference.coordinates
    @test p.axis.xlabel[]=="Model iteration"
    @test startswith(p.axis.ylabel[],"A: PC1")
    @test startswith(p.axis.zlabel[],"B: PC1")
    @test p.screen===nothing
    @test p.zoom_slider !== nothing && p.reset_button !== nothing
    original=p.axis.limits[]
    RotatingShapeDrift.set_close_to!(p.zoom_slider,2.)
    @test p.axis.limits[]!=original
    @test p.zoom_slider.value[]≈2.
    p.axis.azimuth[]=.3
    p.axis.elevation[]=.4
    p.reset_button.clicks[]+=1
    @test p.axis.azimuth[]≈1.55pi
    @test p.axis.elevation[]≈pi/8
    @test p.zoom_slider.value[]≈1.
    @test p.axis.limits[]==original
    @test reference.zoom_slider===nothing
    @test_throws ArgumentError plot_subnetwork_pc1_time(r;window=11:70,backend=:invalid)
end

@testset "Orientation-filtered subnetwork PCA" begin
    spikes=Float64.([mod(i*j+i÷(j+1),2) for i in 1:600,j in 1:12])
    r=(;ts=(;spikes,angle=collect(0:599).*pi/60))
    mktempdir() do dir
        p=plot_horizontal_subnetwork_pc12(r;window=1:600,savepath=joinpath(dir,"horizontal.png"))
        @test p.selected_iterations==collect(31:120:600)
        @test filesize(joinpath(dir,"horizontal.png"))>1000
        for (j,a) in enumerate(p.analyses)
            @test a.iterations==p.selected_iterations
            ids=j==1 ? (1:6) : (7:12)
            @test a.spikes==spikes[p.selected_iterations,ids]
            @test size(a.component_scores)==(5,2)
            @test a.component_scores≈(a.spikes.-transpose(a.center))*a.component_loadings
        end
        @test plot_horizontal_subnetwork_pc12(r;window=nothing).selected_iterations==p.selected_iterations
        shifted=plot_horizontal_subnetwork_pc12(r;window=1:600,target_angle_degrees=450.)
        @test shifted.selected_iterations==p.selected_iterations
        circular=(;ts=(;spikes=spikes[1:8,:],angle=deg2rad.([10.,359.,20.,1.,30.,0.,40.,360.])))
        @test plot_horizontal_subnetwork_pc12(circular;window=1:8,target_angle_degrees=0.,tolerance_degrees=1.).selected_iterations==[2,4,6,8]
        @test_throws ArgumentError plot_horizontal_subnetwork_pc12(r;window=1:100)
        @test_throws ArgumentError plot_horizontal_subnetwork_pc12(r;window=1:600,tolerance_degrees=-1.)
        @test_throws ArgumentError plot_horizontal_subnetwork_pc12(r;window=1:700)
    end
end

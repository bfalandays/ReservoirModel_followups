using Test
include(joinpath(@__DIR__,"RotatingShape_Functions.jl"))
using .RotatingShapeEnvironment

@testset "Pixel environment and constant-speed rotation" begin
    env = initialize_environment(;N=64,angular_speed=pi/2,dt=.1)
    f = render_frame(env)
    @test size(f.pixels) == (64,64)
    @test all(x -> x==0 || x==1,f.pixels)
    @test all(f.pixels[f.object_mask] .== 0)
    @test all(f.pixels[.!f.object_mask] .== 1)
    @test any(f.object_mask) && any(.!f.object_mask)
    @test f.time==0 && f.angle==0
    @test frame_at_time(env,4.).pixels == f.pixels
    @test frame_at_time(env,.5).pixels != f.pixels
    @test frame_at_time(env,1.).pixels != f.pixels
    @test env.step == 0
    next = step_environment!(env)
    @test next.time ≈ .1 && next.angle ≈ pi/20
    later = step_environment!(env;nsteps=9)
    @test later.time ≈ 1. && later.angle ≈ pi/2
    @test later.pixels == frame_at_time(env,1.).pixels
    left,right = visual_hemispheres(f)
    @test size(left)==(64,32) && size(right)==(64,32)
    @test hcat(left,right)==f.pixels
    @test env.xcenters[32]<0<env.xcenters[33]
    @test env.ycenters[1]>env.ycenters[end]
    cx,cy = RotatingShapeEnvironment.polygon_centroid(env.vertices)
    @test abs(cx)<1e-12 && abs(cy)<1e-12
    @test maximum(hypot.(env.vertices[:,1],env.vertices[:,2])) ≈ env.object_radius
    @test all(iszero,f.object_mask[1,:]) && all(iszero,f.object_mask[end,:])
    @test all(iszero,f.object_mask[:,1]) && all(iszero,f.object_mask[:,end])
    stationary = initialize_environment(;angular_speed=0.)
    @test frame_at_time(stationary,10.).pixels==render_frame(stationary).pixels
    clockwise = initialize_environment(;angular_speed=-pi/2)
    @test frame_at_time(clockwise,1.).pixels==frame_at_time(initialize_environment(),3.).pixels
    @test_throws ArgumentError initialize_environment(;N=63)
    @test_throws ArgumentError initialize_environment(;object_radius=1.)
    @test_throws ArgumentError initialize_environment(;dt=0.)
    @test_throws ArgumentError frame_at_time(env,-1.)
    @test_throws ArgumentError step_environment!(env;nsteps=-1)
end

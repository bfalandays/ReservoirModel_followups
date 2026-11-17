module RotatingShapeEnvironment

using CairoMakie
export initialize_environment, render_frame, frame_at_time, step_environment!,
       visual_hemispheres, plot_frame, plot_rotation_sequence, record_rotation

mutable struct VisualEnvironment
    N::Int
    side_length::Float64
    object_radius::Float64
    angular_speed::Float64
    dt::Float64
    theta0::Float64
    vertices::Matrix{Float64}
    xcenters::Vector{Float64}
    ycenters::Vector{Float64}
    step::Int
end

# Asymmetric L with unequal arms, subsequently centered on its area centroid.
const DEFAULT_SHAPE = [-.45 -.50; .45 -.50; .45 -.20; -.15 -.20; -.15 .80; -.45 .80]

function polygon_centroid(vertices)
    twice_area, cx, cy = 0., 0., 0.
    n = size(vertices,1)
    for i in 1:n
        j = mod1(i+1,n)
        cross = vertices[i,1]*vertices[j,2] - vertices[j,1]*vertices[i,2]
        twice_area += cross
        cx += (vertices[i,1]+vertices[j,1])*cross
        cy += (vertices[i,2]+vertices[j,2])*cross
    end
    abs(twice_area) > eps(Float64) || throw(ArgumentError("Object polygon must have nonzero area"))
    return (cx/(3*twice_area),cy/(3*twice_area))
end

"""
    initialize_environment(; N=64, side_length=2., object_radius=.65,
                            angular_speed=pi/2, dt=1/30, theta0=0., vertices=...)

N-by-N pixel square, with equal left/right halves. Angles are radians; speed is
radians per unit time, and dt is time per simulation step. Positive speed means
counterclockwise rotation. The polygon is centered on its area centroid and
scaled so its furthest vertex is object_radius from the square center.
"""
function initialize_environment(; N=64, side_length=2., object_radius=.65,
        angular_speed=pi/2, dt=1/30, theta0=0., vertices=DEFAULT_SHAPE)
    N isa Integer && N >= 2 && iseven(N) ||
        throw(ArgumentError("N must be an even integer >= 2 for equal visual hemispheres"))
    isfinite(side_length) && side_length > 0 || throw(ArgumentError("side_length must be positive and finite"))
    isfinite(object_radius) && 0 < object_radius < side_length/2 ||
        throw(ArgumentError("object_radius must lie strictly between 0 and side_length/2"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be positive and finite"))
    isfinite(angular_speed) && isfinite(theta0) || throw(ArgumentError("Angles and angular_speed must be finite"))
    size(vertices,2) == 2 && size(vertices,1) >= 3 && all(isfinite,vertices) ||
        throw(ArgumentError("vertices must be a finite M-by-2 simple polygon, M >= 3"))
    shape = Matrix{Float64}(vertices)
    cx,cy = polygon_centroid(shape)
    shape[:,1] .-= cx
    shape[:,2] .-= cy
    radius = maximum(hypot.(shape[:,1],shape[:,2]))
    shape .*= object_radius/radius
    pixel_width = side_length/N
    xcenters = collect(range(-side_length/2+pixel_width/2;step=pixel_width,length=N))
    ycenters = reverse(copy(xcenters))  # matrix row 1 is the TOP of the image
    return VisualEnvironment(N,side_length,object_radius,angular_speed,dt,theta0,
        shape,xcenters,ycenters,0)
end

# Pixel-center inclusion in a simple polygon; points on its boundary count inside.
function inside_polygon(x,y,vertices)
    inside = false
    n = size(vertices,1)
    for i in 1:n
        j = mod1(i+1,n)
        ax,ay = vertices[i,1],vertices[i,2]
        bx,by = vertices[j,1],vertices[j,2]
        cross = (x-ax)*(by-ay) - (y-ay)*(bx-ax)
        tolerance = 32eps(Float64)*max(1.,abs(ax),abs(ay),abs(bx),abs(by))
        if abs(cross) <= tolerance && min(ax,bx)-tolerance <= x <= max(ax,bx)+tolerance &&
                min(ay,by)-tolerance <= y <= max(ay,by)+tolerance
            return true
        end
        if (ay > y) != (by > y)
            x_intersection = ax + (y-ay)*(bx-ax)/(by-ay)
            x < x_intersection && (inside = !inside)
        end
    end
    return inside
end

"""
    frame_at_time(env, time)

Render at theta0 + angular_speed*time without changing env. Pixels use optical
intensity: 0.0 = black object, 1.0 = white background. object_mask uses true for
object occupancy. Matrices are [row, column], with row 1 at the top and column 1
at the left. A pixel is black when its center lies inside the rotated polygon.
"""
function frame_at_time(env::VisualEnvironment,time)
    isfinite(time) && time >= 0 || throw(ArgumentError("time must be finite and nonnegative"))
    angle = env.theta0 + env.angular_speed*time
    s,c = sincos(mod(angle,2pi))
    mask = falses(env.N,env.N)
    for col in 1:env.N, row in 1:env.N
        x,y = env.xcenters[col],env.ycenters[row]
        # Rotate pixel centers backwards; test against the fixed original polygon.
        local_x = c*x+s*y
        local_y = -s*x+c*y
        mask[row,col] = inside_polygon(local_x,local_y,env.vertices)
    end
    return (; pixels=Float64.(.!mask),object_mask=mask,time=Float64(time),angle)
end

render_frame(env::VisualEnvironment) = frame_at_time(env,env.step*env.dt)

function step_environment!(env::VisualEnvironment; nsteps=1)
    nsteps isa Integer && nsteps >= 0 || throw(ArgumentError("nsteps must be a nonnegative integer"))
    env.step += nsteps
    return render_frame(env)
end

"""Split image columns into equal left/right views without changing pixel order."""
function visual_hemispheres(frame)
    n = size(frame.pixels,2)
    iseven(n) || throw(ArgumentError("Image width must be even"))
    half = n ÷ 2
    return (; left=@view(frame.pixels[:,1:half]),right=@view(frame.pixels[:,half+1:n]))
end

# Makie matrix coordinates are [x,y], while our image coordinates are [row,column].
plot_pixels(frame) = permutedims(reverse(frame.pixels;dims=1))
frame_title(frame) = "t = $(round(frame.time;digits=2)), angle = $(round(rad2deg(mod(frame.angle,2pi));digits=1))°"

function decorate_axis!(ax,env; show_split=true, show_grid=false)
    half = env.side_length/2
    if show_grid
        boundaries = range(-half,half;length=env.N+1)
        vlines!(ax,boundaries;color=(:gray,.2),linewidth=.4)
        hlines!(ax,boundaries;color=(:gray,.2),linewidth=.4)
    end
    if show_split
        vlines!(ax,[0.];color=:darkorange,linestyle=:dash,linewidth=2)
    end
    limits!(ax,-half,half,-half,half)
    return ax
end

function plot_frame(env::VisualEnvironment; time=env.step*env.dt,
        show_split=true, show_grid=false,
        output_path=joinpath(@__DIR__,"Figures","RotatingShape_frame.png"))
    frame = frame_at_time(env,time)
    fig = Figure(resolution=(700,750))
    ax = Axis(fig[1,1];aspect=DataAspect(),xlabel="x",ylabel="y",title=frame_title(frame))
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    heatmap!(ax,edges,edges,plot_pixels(frame);colormap=[:black,:white],
        colorrange=(0,1),interpolate=false)
    decorate_axis!(ax,env;show_split,show_grid)
    Label(fig[2,1],"$(env.N) × $(env.N) pixels | Left visual half: x < 0 | Right visual half: x > 0";tellwidth=false)
    if output_path !== nothing
        mkpath(dirname(output_path))
        save(output_path,fig)
    end
    return fig
end

function plot_rotation_sequence(env::VisualEnvironment; times=collect(0:.5:3.5),
        ncols=4, show_split=true,
        output_path=joinpath(@__DIR__,"Figures","RotatingShape_sequence.png"))
    !isempty(times) || throw(ArgumentError("times must be nonempty"))
    ncols isa Integer && ncols >= 1 || throw(ArgumentError("ncols must be positive"))
    fig = Figure(resolution=(350*ncols,350*cld(length(times),ncols)))
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    for (i,time) in enumerate(times)
        frame = frame_at_time(env,time)
        row,col = fldmod(i-1,ncols)
        ax = Axis(fig[row+1,col+1];aspect=DataAspect(),title=frame_title(frame))
        heatmap!(ax,edges,edges,plot_pixels(frame);colormap=[:black,:white],
            colorrange=(0,1),interpolate=false)
        decorate_axis!(ax,env;show_split)
    end
    if output_path !== nothing
        mkpath(dirname(output_path))
        save(output_path,fig)
    end
    return fig
end

"""Record a constant-speed animation without advancing the simulation state."""
function record_rotation(env::VisualEnvironment; duration=4., framerate=30,
        show_split=true,
        output_path=joinpath(@__DIR__,"Figures","RotatingShape.gif"))
    isfinite(duration) && duration > 0 || throw(ArgumentError("duration must be positive and finite"))
    framerate isa Integer && framerate > 0 || throw(ArgumentError("framerate must be positive"))
    initial = frame_at_time(env,0.)
    image = Observable(plot_pixels(initial))
    fig = Figure(resolution=(650,650))
    ax = Axis(fig[1,1];aspect=DataAspect(),xlabel="x",ylabel="y",title=frame_title(initial))
    edges = range(-env.side_length/2,env.side_length/2;length=env.N+1)
    heatmap!(ax,edges,edges,image;colormap=[:black,:white],colorrange=(0,1),interpolate=false)
    decorate_axis!(ax,env;show_split)
    times = range(0.;step=1/framerate,length=max(1,ceil(Int,duration*framerate)))
    mkpath(dirname(output_path))
    record(fig,output_path,times;framerate) do time
        frame = frame_at_time(env,time)
        image[] = plot_pixels(frame)
        ax.title[] = frame_title(frame)
    end
    return output_path
end
end # module

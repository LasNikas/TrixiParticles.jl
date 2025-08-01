"""
    BoundarySPHSystem(initial_condition, boundary_model; movement=nothing, adhesion_coefficient=0.0)

System for boundaries modeled by boundary particles.
The interaction between fluid and boundary particles is specified by the boundary model.

# Arguments
- `initial_condition`: Initial condition (see [`InitialCondition`](@ref))
- `boundary_model`: Boundary model (see [Boundary Models](@ref boundary_models))

# Keyword Arguments
- `movement`: For moving boundaries, a [`BoundaryMovement`](@ref) can be passed.
- `adhesion_coefficient`: Coefficient specifying the adhesion of a fluid to the surface.
   Note: currently it is assumed that all fluids have the same adhesion coefficient.
"""
struct BoundarySPHSystem{BM, NDIMS, ELTYPE <: Real, IC, CO, M, IM,
                         CA} <: BoundarySystem{NDIMS}
    initial_condition    :: IC
    coordinates          :: CO # Array{ELTYPE, 2}
    boundary_model       :: BM
    movement             :: M
    ismoving             :: IM # Ref{Bool} (to make a mutable field compatible with GPUs)
    adhesion_coefficient :: ELTYPE
    cache                :: CA
    buffer               :: Nothing

    # This constructor is necessary for Adapt.jl to work with this struct.
    # See the comments in general/gpu.jl for more details.
    function BoundarySPHSystem(initial_condition, coordinates, boundary_model, movement,
                               ismoving, adhesion_coefficient, cache, buffer)
        ELTYPE = eltype(coordinates)

        new{typeof(boundary_model), size(coordinates, 1), ELTYPE, typeof(initial_condition),
            typeof(coordinates), typeof(movement), typeof(ismoving),
            typeof(cache)}(initial_condition, coordinates, boundary_model, movement,
                           ismoving, adhesion_coefficient, cache, buffer)
    end
end

function BoundarySPHSystem(initial_condition, model; movement=nothing,
                           adhesion_coefficient=0.0, color_value=0)
    coordinates = copy(initial_condition.coordinates)

    ismoving = Ref(!isnothing(movement))

    cache = create_cache_boundary(movement, initial_condition)
    cache = (cache..., color=Int(color_value))

    if movement !== nothing && isempty(movement.moving_particles)
        # Default is an empty vector, since the number of particles is not known when
        # instantiating `BoundaryMovement`.
        resize!(movement.moving_particles, nparticles(initial_condition))
        movement.moving_particles .= collect(1:nparticles(initial_condition))
    end

    # Because of dispatches boundary model needs to be first!
    return BoundarySPHSystem(initial_condition, coordinates, model, movement,
                             ismoving, adhesion_coefficient, cache, nothing)
end

function Base.show(io::IO, system::BoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "BoundarySPHSystem{", ndims(system), "}(")
    print(io, system.boundary_model)
    print(io, ", ", system.movement)
    print(io, ", ", system.adhesion_coefficient)
    print(io, ", ", system.cache.color)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::BoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "BoundarySPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "movement function",
                     isnothing(system.movement) ? "nothing" :
                     string(system.movement.movement_function))
        summary_line(io, "adhesion coefficient", system.adhesion_coefficient)
        summary_line(io, "color", system.cache.color)
        summary_footer(io)
    end
end

"""
    BoundaryDEMSystem(initial_condition, normal_stiffness)

System for boundaries modeled by boundary particles.
The interaction between fluid and boundary particles is specified by the boundary model.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in a future releases.

"""
struct BoundaryDEMSystem{NDIMS, ELTYPE <: Real, IC,
                         ARRAY1D, ARRAY2D} <: BoundarySystem{NDIMS}
    initial_condition :: IC
    coordinates       :: ARRAY2D # [dimension, particle]
    radius            :: ARRAY1D # [particle]
    normal_stiffness  :: ELTYPE
    buffer            :: Nothing

    function BoundaryDEMSystem(initial_condition, normal_stiffness)
        coordinates = initial_condition.coordinates
        radius = 0.5 * initial_condition.particle_spacing *
                 ones(length(initial_condition.mass))
        NDIMS = size(coordinates, 1)

        return new{NDIMS, eltype(coordinates), typeof(initial_condition), typeof(radius),
                   typeof(coordinates)}(initial_condition, coordinates, radius,
                                        normal_stiffness, nothing)
    end
end

function Base.show(io::IO, system::BoundaryDEMSystem)
    @nospecialize system # reduce precompilation time

    print(io, "BoundaryDEMSystem{", ndims(system), "}(")
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::BoundaryDEMSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "BoundaryDEMSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_footer(io)
    end
end

"""
    BoundaryMovement(movement_function, is_moving; moving_particles=nothing)

# Arguments
- `movement_function`: Time-dependent function returning an `SVector` of ``d`` dimensions
                       for a ``d``-dimensional problem.
- `is_moving`: Function to determine in each timestep if the particles are moving or not. Its
    boolean return value is mandatory to determine if the neighborhood search will be updated.

# Keyword Arguments
- `moving_particles`: Indices of moving particles. Default is each particle in [`BoundarySPHSystem`](@ref).

In the example below, `movement` describes particles moving in a circle as long as
the time is lower than `1.5`.

# Examples
```jldoctest; output = false
movement_function(t) = SVector(cos(2pi*t), sin(2pi*t))
is_moving(t) = t < 1.5

movement = BoundaryMovement(movement_function, is_moving)

# output
BoundaryMovement{typeof(movement_function), typeof(is_moving), Vector{Int64}}(movement_function, is_moving, Int64[])
```
"""
struct BoundaryMovement{MF, IM, MP}
    movement_function :: MF
    is_moving         :: IM
    moving_particles  :: MP # Vector{Int}
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function BoundaryMovement(movement_function, is_moving; moving_particles=nothing)
    if !(movement_function(0.0) isa SVector)
        @warn "Return value of `movement_function` is not of type `SVector`. " *
              "Returning regular `Vector`s causes allocations and significant performance overhead."
    end

    # Default value is an empty vector, which will be resized in the `BoundarySPHSystem`
    # constructor to move all particles.
    moving_particles = isnothing(moving_particles) ? Int[] : vec(moving_particles)

    return BoundaryMovement(movement_function, is_moving, moving_particles)
end

create_cache_boundary(::Nothing, initial_condition) = (;)

function create_cache_boundary(::BoundaryMovement, initial_condition)
    initial_coordinates = copy(initial_condition.coordinates)
    velocity = zero(initial_condition.velocity)
    acceleration = zero(initial_condition.velocity)
    return (; velocity, acceleration, initial_coordinates)
end

timer_name(::Union{BoundarySPHSystem, BoundaryDEMSystem}) = "boundary"

@inline function Base.eltype(system::Union{BoundarySPHSystem, BoundaryDEMSystem})
    eltype(system.coordinates)
end

@inline function initial_coordinates(system::BoundarySPHSystem)
    initial_coordinates(system::BoundarySPHSystem, system.movement)
end

@inline initial_coordinates(system::BoundaryDEMSystem) = system.coordinates

@inline initial_coordinates(system::BoundarySPHSystem, ::Nothing) = system.coordinates

# We need static initial coordinates as reference when system is moving
@inline function initial_coordinates(system::BoundarySPHSystem, movement)
    return system.cache.initial_coordinates
end

function (movement::BoundaryMovement)(system, t, semi)
    (; coordinates, cache) = system
    (; movement_function, is_moving, moving_particles) = movement
    (; acceleration, velocity) = cache

    system.ismoving[] = is_moving(t)

    is_moving(t) || return system

    @threaded semi for particle in moving_particles
        pos_new = initial_coords(system, particle) + movement_function(t)
        vel = ForwardDiff.derivative(movement_function, t)
        acc = ForwardDiff.derivative(t_ -> ForwardDiff.derivative(movement_function, t_), t)

        @inbounds for i in 1:ndims(system)
            coordinates[i, particle] = pos_new[i]
            velocity[i, particle] = vel[i]
            acceleration[i, particle] = acc[i]
        end
    end

    return system
end

function (movement::Nothing)(system::System, t, semi)
    system.ismoving[] = false

    return system
end

@inline function nparticles(system::Union{BoundaryDEMSystem, BoundarySPHSystem})
    size(system.coordinates, 2)
end

# No particle positions are advanced for boundary systems,
# except when using `BoundaryModelDummyParticles` with `ContinuityDensity`.
@inline function n_moving_particles(system::Union{BoundarySPHSystem, BoundaryDEMSystem})
    return 0
end

@inline function n_moving_particles(system::BoundarySPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return nparticles(system)
end

@inline u_nvariables(system::Union{BoundarySPHSystem, BoundaryDEMSystem}) = 0

# For BoundaryModelDummyParticles with ContinuityDensity, this needs to be 1.
# For all other models and density calculators, it's irrelevant.
@inline v_nvariables(system::BoundarySPHSystem) = 1
@inline v_nvariables(system::BoundaryDEMSystem) = 0

@inline function current_coordinates(u, system::Union{BoundarySPHSystem, BoundaryDEMSystem})
    return system.coordinates
end

@inline function current_velocity(v, system::BoundaryDEMSystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_velocity(v, system::BoundarySPHSystem, particle)
    return current_velocity(v, system, system.movement, particle)
end

@inline function current_velocity(v, system, movement, particle)
    (; cache, ismoving) = system

    if ismoving[]
        return extract_svector(cache.velocity, system, particle)
    end

    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_velocity(v, system, movement::Nothing, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_velocity(v, system::BoundarySPHSystem)
    error("`current_velocity(v, system)` is not implemented for `BoundarySPHSystem`")
end

@inline function current_acceleration(system::BoundarySPHSystem, particle)
    return current_acceleration(system, system.movement, particle)
end

@inline function current_acceleration(system, movement, particle)
    (; cache, ismoving) = system

    if ismoving[]
        return extract_svector(cache.acceleration, system, particle)
    end

    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_acceleration(system, movement::Nothing, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function viscous_velocity(v, system::BoundarySPHSystem, particle)
    return viscous_velocity(v, system.boundary_model.viscosity, system, particle)
end

@inline function viscous_velocity(v, viscosity, system, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@inline function viscous_velocity(v, viscosity::Nothing, system, particle)
    return current_velocity(v, system, particle)
end

@inline function current_density(v, system::BoundarySPHSystem)
    return current_density(v, system.boundary_model, system)
end

@inline function current_pressure(v, system::BoundarySPHSystem)
    return current_pressure(v, system.boundary_model, system)
end

@inline function hydrodynamic_mass(system::BoundarySPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@inline function smoothing_kernel(system::BoundarySPHSystem, distance, particle)
    (; smoothing_kernel, smoothing_length) = system.boundary_model
    return kernel(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_length(system::BoundarySPHSystem, particle)
    return smoothing_length(system.boundary_model, particle)
end

function update_positions!(system::BoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    (; movement) = system

    movement(system, t, semi)
end

function update_quantities!(system::BoundarySPHSystem, v, u, v_ode, u_ode, semi, t)
    (; boundary_model) = system

    update_density!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

# This update depends on the computed quantities of the fluid system and therefore
# has to be in `update_final!` after `update_quantities!`.
function update_final!(system::BoundarySPHSystem, v, u, v_ode, u_ode, semi, t;
                       update_from_callback=false)
    (; boundary_model) = system

    # Note that `update_pressure!(::BoundarySPHSystem, ...)` is empty
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)

    return system
end

function write_u0!(u0, system::Union{BoundarySPHSystem, BoundaryDEMSystem})
    return u0
end

function write_v0!(v0,
                   system::Union{BoundarySPHSystem,
                                 BoundaryDEMSystem})
    return v0
end

function write_v0!(v0,
                   system::BoundarySPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    (; cache) = system.boundary_model
    (; initial_density) = cache

    for particle in eachparticle(system)
        # Set particle densities
        v0[1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::BoundarySPHSystem, v, u)
    return system
end

function restart_with!(system::BoundarySPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                       v, u)
    (; initial_density) = model.cache

    for particle in eachparticle(system)
        initial_density[particle] = v[1, particle]
    end

    return system
end

# To incorporate the effect at boundaries in the viscosity term of the RHS the neighbor
# viscosity model has to be used.
@inline function viscosity_model(system::BoundarySPHSystem,
                                 neighbor_system::FluidSystem)
    return neighbor_system.viscosity
end

function calculate_dt(v_ode, u_ode, cfl_number, system::BoundarySystem, semi)
    return Inf
end

function initialize!(system::BoundarySPHSystem, semi)
    initialize_colorfield!(system, system.boundary_model, semi)
    return system
end

function initialize_colorfield!(system, boundary_model, semi)
    return system
end

function initialize_colorfield!(system, ::BoundaryModelDummyParticles, semi)
    system_coords = system.coordinates
    (; smoothing_kernel, smoothing_length, cache) = system.boundary_model

    if haskey(cache, :initial_colorfield)
        foreach_point_neighbor(system, system, system_coords, system_coords, semi,
                               points=eachparticle(system)) do particle, neighbor,
                                                               pos_diff, distance
            cache.initial_colorfield[particle] += system.initial_condition.mass[particle] /
                                                  system.initial_condition.density[particle] *
                                                  system.cache.color *
                                                  kernel(smoothing_kernel,
                                                         distance,
                                                         smoothing_length)
            cache.neighbor_count[particle] += 1
        end
    end
    return system
end

function system_smoothing_kernel(system::BoundarySPHSystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.smoothing_kernel
end

function system_correction(system::BoundarySPHSystem{<:BoundaryModelDummyParticles})
    return system.boundary_model.correction
end

function system_data(system::BoundarySPHSystem, v_ode, u_ode, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    velocity = current_velocity(v, system)
    density = current_density(v, system)
    pressure = current_pressure(v, system)

    return (; coordinates, velocity, density, pressure)
end

function available_data(::BoundarySPHSystem)
    return (:coordinates, :velocity, :density, :pressure)
end

function system_data(system::BoundaryDEMSystem, v_ode, u_ode, semi)
    (; coordinates, radius, normal_stiffness) = system

    return (; coordinates, radius, normal_stiffness)
end

function available_data(::BoundaryDEMSystem)
    return (:coordinates, :radius, :normal_stiffness)
end

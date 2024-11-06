@doc raw"""
    EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                smoothing_length, sound_speed;
                                pressure_acceleration=inter_particle_averaged_pressure,
                                density_calculator=SummationDensity(),
                                transport_velocity=nothing,
                                alpha=0.5, viscosity=nothing,
                                acceleration=ntuple(_ -> 0.0, NDIMS), buffer_size=nothing,
                                source_terms=nothing)

System for particles of a fluid.
As opposed to the [weakly compressible SPH scheme](@ref wcsph), which uses an equation of state,
this scheme uses a pressure evolution equation to calculate the pressure.
See [Entropically Damped Artificial Compressibility for SPH](@ref edac) for more details on the method.

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.
- `sound_speed`:        Speed of sound.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).

# Keyword Arguments
- `viscosity`:      Viscosity model for this system (default: no viscosity).
                    Recommended: [`ViscosityAdami`](@ref).
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `pressure_acceleration`: Pressure acceleration formulation (default: inter-particle averaged pressure).
                        When set to `nothing`, the pressure acceleration formulation for the
                        corresponding [density calculator](@ref density_calculator) is chosen.
- `density_calculator`: [Density calculator](@ref density_calculator) (default: [`SummationDensity`](@ref))
- `transport_velocity`: [Transport Velocity Formulation (TVF)](@ref transport_velocity_formulation). Default is no TVF.
- `buffer_size`:    Number of buffer particles.
                    This is needed when simulating with [`OpenBoundarySPHSystem`](@ref).
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure, t)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).
                    Note that these source terms will not be used in the calculation of the
                    boundary pressure when using a boundary with
                    [`BoundaryModelDummyParticles`](@ref) and [`AdamiPressureExtrapolation`](@ref).
                    The keyword argument `acceleration` should be used instead for
                    gravity-like source terms.
"""
struct EntropicallyDampedSPHSystem{NDIMS, ELTYPE <: Real, IC, M, DC, K, V, TV, PR,
                                   PF, ST, B, C} <: FluidSystem{NDIMS, IC}
    initial_condition                 :: IC
    mass                              :: M # Vector{ELTYPE}: [particle]
    density_calculator                :: DC
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
    sound_speed                       :: ELTYPE
    viscosity                         :: V
    nu_edac                           :: ELTYPE
    acceleration                      :: SVector{NDIMS, ELTYPE}
    correction                        :: Nothing
    pressure_acceleration_formulation :: PF
    transport_velocity                :: TV
    particle_refinement               :: PR
    source_terms                      :: ST
    buffer                            :: B
    cache                             :: C

    function EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                         smoothing_length, sound_speed;
                                         pressure_acceleration=inter_particle_averaged_pressure,
                                         density_calculator=SummationDensity(),
                                         transport_velocity=nothing,
                                         particle_refinement=nothing,
                                         alpha=0.5, viscosity=nothing,
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         source_terms=nothing, buffer_size=nothing)
        buffer = isnothing(buffer_size) ? nothing :
                 SystemBuffer(nparticles(initial_condition), buffer_size)

        initial_condition = allocate_buffer(initial_condition, buffer)

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        pressure_acceleration = choose_pressure_acceleration_formulation(pressure_acceleration,
                                                                         density_calculator,
                                                                         NDIMS, ELTYPE,
                                                                         nothing)

        nu_edac = (alpha * smoothing_length * sound_speed) / 8

        cache = create_cache_density(initial_condition, density_calculator)
        cache = (; create_cache_edac(initial_condition, transport_velocity)...,
                 create_cache_edac_particle_refinement(initial_condition,
                                                       particle_refinement)..., cache...)

        new{NDIMS, ELTYPE, typeof(initial_condition), typeof(mass),
            typeof(density_calculator), typeof(smoothing_kernel), typeof(viscosity),
            typeof(transport_velocity), typeof(particle_refinement),
            typeof(pressure_acceleration), typeof(source_terms), typeof(buffer),
            typeof(cache)}(initial_condition, mass, density_calculator, smoothing_kernel,
                           smoothing_length, sound_speed, viscosity, nu_edac, acceleration_,
                           nothing, pressure_acceleration, transport_velocity,
                           particle_refinement, source_terms, buffer, cache)
    end
end

function Base.show(io::IO, system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "EntropicallyDampedSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "EntropicallyDampedSPHSystem{$(ndims(system))}")
        if system.buffer isa SystemBuffer
            summary_line(io, "#particles", nparticles(system))
            summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        else
            summary_line(io, "#particles", nparticles(system))
        end
        summary_line(io, "density calculator",
                     system.density_calculator |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity |> typeof |> nameof)
        summary_line(io, "ν₍EDAC₎", "≈ $(round(system.nu_edac; digits=3))")
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "tansport velocity formulation",
                     system.transport_velocity |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

create_cache_edac(initial_condition, ::Nothing) = (;)

function create_cache_edac(initial_condition, ::TransportVelocityAdami)
    pressure_average = copy(initial_condition.pressure)
    neighbor_counter = Vector{Int}(undef, nparticles(initial_condition))
    update_callback_used = Ref(false)

    return (; pressure_average, neighbor_counter, update_callback_used)
end

function create_cache_edac_particle_refinement(initial_condition, ::Nothing)
    return (;)
end

function create_cache_edac_particle_refinement(initial_condition, ::ParticleRefinement)
    return (;)
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, density_calculator)
    return ndims(system) * factor_tvf(system) + 1
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    return ndims(system) * factor_tvf(system) + 2
end

@inline function particle_density(v, ::ContinuityDensity,
                                  system::EntropicallyDampedSPHSystem, particle)
    return v[end - 1, particle]
end

@inline function particle_pressure(v, system::EntropicallyDampedSPHSystem, particle)
    return v[end, particle]
end

# WARNING!
# These functions are intended to be used internally to set the pressure
# of newly activated particles in a callback.
# DO NOT use outside a callback. OrdinaryDiffEq does not allow changing `v` and `u`
# outside of callbacks.
@inline function set_particle_pressure!(v, system::EntropicallyDampedSPHSystem, particle,
                                        pressure)
    v[end, particle] = pressure

    return v
end

@inline system_sound_speed(system::EntropicallyDampedSPHSystem) = system.sound_speed

@inline average_pressure(system, particle) = zero(eltype(system))

@inline function average_pressure(system::EntropicallyDampedSPHSystem, particle)
    average_pressure(system, system.transport_velocity, particle)
end

@inline function average_pressure(system, ::TransportVelocityAdami, particle)
    return system.cache.pressure_average[particle]
end

@inline average_pressure(system, ::Nothing, particle) = zero(eltype(system))

function update_quantities!(system::EntropicallyDampedSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    compute_density!(system, u, u_ode, semi, system.density_calculator)
    update_average_pressure!(system, system.transport_velocity, v_ode, u_ode, semi)
end

function update_average_pressure!(system, ::Nothing, v_ode, u_ode, semi)
    return system
end

# This technique is for a more robust `pressure_acceleration` but only with TVF.
# It results only in significant improvement for EDAC and not for WCSPH.
# See Ramachandran (2019) p. 582.
function update_average_pressure!(system, ::TransportVelocityAdami, v_ode, u_ode, semi)
    (; cache) = system
    (; pressure_average, neighbor_counter) = cache

    set_zero!(pressure_average)
    set_zero!(neighbor_counter)

    u = wrap_u(u_ode, system, semi)

    # Use all other systems for the average pressure
    @trixi_timeit timer() "compute average pressure" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               neighborhood_search) do particle, neighbor,
                                                       pos_diff, distance
            pressure_average[particle] += particle_pressure(v_neighbor_system,
                                                            neighbor_system,
                                                            neighbor)
            neighbor_counter[particle] += 1
        end
    end

    # We do not need to check for zero division here, as `neighbor_counter = 1`
    # for zero neighbors. That is, the `particle` itself is also taken into account.
    pressure_average ./= neighbor_counter

    return system
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::SummationDensity)
    for particle in eachparticle(system)
        v0[end, particle] = system.initial_condition.pressure[particle]
    end

    return v0
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    for particle in eachparticle(system)
        v0[end - 1, particle] = system.initial_condition.density[particle]
        v0[end, particle] = system.initial_condition.pressure[particle]
    end

    return v0
end

function restart_with!(system::EntropicallyDampedSPHSystem, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
        system.initial_condition.pressure[particle] = v[end, particle]
    end
end

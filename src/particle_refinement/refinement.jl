include("refinement_pattern.jl")
include("refinement_criteria.jl")

mutable struct ParticleRefinement{RL, NDIMS, ELTYPE, RP, RC, CNL}
    candidates           :: Vector{Int}
    candidates_mass      :: Vector{ELTYPE}
    refinement_pattern   :: RP
    refinement_criteria  :: RC
    criteria_next_levels :: CNL
    available_children   :: Int

    # Depends on refinement pattern, particle spacing and parameters ϵ and α.
    # Should be obtained prior to simulation in `create_child_system()`
    rel_position_children :: Vector{SVector{NDIMS, ELTYPE}}
    mass_ratio            :: Vector{ELTYPE}

    # It is essential to know the child system, which is empty at the beginning
    # and will be created in `create_child_system()` at the beginning of the simulation
    system_child::System

    # API --> parent system with `RL=0`
    function ParticleRefinement(refinement_criteria...;
                                refinement_pattern=CubicSplitting(),
                                criteria_next_levels=[])
        ELTYPE = eltype(first(refinement_criteria))
        NDIMS = ndims(first(refinement_criteria))

        return new{0, NDIMS, ELTYPE, typeof(refinement_pattern),
                   typeof(refinement_criteria),
                   typeof(criteria_next_levels)}([], [], refinement_pattern,
                                                 refinement_criteria,
                                                 criteria_next_levels, 0)
    end

    # Internal constructor for multiple refinement levels
    function ParticleRefinement{RL}(refinement_criteria, refinement_pattern,
                                    criteria_next_levels) where {RL}
        if refinement_criteria isa Tuple
            ELTYPE = eltype(first(refinement_criteria))
            NDIMS = ndims(first(refinement_criteria))
        else
            ELTYPE = eltype(refinement_criteria)
            NDIMS = ndims(refinement_criteria)
        end

        return new{RL, NDIMS, ELTYPE, typeof(refinement_pattern),
                   typeof(refinement_criteria),
                   typeof(criteria_next_levels)}([], [], refinement_pattern,
                                                 refinement_criteria,
                                                 criteria_next_levels, 0)
    end
end

@inline refinement_level(::ParticleRefinement{RL}) where {RL} = RL

@inline child_set(system, particle_refinement) = Base.OneTo(nchilds(system,
                                                                    particle_refinement))

@inline nchilds(system, ::Nothing) = 0
@inline nchilds(system, pr::ParticleRefinement) = nchilds(system, pr.refinement_pattern)

include("resize.jl")

# ==== Create child systems
function create_child_systems(systems)
    systems_ = ()
    foreach_system(systems) do system
        systems_ = (systems_..., create_child_system(system)...)
    end

    return (systems..., systems_...)
end

create_child_system(system) = ()
function create_child_system(system::FluidSystem)
    create_child_system(system, system.particle_refinement)
end

create_child_system(system::FluidSystem, ::Nothing) = ()

function create_child_system(system::FluidSystem,
                             particle_refinement::ParticleRefinement{RL}) where {RL}
    (; smoothing_length) = system
    (; criteria_next_levels, refinement_pattern, refinement_criteria) = particle_refinement

    NDIMS = ndims(system)

    # Distribute values according to refinement pattern
    smoothing_length_ = refinement_pattern.smoothing_ratio * system.smoothing_length
    particle_refinement.rel_position_children = relative_position_children(system,
                                                                           refinement_pattern)
    particle_refinement.mass_ratio = mass_distribution(system, refinement_pattern)

    # Create "empty" `InitialCondition` for child system
    particle_spacing_ = smoothing_length * refinement_pattern.separation_parameter
    coordinates_ = zeros(NDIMS, 2)
    velocity_ = similar(coordinates_)
    density_ = system.initial_condition.density[1]
    pressure_ = system.initial_condition.pressure[1]
    mass_ = nothing

    empty_ic = InitialCondition{NDIMS}(coordinates_, velocity_, mass_, density_, pressure_,
                                       particle_spacing_)

    #  Let recursive dispatch handle multiple refinement levels
    level = RL + 1
    particle_refinement_ = if isempty(criteria_next_levels)
        nothing
    else
        refinement_criteria = first(criteria_next_levels)
        ParticleRefinement{level}(refinement_criteria, refinement_pattern,
                                  criteria_next_levels[(level + 1):end])
    end

    system_child = copy_system(system; initial_condition=empty_ic,
                               smoothing_length=smoothing_length_,
                               particle_refinement=particle_refinement_,
                               particle_coarsening=ParticleCoarsening(system))

    # Empty mass vector leads to `nparticles(system_child) = 0`
    resize!(system_child.mass, 0)

    particle_refinement.system_child = system_child

    return (system_child,
            create_child_system(system_child, system_child.particle_refinement)...)
end

# ==== Refinement
function refinement!(v_ode, u_ode, _v_cache, _u_cache, semi, callback, t)
    foreach_system(semi) do system
        check_refinement_criteria!(system, v_ode, u_ode, semi, t)
    end

    if callback.coarsen
        foreach_system(semi) do system
            check_coarsening_criteria!(system, v_ode, u_ode, semi, t)
        end
    end

    resize_and_copy!(callback, semi, v_ode, u_ode, _v_cache, _u_cache)

    refine_particles!(callback, semi, v_ode, u_ode, _v_cache, _u_cache)

    if callback.coarsen
        coarsen_particles!(callback, semi, v_ode, u_ode, _v_cache, _u_cache)
    end
end

check_refinement_criteria!(system, v_ode, u_ode, semi, t) = system

function check_refinement_criteria!(system::FluidSystem, v_ode, u_ode, semi, t)
    check_refinement_criteria!(system, system.particle_refinement, v_ode, u_ode, semi, t)
end

check_refinement_criteria!(system, ::Nothing, v_ode, u_ode, semi, t) = system

function check_refinement_criteria!(system, particle_refinement::ParticleRefinement,
                                    v_ode, u_ode, semi, t)
    (; candidates, candidates_mass, refinement_criteria) = particle_refinement

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    Base.resize!(candidates, 0)
    Base.resize!(candidates_mass, 0)

    for particle in each_moving_particle(system)
        for refinement_criterion in refinement_criteria
            if (isempty(candidates) || particle != last(candidates)) &&
               refinement_criterion(system, particle, v, u, v_ode, u_ode, semi, t)
                push!(candidates, particle)
                # Store mass of candidate, since we lose the mass of the particle
                # when resizing the systems
                push!(candidates_mass, system.mass[particle])
            end
        end
    end

    return system
end

function refine_particles!(callback, semi, v_ode, u_ode, _v_cache, _u_cache)

    # Refine particles in all systems
    foreach_system(semi) do system
        refine_particles!(system, v_ode, u_ode, _v_cache, _u_cache, callback, semi)
    end
end

refine_particles!(system, v_ode, u_ode, _v_cache, _u_cache, callback, semi) = system

function refine_particles!(system::FluidSystem, v_ode, u_ode, _v_cache, _u_cache,
                           callback, semi)
    refine_particles!(system, system.particle_refinement, v_ode, u_ode, _v_cache, _u_cache,
                      callback, semi)
end

function refine_particles!(system::FluidSystem, ::Nothing, v_ode, u_ode, _v_cache, _u_cache,
                           callback, semi)
    return system
end

function refine_particles!(system_parent::FluidSystem,
                           particle_refinement::ParticleRefinement,
                           v_ode, u_ode, _v_cache, _u_cache, callback, semi)
    (; candidates, candidates_mass, system_child) = particle_refinement

    if !isempty(candidates)
        nhs = get_neighborhood_search(system_parent, semi)

        # Old storage
        v_parent = _wrap_v(_v_cache, system_parent, semi, callback)
        u_parent = _wrap_u(_u_cache, system_parent, semi, callback)

        # Resized storage
        v_child = wrap_v(v_ode, system_child, semi)
        u_child = wrap_u(u_ode, system_child, semi)

        particle_refinement.available_children = length(candidates) *
                                                 nchilds(system_parent, particle_refinement)

        # Loop over all refinement candidates
        mass_index = 1
        for particle_parent in candidates
            mass_parent = candidates_mass[mass_index]
            bear_children!(system_child, system_parent, particle_parent, mass_parent, nhs,
                           particle_refinement, v_parent, u_parent, v_child, u_child)

            particle_refinement.available_children -= nchilds(system_parent,
                                                              particle_refinement)
            mass_index += 1
        end
    end
end

# 6 (8) unkowns in 2d (3D) need to be determined for each newly born child particle
# --> mass, position, velocity, smoothing length
#
# Reducing the dof by using a fixed regular refinement pattern
# (given: position and number of child particles)
function bear_children!(system_child, system_parent, particle_parent, mass_parent, nhs,
                        particle_refinement, v_parent, u_parent, v_child, u_child)
    (; rel_position_children, available_children, mass_ratio) = particle_refinement

    parent_coords = current_coords(u_parent, system_parent, particle_parent)

    # Loop over all child particles of parent particle
    # The number of child particles depends on the refinement pattern
    for particle_child in child_set(system_parent, particle_refinement)
        absolute_index = particle_child + nparticles(system_child) - available_children

        system_child.mass[absolute_index] = mass_parent * mass_ratio[particle_child]

        # spread child positions according to the refinement pattern
        child_coords = parent_coords + rel_position_children[particle_child]
        for dim in 1:ndims(system_child)
            u_child[dim, absolute_index] = child_coords[dim]
        end

        volume = zero(eltype(system_child))
        p_a = zero(eltype(system_child))
        rho_a = zero(eltype(system_child))

        for dim in 1:ndims(system_child)
            v_child[dim, absolute_index] = zero(eltype(system_child))
        end

        for neighbor in eachneighbor(child_coords, nhs)
            neighbor_coords = current_coords(u_parent, system_parent, neighbor)
            pos_diff = child_coords - neighbor_coords

            distance2 = dot(pos_diff, pos_diff)

            # TODO: Check the following statement
            #
            # For the Navier–Stokes equations Feldman and Bonet showed that
            # the only way to conserve both total momentum and energy is to deﬁne
            # the velocities of the daughter particles `v_child` equal to the
            # the velocity of the original parent particle therefore: `v_child = v_parent`
            if distance2 <= nhs.search_radius^2
                distance = sqrt(distance2)
                kernel_weight = smoothing_kernel(system_parent, distance)
                volume += kernel_weight

                v_b = current_velocity(v_parent, system_parent, neighbor)
                p_b = particle_pressure_parent(v_parent, system_parent, neighbor)
                rho_b = particle_density_parent(v_parent, system_parent, neighbor)

                for dim in 1:ndims(system_child)
                    v_child[dim, absolute_index] += kernel_weight * v_b[dim]
                end

                rho_a += kernel_weight * rho_b
                p_a += kernel_weight * p_b
            end
        end

        if volume > eps()
            for dim in 1:ndims(system_child)
                v_child[dim, absolute_index] /= volume
            end

            rho_a /= volume
            p_a /= volume

            set_particle_density(absolute_index, v_child, system_child.density_calculator,
                                 system_child, rho_a)
            set_particle_pressure(absolute_index, v_child, system_child, p_a)
        end
    end

    return system_child
end

@inline particle_pressure_parent(v, ::WeaklyCompressibleSPHSystem, particle) = 0.0
@inline particle_pressure_parent(v, system::EntropicallyDampedSPHSystem, particle) = particle_pressure(v,
                                                                                                       system,
                                                                                                       particle)

@inline particle_density_parent(v, system, particle) = particle_density_parent(v, system,
                                                                               system.density_calculator,
                                                                               particle)

@inline particle_density_parent(v, system, ::SummationDensity, particle) = 0.0
@inline particle_density_parent(v, system, ::ContinuityDensity, particle) = particle_density(v,
                                                                                             system,
                                                                                             particle)
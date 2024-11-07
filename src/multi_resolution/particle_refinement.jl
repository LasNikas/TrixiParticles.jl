struct ParticleRefinement{SP, ELTYPE}
    splitting_pattern         :: SP
    max_spacing_ratio         :: ELTYPE
    mass_ref                  :: Vector{ELTYPE}
    merge_candidates          :: Vector{ELTYPE} # length = nparticles
    split_candidates          :: Vector{ELTYPE} # length = nparticles
    delete_candidates         :: Vector{Bool} # length = nparticles
    n_particles_before_resize :: Int
end

function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, t)
    check_refinement_criteria!(semi, v_ode, u_ode)

    # Update the spacing of particles (Algorthm 1)

    # Split the particles (Algorithm 2)

    # Merge the particles (Algorithm 3)

    # Shift the particles

    # Correct the particles

    # Update smoothing lengths

    resize!(semi, v_ode, u_ode, v_tmp, u_tmp)

    # Resize neighborhood search
    foreach_system(semi) do system
        foreach_system(semi) do neighbor_system
            search = get_neighborhood_search(system, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)

            # TODO
            resize_nhs!(search, system, neighbor_system, u_neighbor)
        end
    end

    return semi
end

function check_refinement_criteria!(semi::Semidiscretization, v_ode, u_ode)
    foreach_system(semi) do system
        check_refinement_criteria!(system, v_ode, u_ode)
    end
end

@inline check_refinement_criteria!(system, v_ode, u_ode) = system

@inline function check_refinement_criteria!(system::FluidSystem, v_ode, u_ode)
    (; refinement_criteria) = system.particle_refinement
    for criterion in refinement_criteria
        criterion(system, semi, v, u)
    end
end

function update_particle_spacing(semi::Semidiscretization, u_ode)
    foreach_system(semi) do system
        u = wrap_u(u_ode, system, semi)
        update_particle_spacing(system, u, semi)
    end
end

@inline update_particle_spacing(system, u, semi) = system

@inline function update_particle_spacing(system::FluidSystem, u, semi)
    (; smoothing_length, smoothing_length_factor) = system.cache
    (; mass_ref) = system.particle_refinement

    system_coords = current_coordinates(u, system)

    for particle in eachparticle(system)
        dp_min, dp_max, dp_avg = min_max_avg_spacing(system, semi, system_coords, particle)

        if dp_max / dp_min < max_spacing_ratio^3
            new_spacing = min(dp_max, max_spacing_ratio * dp_min)
        else
            new_spacing = dp_avg
        end

        smoothing_length[particle] = smoothing_length_factor * new_spacing
        mass_ref[particle] = system.density[particle] * new_spacing^(ndims(system))
    end

    return system
end

@inline function min_max_avg_spacing(system, semi, system_coords, particle)
    dp_min = Inf
    dp_max = zero(eltype(system))
    dp_avg = zero(eltype(system))
    counter_neighbors = 0

    foreach_system(semi) do neighbor_system
        neighborhood_search = get_neighborhood_search(particle_system, neighbor_system,
                                                      semi)

        u_neighbor_system = wrap_u(u_ode, neighbor, semi)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        PointNeighbors.foreach_neighbor(system_coords, neighbor_coords, neighborhood_search,
                                        particle) do particle, neighbor, pos_diff, distance
            dp_neighbor = particle_spacing(neighbor_system, neighbor)

            dp_min = min(dp_min, dp_neighbor)
            dp_max = max(dp_max, dp_neighbor)
            dp_avg += dp_neighbor

            counter_neighbors += 1
        end
    end

    dp_avg / counter_neighbors

    return dp_min, dp_max, dp_avg
end

@inline function particle_spacing(system, particle)
    return particle_spacing(system, system.particle_refinement, particle)
end

@inline particle_spacing(system, ::Nohing, _) = system.initial_condition.particle_spacing

@inline function particle_spacing(system, refinement, particle)
    (; smoothing_length_factor) = system.cache
    return smoothing_length(system, particle) / smoothing_length_factor
end

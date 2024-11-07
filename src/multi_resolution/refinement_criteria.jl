abstract type RefinementCriteria end

struct SpatialRefinementCriterion <: RefinementCriteria end

struct SolutionRefinementCriterion <: RefinementCriteria end

@inline function (criterion::SpatialRefinementCriterion)(system, semi, v_ode, u_ode)
    u = wrap_u(u_ode, system, semi)
    system_coords = current_coordinates(u, system)

    foreach_system(semi) do neighbor_system
        set_particle_spacing!(system, neighbor_system, system_coords)
    end
    return system
end

@inline set_particle_spacing!(system, neighbor_system, system_coords) = system

@inline function set_particle_spacing!(particle_system,
                                       neighbor_system::Union{BoundarySPHSystem,
                                                              TotalLagrangianSPHSystem},
                                       system_coords)
    (; smoothing_length, smoothing_length_factor) = particle_system.cache

    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords,
                           neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        dp_neighbor = particle_spacing(neighbor_system, neighbor)
        dp_particle = smoothing_length[particle] / smoothing_length_factor

        smoothing_length[particle] = smoothing_length_factor * min(dp_neighbor, dp_particle)
    end

    return particle_system
end

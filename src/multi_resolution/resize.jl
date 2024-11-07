function resize!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    copyto!(_v_ode, v_ode)
    copyto!(_u_ode, u_ode)

    # Resize all systems
    foreach_system(semi) do system
        resize!(system, capacity(system))
    end

    ranges_v_old = semi.ranges_v
    ranges_u_old = semi.ranges_u

    ranges_v_new, ranges_u_new = ranges_vu(semi.systems)

    capacity_global = sum(system -> capacity(system), semi.systems)

    resize!(v_ode, capacity_global)
    resize!(u_ode, capacity_global)

    for i in eachindex(ranges_u_old)
        old_length_u = length(ranges_u_old[i])
        for j in 1:old_length_u
            u_ode[ranges_u_new[i][1] + j] = _u_ode[ranges_u_old[i][1] + j]
        end

        old_length_v = length(ranges_v_old[i])
        for j in 1:old_length_v
            v_ode[ranges_v_new[i][1] + j] = _v_ode[ranges_v_old[i][1] + j]
        end
    end

    # Set ranges after resizing the systems
    for i in 1:length(semi.systems)
        semi.ranges_v[i] = ranges_v_new[i]
        semi.ranges_u[i] = ranges_u_old[i]
    end

    resize!(_v_ode, capacity_global)
    resize!(_u_ode, capacity_global)

    # TODO: Do the following in the callback
    # resize!(integrator, (length(v_ode), length(u_ode)))

    # # Tell OrdinaryDiffEq that u has been modified
    # u_modified!(integrator, true)

    return semi
end

resize!(system, capacity_system) = system

function resize!(system::FluidSystem, capacity_system)
    return resize!(system, system.particle_refinement, capacity_system)
end

resize!(system, ::Nothing, capacity_system) = system

function resize!(system::WeaklyCompressibleSPHSystem, refinement, capacity_system::Int)
    (; mass, pressure, cache, density_calculator) = system

    refinement.n_particles_before_resize = nparticles(system)

    resize!(mass, capacity_system)
    resize!(pressure, capacity_system)
    resize_density!(system, capacity_system, density_calculator)
    resize_cache!(system, cache, n)
end

function resize!(system::EntropicallyDampedSPHSystem, refinement, capacity_system::Int)
    (; mass, cache, density_calculator) = system

    refinement.n_particles_before_resize = nparticles(system)

    resize!(mass, capacity_system)
    resize_density!(system, capacity_system, density_calculator)
    resize_cache!(system, cache, capacity_system)

    return system
end

resize_density!(system, n::Int, ::SummationDensity) = resize!(system.cache.density, n)
resize_density!(system, n::Int, ::ContinuityDensity) = system

function resize_cache!(system, n::Int)
    resize!(system.cache.smoothing_length, n)

    return system
end

function resize_cache!(system::EntropicallyDampedSPHSystem, n)
    resize!(system.cache.smoothing_length, n)
    resize!(system.cache.pressure_average)
    resize!(system.cache.neighbor_counter)

    return system
end

@inline capacity(system) = capacity(system, system.particle_refinement)

@inline capacity(system, ::Nothing) = nparticles(system)

@inline function capacity(system, particle_refinement)
    return particle_refinement.n_new_particles + nparticles(system)
end

function deleteat!(semi::Semidiscretization, v_ode, u_ode, _v_ode, _u_ode)
    copyto!(_v_ode, v_ode)
    copyto!(_u_ode, u_ode)

    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        deleteat!(system, v, u)
    end

    ranges_v_old = semi.ranges_v
    ranges_u_old = semi.ranges_u

    ranges_v_new, ranges_u_new = ranges_vu(semi.systems)

    for i in eachindex(ranges_u_old)
        new_length_u = length(ranges_u_new[i])
        for j in 1:new_length_u
            u_ode[ranges_u_new[i][1] + j] = _u_ode[ranges_u_old[i][1] + j]
        end

        new_length_v = length(ranges_v_new[i])
        for j in 1:new_length_v
            v_ode[ranges_v_new[i][1] + j] = _v_ode[ranges_v_old[i][1] + j]
        end
    end

    capacity_global = sum(system -> nparticles(system), semi.systems)

    resize!(v_ode, capacity_global)
    resize!(u_ode, capacity_global)

    # Set ranges after resizing the systems
    for i in 1:length(semi.systems)
        semi.ranges_v[i] = ranges_v_new[i]
        semi.ranges_u[i] = ranges_u_old[i]
    end

    resize!(_v_ode, capacity_global)
    resize!(_u_ode, capacity_global)

    # TODO: Do the following in the callback
    # resize!(integrator, (length(v_ode), length(u_ode)))

    # # Tell OrdinaryDiffEq that u has been modified
    # u_modified!(integrator, true)
end

deleteat!(system, v, u) = system

function deleteat!(system::FluidSystem, v, u)
    return deleteat!(system, system.particle_refinement, v, u)
end

deleteat!(system, ::Nothing, v, u) = system

function deleteat!(system::WeaklyCompressibleSPHSystem, refinement, v, u)
    (; mass, pressure, cache, density_calculator) = system
    (; delete_candidates) = refinement


    delete_counter = 0

    for particle in eachparticl(system)
        if !iszero(delete_candidates[particle])
            # swap particles (keep -> delete)
            dump_id = nparticles(system) - delete_counter

            vel_keep = current_velocity(v, system, dump_id)
            pos_keep = current_coords(u, system, dump_id)

            mass_keep = hydrodynamic_mass(system, dump_id)
            density_keep = particle_density(system, v, dump_id)
            pressure_keep = particle_pressure(system, v, dump_id)
            smoothing_length_keep = smoothing_length(system, dump_id)

            system.mass[particle] = mass_keep
            system.cache.smoothing_length[particle] = smoothing_length_keep

            set_particle_pressure!(system, v, particle, pressure_keep)

            set_particle_density!(system, v, particle, density_keep)

            for dim in 1:ndims(system)
                v[dim, particle] = vel_keep[dim]
                u[dim, particle] = pos_keep[dim]
                v[dim, ]
            end

            delete_counter += 1
        end
    end

    resize!(system, nparticles(system) - delete_counter)

    return system
end

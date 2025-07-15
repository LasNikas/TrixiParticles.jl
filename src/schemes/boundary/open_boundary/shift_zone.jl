struct ShiftZone{SF, BC, NB}
    shift_factor         :: SF
    boundary_coordinates :: BC
    nhs_boundary         :: NB
end

function ShiftZone(; shift_factor, boundary_coordinates, initial_condition, fluid_system)
    min_corner = minimum(boundary_coordinates, dims=2)
    max_corner = maximum(boundary_coordinates, dims=2)
    cell_list = FullGridCellList(; min_corner, max_corner)

    NDIMS = ndims(initial_condition)
    nhs = GridNeighborhoodSearch{NDIMS}(; cell_list, update_strategy=ParallelUpdate())

    nhs_boundary = copy_neighborhood_search(nhs,
                                            compact_support(fluid_system, fluid_system),
                                            nparticles(initial_condition))

    return ShiftZone(shift_factor, boundary_coordinates, nhs_boundary)
end

initialize_shift_zone!(shift_zone, system) = shift_zone

function initialize_shift_zone!(shift_zone::ShiftZone, system)
    (; nhs_boundary, boundary_coordinates) = shift_zone

    PointNeighbors.initialize!(nhs_boundary, initial_coordinates(system),
                               boundary_coordinates)

    return shift_zone
end

apply_shifting!(dv_ode, v_ode, u_ode, system, shift_zone, semi) = dv_ode

function apply_shifting!(dv_ode, v_ode, u_ode, system, shift_zone::ShiftZone, semi)
    (; fluid_system) = system
    (; shift_factor, boundary_coordinates, nhs_boundary) = shift_zone

    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)
    v_fluid = wrap_v(v_ode, fluid_system, semi)
    u_fluid = wrap_u(u_ode, fluid_system, semi)

    set_zero!(dv)

    system_coords = current_coordinates(u, system)
    fluid_coords = current_coordinates(u_fluid, fluid_system)

    nhs = get_neighborhood_search(fluid_system, system, semi)

    # Open boundary
    PointNeighbors.foreach_point_neighbor(system_coords, system_coords, nhs;
                                          parallelization_backend=semi.parallelization_backend,
                                          points=each_moving_particle(system)) do particle,
                                                                                  neighbor,
                                                                                  pos_diff,
                                                                                  distance
        m_a = hydrodynamic_mass(system, particle)
        rho_a = system.initial_condition.density[particle]
        m_b = hydrodynamic_mass(system, neighbor)
        rho_b = system.initial_condition.density[neighbor]

        V_a = m_a / rho_a
        V_b = m_b / rho_b

        apply_packing!(dv, system, particle, m_a, V_a, V_b, pos_diff, distance,
                       shift_factor)
    end

    # Fluid
    nhs_fluid = get_neighborhood_search(fluid_system, fluid_system, semi)
    PointNeighbors.foreach_point_neighbor(system_coords, fluid_coords, nhs_fluid;
                                          parallelization_backend=semi.parallelization_backend,
                                          points=each_moving_particle(system)) do particle,
                                                                                  neighbor,
                                                                                  pos_diff,
                                                                                  distance
        m_a = hydrodynamic_mass(system, particle)
        rho_a = system.initial_condition.density[particle]
        m_b = hydrodynamic_mass(fluid_system, neighbor)
        rho_b = fluid_system.initial_condition.density[neighbor]

        V_a = m_a / rho_a
        V_b = m_b / rho_b

        apply_packing!(dv, system, particle, m_a, V_a, V_b, pos_diff, distance,
                       shift_factor)
    end

    # Boundary
    PointNeighbors.update!(nhs_boundary, system_coords, boundary_coordinates;
                           points_moving=(true, false),
                           parallelization_backend=semi.parallelization_backend)
    PointNeighbors.foreach_point_neighbor(system_coords, boundary_coordinates, nhs_boundary;
                                          parallelization_backend=semi.parallelization_backend,
                                          points=each_moving_particle(system)) do particle,
                                                                                  neighbor,
                                                                                  pos_diff,
                                                                                  distance
        m_a = hydrodynamic_mass(system, particle)
        rho_a = system.initial_condition.density[particle]

        V_a = m_a / rho_a

        apply_packing!(dv, system, particle, m_a, V_a, V_a, pos_diff, distance,
                       shift_factor)
    end

    return dv_ode
end

function apply_packing!(dv, system, particle, m_a, V_a, V_b, pos_diff, distance,
                        shift_factor)
    grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

    # This vanishes for uniform particle distributions
    dv_repulsive_pressure = -(2 / m_a) * V_a * V_b * shift_factor * grad_kernel

    for i in 1:ndims(system)
        dv[i, particle] += dv_repulsive_pressure[i]
    end

    return dv
end

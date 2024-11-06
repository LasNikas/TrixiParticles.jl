struct ParticleRefinement{SP}
    splitting_pattern :: SP
end



function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, t)
    # check refnement criteria

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

function split_particles!(semi::Semidiscretization, v_ode, u_ode)
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        split_particles!(system, v, u)
    end

    return semi
end

@inline split_particles!(system, v, u) = System

@inline function split_particles!(system::FluidSystem, v, u)
    return split_particles!(system, system.particle_refinement, v, u)
end

@inline split_particles!(system::FluidSystem, ::Nothing, v, u) = system

@inline function split_particles!(system::FluidSystem, particle_refinement, v, u)
    (; mass_ref, max_spacing_ratio, refinement_pattern) = particle_refinement

    split_candidates .= false

    for particle in eachparticle(system)
        m_a = hydrodynamic_mass(system, particle)
        m_max = max_spacing_ratio * mass_ref[particle]

        if m_a > m_max
            split_candidates[particle] = true
        end
    end

    n_childs_exclude_one = nchilds(system, refinement_pattern) - 1
    n_new_particles = sum(split_candidates) * n_childs_exclude_one
end

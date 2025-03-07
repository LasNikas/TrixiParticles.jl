using TrixiParticles
using OrdinaryDiffEq

filename = "circle"
file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", filename * ".asc")

# ==========================================================================================
# ==== Packing parameters
save_intervals = false
tlsph = true
pack_boundary = true

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.03

# The following depends on the sampling of the particles. In this case `boundary_thickness`
# means literally the thickness of the boundary packed with boundary particles and *not*
# how many rows of boundary particles will be sampled.
boundary_thickness = 8particle_spacing

# ==========================================================================================
# ==== Load complex geometry
density = 1.0

geometry = load_geometry(file)

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

point_in_geometry_algorithm = WindingNumberJacobson(; geometry,
                                                    winding_number_factor=sqrt(eps()),
                                                    hierarchical_winding=true)
# Returns `InitialCondition`
shape_sampled = ComplexShape(geometry; particle_spacing, density, grid_offset=0.0,
                             point_in_geometry_algorithm)

shape_sampled.mass .= density * TrixiParticles.volume(geometry) /
                      nparticles(shape_sampled)

if pack_boundary
    # Returns `InitialCondition`
    boundary_sampled = sample_boundary(signed_distance_field; boundary_density=density,
                                       boundary_thickness, tlsph)
    boundary_sampled.mass .= first(shape_sampled.mass)
end

# ==========================================================================================
# ==== Packing

# Large `background_pressure` can cause high accelerations. That is, the adaptive
# time-stepsize will be adjusted properly.
background_pressure = 1.0

smoothing_kernel = SchoenbergQuinticSplineKernel{ndims(geometry)}()
smoothing_length = 1.0 * particle_spacing

packing_system = ParticlePackingSystem(shape_sampled;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       smoothing_length_interpolation=smoothing_length,
                                       signed_distance_field, tlsph=tlsph,
                                       background_pressure)
if pack_boundary
    boundary_system = ParticlePackingSystem(boundary_sampled;
                                            smoothing_kernel=smoothing_kernel,
                                            smoothing_length=smoothing_length,
                                            smoothing_length_interpolation=smoothing_length,
                                            boundary_compress_factor=1.0,
                                            is_boundary=true, signed_distance_field,
                                            tlsph=tlsph, background_pressure)
end

# ==========================================================================================
# ==== Simulation
semi = pack_boundary ? Semidiscretization(packing_system, boundary_system) :
       Semidiscretization(packing_system)

# Use a high `tspan` to guarantee that the simulation runs at least for `maxiters`
tspan = (0, 10000.0)
ode = semidiscretize(semi, tspan)

# Use this callback to stop the simulation when it is sufficiently close to a steady state
steady_state = SteadyStateReachedCallback(; interval=1, interval_size=10,
                                          abstol=1.0e-5, reltol=1.0e-3)

info_callback = InfoCallback(interval=50)

saving_callback = save_intervals ?
                  SolutionSavingCallback(interval=10, prefix="", ekin=kinetic_energy,
                                         output_directory="out") :
                  nothing

pp_callback = nothing

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback, steady_state,
                        pp_callback)
maxiters = 1000
time_integrator = RDPK3SpFSAL35()

dtmax = nothing
if dtmax isa Number
    sol = solve(ode, time_integrator;
                abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
                # reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
                dtmax=dtmax,
                save_everystep=false, maxiters=maxiters, callback=callbacks)
else
    sol = solve(ode, time_integrator;
                abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
                # reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
                #dtmax=set_dtmax,
                save_everystep=false, maxiters=maxiters, callback=callbacks)
end

packed_ic = InitialCondition(sol, packing_system, semi)
packed_boundary_ic = pack_boundary ? InitialCondition(sol, boundary_system, semi) : nothing

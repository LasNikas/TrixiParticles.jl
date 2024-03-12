# Lid-driven cavity
#
# S. Adami et al
# "A transport-velocity formulation for smoothed particle hydrodynamics".
# In: Journal of Computational Physics, Volume 241 (2013), pages 292-307.
# https://doi.org/10.1016/j.jcp.2013.01.043

using TrixiParticles
using OrdinaryDiffEq

wcsph = true

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.02

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 30.0)
reynolds_number = 100.0

cavity_size = (1.0, 1.0)

fluid_density = 1.0

const velocity_lid = 1.0
sound_speed = 10 * velocity_lid

viscosity = ViscosityAdami(; nu=velocity_lid / reynolds_number)

pressure = sound_speed^2 * fluid_density

cavity = RectangularTank(particle_spacing, cavity_size, cavity_size, fluid_density,
                         n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                         faces=(true, true, true, false), pressure=pressure)

lid_position = 0.0 - particle_spacing * boundary_layers
lid_length = cavity.n_particles_per_dimension[1] + 2boundary_layers

lid = RectangularShape(particle_spacing, (lid_length, 3),
                       (lid_position, cavity_size[2]), density=fluid_density)

# ==========================================================================================
# ==== Fluid

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

density_calculator = SummationDensity()

if wcsph
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=1)
    fluid_system = WeaklyCompressibleSPHSystem(cavity.fluid, density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity=viscosity,
                                               transport_velocity=TransportVelocityAdami(pressure))
else
    state_equation = nothing
    pressure_acceleration = TrixiParticles.inter_particle_averaged_pressure
    fluid_system = EntropicallyDampedSPHSystem(cavity.fluid, smoothing_kernel,
                                               smoothing_length, sound_speed;
                                               pressure_acceleration,
                                               density_calculator=density_calculator,
                                               viscosity=viscosity,
                                               transport_velocity=TransportVelocityAdami(pressure))
end
# ==========================================================================================
# ==== Boundary

movement_function(t) = SVector(velocity_lid * t, 0.0)

is_moving(t) = true

movement = BoundaryMovement(movement_function, is_moving)

boundary_model_cavity = BoundaryModelDummyParticles(cavity.boundary.density,
                                                    cavity.boundary.mass,
                                                    AdamiPressureExtrapolation(),
                                                    viscosity=viscosity,
                                                    state_equation=state_equation,
                                                    smoothing_kernel, smoothing_length)

boundary_model_lid = BoundaryModelDummyParticles(lid.density, lid.mass,
                                                 AdamiPressureExtrapolation(),
                                                 viscosity=viscosity,
                                                 state_equation=state_equation,
                                                 smoothing_kernel, smoothing_length)

boundary_system_cavity = BoundarySPHSystem(cavity.boundary, boundary_model_cavity)

boundary_system_lid = BoundarySPHSystem(lid, boundary_model_lid, movement=movement)

# ==========================================================================================
# ==== Simulation
bnd_thickness = boundary_layers * particle_spacing
semi = Semidiscretization(fluid_system, boundary_system_cavity, boundary_system_lid,
                          neighborhood_search=GridNeighborhoodSearch,
                          periodic_box_min_corner=[-bnd_thickness, -bnd_thickness],
                          periodic_box_max_corner=cavity_size .+
                                                  [bnd_thickness, bnd_thickness])

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

saving_callback = SolutionSavingCallback(dt=0.02, prefix="", ekin=kinetic_energy)
steady_state = SteadyStateCallback(abstol=1e-8, reltol=1e-6, interval_size=1000)
callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(), steady_state)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

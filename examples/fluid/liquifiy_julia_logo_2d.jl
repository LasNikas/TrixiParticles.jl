using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution

particle_spacing = 0.01

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 5.0)

fluid_density = 1000.0
solid_density = 300.0

# Young's modulus and Poisson ratio
E = 1e6
nu = 0.0

sound_speed = 20 * sqrt(gravity * 2.0)

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "preprocessing", "liquify_julia_logo_2d.jl"),
              particle_spacing=particle_spacing, fluid_density=fluid_density,
              solid_density=solid_density, scale_shape=0.1)

tank = RectangularTank(particle_spacing, (0.0, 0.0), (5.0, 3.0),
                       fluid_density; boundary_density=fluid_density,
                       n_layers=4, min_coordinates=[0.0, -2.8],
                       faces=(true, true, true, false))
# ==========================================================================================
# ==== Fluid
smoothing_length = 3.0 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

alpha = 0.02

fluid_density_calculator = ContinuityDensity()
# viscosity = ArtificialViscosityMonaghan(alpha=alpha, beta=0.0)
# fluid_system = WeaklyCompressibleSPHSystem(letters, fluid_density_calculator,
#                                            state_equation, smoothing_kernel,
#                                            smoothing_length, viscosity=viscosity,
#                                            acceleration=(0.0, -gravity))
viscosity = ViscosityAdami(nu=alpha * smoothing_length * sound_speed / 8)
fluid_system = EntropicallyDampedSPHSystem(letters, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           density_calculator=fluid_density_calculator,
                                           acceleration=(0.0, -gravity))
# ==========================================================================================
# ==== Solid
solid_smoothing_length = 2 * sqrt(2) * particle_spacing
solid_smoothing_kernel = WendlandC2Kernel{2}()

solid_systems = Vector{TotalLagrangianSPHSystem}()
for i in eachindex(file_names[1:4])
    ic = initial_conditions[i]
    hydrodynamic_densites = fluid_density * ones(size(ic.density))
    hydrodynamic_masses = hydrodynamic_densites * particle_spacing^ndims(fluid_system)

    solid_boundary_model = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                       hydrodynamic_masses,
                                                       state_equation=state_equation,
                                                       AdamiPressureExtrapolation(),
                                                       smoothing_kernel,
                                                       smoothing_length)

    solid_system = TotalLagrangianSPHSystem(ic,
                                            solid_smoothing_kernel,
                                            solid_smoothing_length,
                                            E, nu,
                                            acceleration=(0.0, -gravity),
                                            boundary_model=solid_boundary_model,
                                            penalty_force=PenaltyForceGanzenmueller(alpha=0.3))
    push!(solid_systems, solid_system)
end

# ==========================================================================================
# ==== Boundary
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             #viscosity=ViscosityAdami(nu=1e-4),
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, solid_systems...)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="", output_directory="out_julia_logo_2d_dp_$particle_spacing")

callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

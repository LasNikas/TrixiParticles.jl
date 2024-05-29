using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.02

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

tank_size = (150, 3)

fluid_size = (40, 15)

# Boundary geometry and initial fluid particle positions

fluid_density = 1000.0
boundary_density = 7800.0 #not used in this example
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

tank_bottom = RectangularShape(particle_spacing, tank_size,
                               (0.0, 0.0), density=fluid_density)
tank_top = RectangularShape(particle_spacing, tank_size,
                            (0.0, (fluid_size[2] + tank_size[2]) * particle_spacing),
                            density=fluid_density)
tank_left = RectangularShape(particle_spacing, (tank_size[2], fluid_size[2]),
                             (0.0, tank_size[2] * particle_spacing),
                             density=fluid_density)
tank_right = RectangularShape(particle_spacing, (tank_size[2], fluid_size[2]),
                              ((tank_size[1] - tank_size[2]) * particle_spacing,
                               tank_size[2] * particle_spacing),
                              density=fluid_density)

tank = union(tank_bottom, tank_top, tank_left, tank_right)

fluid = RectangularShape(particle_spacing, fluid_size,
                         (5 * particle_spacing,
                          tank_size[2] * particle_spacing),
                         density=fluid_density, velocity=(2, 0.0))

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_density_calculator = ContinuityDensity()
fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity),
                                           source_terms=nothing)

# ==========================================================================================
# ==== Boundary

# This is to set another boundary density calculation with `trixi_include`
boundary_density_calculator = AdamiPressureExtrapolation()

# This is to set wall viscosity with `trixi_include`
viscosity_wall = nothing
boundary_model = BoundaryModelDummyParticles(tank.density,
                                             tank.mass,
                                             boundary_density_calculator, smoothing_kernel,
                                             smoothing_length; viscosity=nothing,
                                             state_equation=state_equation,
                                             correction=nothing)

boundary_system = BoundarySPHSystem(tank, boundary_model,
                                    movement=nothing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="",
                                         output_directory="out_my_simulation_1")

# This is to easily add a new callback with `trixi_include`
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, extra_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks);

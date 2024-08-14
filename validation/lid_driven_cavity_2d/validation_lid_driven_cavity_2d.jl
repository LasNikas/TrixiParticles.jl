using TrixiParticles

# ==========================================================================================
# ==== Resolution
particle_spacings = [0.02, 0.01, 0.005]

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 30.0)
reynolds_numbers = [100.0, 1000.0, 10_000.0]

vx_y(v, u, t, system) = nothing
function vx_y(v, u, t, system::TrixiParticles.FluidSystem)
    n_particles_xy = round(Int, 1.0 / system.initial_condition.particle_spacing)

    values = interpolate_line([0.5, 0.0], [0.5, 1.0], n_particles_xy, semi, system,
                              v, u; endpoint=true, cut_off_bnd=true)

    return stack(values.velocity)[1, :]
end

vy_x(v, u, t, system) = nothing
function vy_x(v, u, t, system::TrixiParticles.FluidSystem)
    n_particles_xy = round(Int, 1.0 / system.initial_condition.particle_spacing)

    values = interpolate_line([0.0, 0.5], [1.0, 0.5], n_particles_xy, semi, system,
                              v, u; endpoint=true, cut_off_bnd=true)

    return stack(values.velocity)[2, :]
end

for particle_spacing in particle_spacings, reynolds_number in reynolds_numbers
    n_particles_xy = round(Int, 1.0 / particle_spacing)

    Re = Int(reynolds_number)

    output_directory = joinpath("out_ldc",
                                "validation_run_lid_driven_cavity_2d_nparticles_$(n_particles_xy)x$(n_particles_xy)_Re_$Re")

    saving_callback = SolutionSavingCallback(dt=0.1, output_directory=output_directory)

    info_callback = InfoCallback(interval=500)

    ekin_cb = PostprocessCallback(; dt=0.02, kinetic_energy, write_file_interval=10,
                                  output_directory=output_directory,
                                  filename="kinetic_energy")

    interval_size = round(Int, 0.2 / particle_spacing)

    steady_state = SteadyStateCallback(; dt=0.04, interval_size=interval_size,
                                       abstol=1.0e-8, reltol=1.0e-6)

    pos = collect(LinRange(0.0, 1.0, n_particles_xy))

    table_data = TableDataSavingCallback(; dt=0.02, save_interval=10,
                                         write_file_interval=10,
                                         start_at=5.0, # condition is `true` if `t >= start_at`
                                         axis_ticks=Dict([vx_y => pos, vy_x => pos]),
                                         output_directory=output_directory,
                                         vx_y=vx_y, vy_x=vy_x)

    # Import variables into scope
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fluid", "lid_driven_cavity_2d.jl"),
                  saving_callback=saving_callback, tspan=tspan,
                  callbacks=(info_callback, saving_callback, UpdateCallback(),
                             ekin_cb, table_data, steady_state),
                  particle_spacing=particle_spacing,
                  reynolds_number=reynolds_number)
end

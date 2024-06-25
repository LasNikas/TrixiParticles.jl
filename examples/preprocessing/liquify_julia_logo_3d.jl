using TrixiParticles

particle_spacing = 0.05
scale_shape = 0.1
fluid_density = 1000.0
solid_density = 300.0

dir = joinpath("examples", "preprocessing", "julia_logo")
hierarchical_winding = true

file_names = [
    "dot_1",
    "dot_2",
    "dot_3",
    "dot_4",
    "letters"
]

# Load shapes
shapes = Vector{TrixiParticles.Shapes}()
for file in file_names
    push!(shapes, load_shape(joinpath(dir, file * ".stl")))
end

# Sample shapes
initial_conditions = Vector{TrixiParticles.InitialCondition}()
for i in eachindex(file_names)
    density = file_names[i] == "letters" ? fluid_density : solid_density
    point_in_shape_algorithm = WindingNumberJacobson(; shape=shapes[i],
                                                     winding_number_factor=0.2,
                                                     hierarchical_winding)

    push!(initial_conditions,
          ComplexShape(shapes[i]; particle_spacing, density, max_nparticles=1e7,
                       point_in_shape_algorithm, seed=shapes[i].min_corner .- sqrt(eps())))

    # write to vtk
    trixi2vtk(initial_conditions[i], filename="initial_condition_" * file_names[i])
end

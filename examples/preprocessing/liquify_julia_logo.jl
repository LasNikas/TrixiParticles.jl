using TrixiParticles

particle_spacing = 0.2
fluid_density = 1000.0
solid_density = 300.0

#dir = joinpath(expanduser("~/"), "Data", "stl-files", "julia_logo")
hierarchical_winding=false
dir = joinpath("examples", "preprocessing", "julia_logo")

file_names = [
    "dot_1",
    "dot_2",
    "dot_3",
    "dot_4",
    "j",
    "u",
    "l",
    "i",
    "a"
]

# Load shapes
shapes = Vector{TrixiParticles.Shapes}()
for file in file_names
    #push!(shapes, load_shape(joinpath(dir, file * ".stl")))
    push!(shapes, load_shape(joinpath(dir, file * ".asc")))
end

# Sample shapes
initial_conditions = Vector{TrixiParticles.InitialCondition}()
for i in eachindex(file_names)
    density = length(file_names[i]) == 1 ? fluid_density : solid_density
    point_in_shape_algorithm = WindingNumberJacobson(; shape=shapes[i],
                                                     winding_number_factor=0.2,
                                                     hierarchical_winding)

    push!(initial_conditions,
          ComplexShape(shapes[i]; particle_spacing, density, max_nparticles=1e7,
                       point_in_shape_algorithm))

    # write to vtk
    trixi2vtk(initial_conditions[i], filename="initial_condition_" * file_names[i])
end

letters = union(initial_conditions[5:end]...)

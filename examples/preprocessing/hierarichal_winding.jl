using TrixiParticles

particle_spacing = 0.5

filename_box = joinpath("out_preprocessing", "unit_box.asc")
filename_shape = joinpath("out_preprocessing", "test_hexagon.asc")

box = load_shape(filename_box)
shape = load_shape(filename_shape)

trixi2vtk(stack([box.edge_vertices[i][1] for i in 1:length(box.edge_vertices)]),
          filename="points_box")
trixi2vtk(stack([shape.edge_vertices[i][1] for i in 1:length(shape.edge_vertices)]),
          filename="points_shape")

edge_shape = 5
edge_box = 3
edge1 = shape.edge_vertices[edge_shape]
edge2 = box.edge_vertices[edge_box]

TrixiParticles.exterior_vertices(box, shape)

dir = joinpath("Data", "stl-files", "examples")
filename_cube = joinpath(expanduser("~/") * dir, "skull.stl")
filename_cube_shifted = joinpath(expanduser("~/") * dir, "shifted_skull.stl")

# Returns `Shape`
cube = load_shape(filename_cube)
@btime shifted_cube = load_shape(filename_cube_shifted)

points = TrixiParticles.exterior_vertices(cube, shifted_cube)
trixi2vtk(stack(points), filename="points")

using TrixiParticles

particle_spacing = 0.02

dir = joinpath("Data", "stl-files", "examples")
filename = joinpath(expanduser("~/") * dir, "aorta.stl")

# Returns `Shape`
shape = load_shape(filename)

min_corner = shape.min_box
max_corner = shape.max_box

# Bisect the box splitting its longest side
# box_edges = max_corner - min_corner
#
# sizes_sorted_index = sortperm(box_edges, rev=true)

grid = TrixiParticles.particle_grid(shape, particle_spacing; pad=2particle_spacing,
                                    seed=nothing, max_nparticles=Int(1e8))

bbox = TrixiParticles.BoundingBoxTree(TrixiParticles.eachface(shape),
                                      min_corner, max_corner)

directed_edges = zeros(Int, 1:length(shape.normals_edge))
TrixiParticles.construct_hierarchy!(bbox, shape, directed_edges);

point_in_shape_algorithm_hier = WindingNumberJacobson(; #winding_number_factor=0.1,
                                                      winding=TrixiParticles.HierarchicalWinding(bbox));

point_in_shape_algorithm_naive = WindingNumberJacobson();

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0, max_nparticles=1e8,
                             point_in_shape_algorithm=point_in_shape_algorithm_hier)

trixi2vtk(shape_sampled.coordinates, filename="coords")

# Returns `InitialCondition`.
#shape_sampled_naive = ComplexShape(shape; particle_spacing, density=1.0, max_nparticles=1e8,
#                                   point_in_shape_algorithm=point_in_shape_algorithm_naive)
#
#trixi2vtk(shape_sampled_naive.coordinates, filename="coords_naive")

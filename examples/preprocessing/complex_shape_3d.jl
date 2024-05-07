using TrixiParticles

particle_spacing = 0.02

dir = joinpath("Data", "stl-files", "examples")
filename = joinpath(expanduser("~/") * dir, "aorta.stl")

# Returns `Shape`
shape = load_shape(filename)

min_corner = rationalize.(shape.min_box)#ceil.(shape.min_box) .-1)
max_corner = rationalize.(shape.max_box)#ceil.(shape.max_box) .+1)

# Bisect the box splitting its longest side
# box_edges = max_corner - min_corner
#
# sizes_sorted_index = sortperm(box_edges, rev=true)

bbox = TrixiParticles.BoundingBoxTree(collect(1:TrixiParticles.nfaces(shape)),
                                      min_corner, max_corner)

directed_edges = zeros(Int, 1:length(shape.normals_edge))
TrixiParticles.construct_hierarchy!(bbox, shape, directed_edges, 0, particle_spacing);

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0, max_nparticles=1e7,
                             #point_in_shape_algorithm=WindingNumberJacobson())
                             point_in_shape_algorithm=WindingNumberJacobson(;
                                                                            winding=TrixiParticles.HierarchicalWinding(bbox)))

trixi2vtk(shape_sampled.coordinates, filename="coords")
trixi2vtk(stack(shape.vertices), filename="points");

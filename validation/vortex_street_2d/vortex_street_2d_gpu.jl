# Flow past a circular cylinder (vortex street), Tafuni et al. (2018).
# Other literature using this validation:
# Vacandio et al. (2013), Marrone et al. (2013), Calhoun (2002), Liu et al. (1998)

using TrixiParticles

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "vortex_street_2d", "vortex_street_2d.jl"),
              sol=nothing, ode=nothing)

# Define a GPU-compatible neighborhood search
min_corner = minimum(pipe.boundary.coordinates .- particle_spacing, dims=2)
max_corner = maximum(pipe.boundary.coordinates .+ particle_spacing, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                update_strategy=ParallelUpdate())

trixi_include_changeprecision(Float32,
                              joinpath(validation_dir(), "vortex_street_2d",
                                       "vortex_street_2d.jl"),
                              parallelization_backend=MetalBackend())

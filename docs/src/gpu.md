# GPU Support

GPU support is still an experimental feature that is actively being worked on.
As of now, the [`WeaklyCompressibleSPHSystem`](@ref) and the [`BoundarySPHSystem`](@ref)
are supported on GPUs.
We have tested this on GPUs by Nvidia and AMD.

To run a simulation on a GPU, we need to use the [`FullGridCellList`](@ref)
as cell list for the [`GridNeighborhoodSearch`](@ref).
This cell list requires a bounding box for the domain, unlike the default cell list, which
uses an unbounded domain.
For simulations that are bounded by a closed tank, we can use the boundary of the tank
to obtain the bounding box as follows.
```jldoctest gpu; output=false, setup=:(using TrixiParticles; trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing))
search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
min_corner = minimum(tank.boundary.coordinates, dims=2) .- search_radius
max_corner = maximum(tank.boundary.coordinates, dims=2) .+ search_radius
cell_list = TrixiParticles.PointNeighbors.FullGridCellList(; min_corner, max_corner)

# output
PointNeighbors.FullGridCellList{PointNeighbors.DynamicVectorOfVectors{Int32, Matrix{Int32}, Vector{Int32}, Base.RefValue{Int32}}, Nothing, SVector{2, Float64}, SVector{2, Float64}}(Vector{Int32}[], nothing, [-0.24500000000000002, -0.24500000000000002], [1.245, 1.245])
```

We then need to pass this cell list to the neighborhood search and the neighborhood search
to the [`Semidiscretization`](@ref).
```jldoctest gpu; output=false
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(; cell_list))

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Semidiscretization                                                                               │
│ ══════════════════                                                                               │
│ #spatial dimensions: ………………………… 2                                                                │
│ #systems: ……………………………………………………… 2                                                                │
│ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
│ total #particles: ………………………………… 636                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

At this point, we should run the simulation and make sure that it still works and that
the bounding box is large enough.
For some simulations where particles move outside the initial tank coordinates,
for example when the tank is not closed or when the tank is moving, an appropriate
bounding box has to be specified.

Then, we only need to specify the data type that is used for the simulation.
On an Nvidia GPU, we specify:
```julia
using CUDA
ode = semidiscretize(semi, tspan, data_type=CuArray)
```
On an AMD GPU, we use:
```julia
using AMDGPU
ode = semidiscretize(semi, tspan, data_type=ROCArray)
```
Then, we can run the simulation as usual.
All data is transferred to the GPU during initialization and all loops over particles
and their neighbors will be executed on the GPU as kernels generated by KernelAbstractions.jl.
Data is only copied to the CPU for saving VTK files via the [`SolutionSavingCallback`](@ref).

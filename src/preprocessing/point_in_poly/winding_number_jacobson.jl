struct NaiveWinding end

@inline function (winding::NaiveWinding)(polygon::Polygon{2}, query_point)
    (; edge_vertices) = polygon

    return naive_winding(polygon, edge_vertices, query_point)
end

@inline function (winding::NaiveWinding)(mesh::TriangleMesh{3}, query_point)
    (; face_vertices) = mesh

    return naive_winding(mesh, face_vertices, query_point)
end

@inline function naive_winding(polygon::Polygon{2}, edges, query_point)
    winding_number = sum(edges, init=zero(eltype(polygon))) do edge
        a = edge[1] - query_point
        b = edge[2] - query_point

        return atan(det([a b]), (dot(a, b)))
    end

    return winding_number
end

@inline function naive_winding(mesh::TriangleMesh{3}, faces, query_point)
    winding_number = sum(faces, init=zero(eltype(mesh))) do face

        # Eq. 6 of Jacobsen et al. Based on A. Van Oosterom (1983),
        # "The Solid Angle of a Plane Triangle" (doi: 10.1109/TBME.1983.325207)
        a = face_vertex(mesh, face, 1) - query_point
        b = face_vertex(mesh, face, 2) - query_point
        c = face_vertex(mesh, face, 3) - query_point
        a_ = norm(a)
        b_ = norm(b)
        c_ = norm(c)

        divisor = a_ * b_ * c_ + dot(a, b) * c_ + dot(b, c) * a_ + dot(c, a) * b_

        return 2 * atan(det([a b c]), divisor)
    end

    return winding_number
end

"""
    WindingNumberJacobson(; geometry=nothing, winding_number_factor=sqrt(eps()),
                          hierarchical_winding=false)
Algorithm for inside-outside segmentation of a complex geometry proposed by Jacobson et al. (2013).

# Keywords
- `geometry`: Complex geometry returned by [`load_geometry`](@ref) and is only required when using
              `hierarchical_winding=true`.
- `hierarchical_winding`: If set to `true`, an optimized hierarchical approach will be used
                          which gives a significant speedup.
                          It is only supported for 3D geometries yet.
- `winding_number_factor`: For leaky geometries, a factor of `0.4` will give a better inside-outside segmentation.

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct WindingNumberJacobson{ELTYPE, W}
    winding_number_factor :: ELTYPE
    winding               :: W

    function WindingNumberJacobson(; geometry=nothing, winding_number_factor=sqrt(eps()),
                                   hierarchical_winding=false)
        if hierarchical_winding && geometry isa Nothing
            throw(ArgumentError("`geometry` must be of type `Polygon` (2D) or `TriangleMesh` (3D) when using hierarchical winding"))
        end

        winding = hierarchical_winding ? HierarchicalWinding(geometry) : NaiveWinding()

        return new{typeof(winding_number_factor), typeof(winding)}(winding_number_factor,
                                                                   winding)
    end
end

function (point_in_poly::WindingNumberJacobson)(geometry, points; store_winding_number=false)
    (; winding_number_factor, winding) = point_in_poly

    inpoly = falses(size(points, 2))

    winding_numbers = Float64[]
    store_winding_number && (winding_numbers = resize!(winding_numbers, length(inpoly)))

    divisor = ndims(geometry) == 2 ? 2pi : 4pi

    @threaded points for query_point in axes(points, 2)
        p = point_position(points, geometry, query_point)

        winding_number = winding(geometry, p) / divisor

        store_winding_number && (winding_numbers[query_point] = winding_number)

        # Relaxed restriction of `(winding_number != 0.0)`
        if !(-winding_number_factor < winding_number < winding_number_factor)
            inpoly[query_point] = true
        end
    end

    return inpoly, winding_numbers
end

# The following functions distinguish between actual triangles and reconstructed triangles
# in the hierarchical winding approach.

# `face` holds the coordinates of each vertex
@inline face_vertex(mesh, face, index) = face[index]

# `face` holds the index of each vertex
@inline function face_vertex(mesh, face::NTuple{3, Int}, index)
    v_id = face[index]

    return mesh.vertices[v_id]
end

# `face` is the index of the face
@inline function face_vertex(mesh, face::Int, index)
    (; face_vertices) = mesh

    return face_vertices[face][index]
end
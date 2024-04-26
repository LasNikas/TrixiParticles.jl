
function exterior_vertices(box, shape::Shapes{2})
    (; edge_vertices) = shape

    boundary_vertices = Vector{SVector{ndims(shape), eltype(shape)}}()

    for edge in TrixiParticles.eachface(shape)
        v1 = edge_vertices[edge][1]
        v2 = edge_vertices[edge][2]

        inbox, P = clip_edge(v1, v2, box.min_box, box.max_box)

        inbox && push!(boundary_vertices, P...)
    end

    return boundary_vertices
end

function exterior_vertices(box, shape::Shapes)
    (; face_vertices) = shape

    boundary_vertices = Vector{SVector{ndims(shape), eltype(shape)}}()

    for face in TrixiParticles.eachface(shape)
        v1 = face_vertices[face][1]
        v2 = face_vertices[face][2]
        v3 = face_vertices[face][3]

        inbox, P = clip_edge(v1, v2, box.min_box, box.max_box)
        inbox && push!(boundary_vertices, P...)
        inbox, P = clip_edge(v2, v3, box.min_box, box.max_box)
        inbox && push!(boundary_vertices, P...)
        inbox, P = clip_edge(v3, v1, box.min_box, box.max_box)
        inbox && push!(boundary_vertices, P...)
    end

    return boundary_vertices
end

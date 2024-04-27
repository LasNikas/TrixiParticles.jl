struct TriangleMesh{NDIMS, ELTYPE} <: Shapes{NDIMS}
    vertices       :: Vector{SVector{NDIMS, ELTYPE}}
    face_vertices  :: Vector{Vector{SVector{NDIMS, ELTYPE}}}
    normals_vertex :: Vector{SVector{NDIMS, ELTYPE}}
    normals_edge   :: Vector{SVector{NDIMS, ELTYPE}}
    edges          :: Vector{SVector{2, Int}}
    normals_face   :: Vector{SVector{NDIMS, ELTYPE}}
    min_box        :: SVector{NDIMS, ELTYPE}
    max_box        :: SVector{NDIMS, ELTYPE}

    function TriangleMesh(mesh)
        NDIMS = length(first(mesh))
        n_faces = length(mesh)

        min_box = SVector([minimum(v[i] for v in mesh.position) for i in 1:NDIMS]...)
        max_box = SVector([maximum(v[i] for v in mesh.position) for i in 1:NDIMS]...)

        ELTYPE = eltype(min_box)

        vertices = union(mesh.position)

        face_vertices_ids = [SVector(0, 0, 0) for _ in 1:n_faces]

        face_vertices = [[fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:3]
                         for _ in 1:n_faces]

        normals_face = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:n_faces]

        edges = Vector{SVector{2, Int}}() # Store ids of vertices
        normals_edge = Vector{SVector{NDIMS, ELTYPE}}()
#=
        for i in 1:n_faces
            v1, v2, v3 = mesh[i]
            vertex_id1 = findfirst(x -> isapprox(v1, x), vertices)
            vertex_id2 = findfirst(x -> isapprox(v2, x), vertices)
            vertex_id3 = findfirst(x -> isapprox(v3, x), vertices)

            face_vertices_ids[i] = [vertex_id1, vertex_id2, vertex_id3]
        end

        not_hanging = union(stack(face_vertices_ids))

        deleteat!(vertices, setdiff(1:length(vertices), not_hanging)) =#

        normals_vertex = [fill(zero(ELTYPE), SVector{NDIMS}) for _ in 1:length(vertices)]

        for i in 1:n_faces
            v1, v2, v3 = mesh[i]

            face_vertices[i] = [SVector(v1...), SVector(v2...), SVector(v3...)]

            # Calculate normals
            n = SVector(normalize(cross(v2 - v1, v3 - v1))...)

            normals_face[i] = n

            vertex_id1 = findfirst(x -> isapprox(v1, x), vertices)
            vertex_id2 = findfirst(x -> isapprox(v2, x), vertices)
            vertex_id3 = findfirst(x -> isapprox(v3, x), vertices)

            face_vertices_ids[i] = [vertex_id1, vertex_id2, vertex_id3]

            normals_vertex[vertex_id1] += n
            normals_vertex[vertex_id2] += n
            normals_vertex[vertex_id3] += n

            edge_1 = SVector(vertex_id1, vertex_id2)
            edge_1_ = SVector(vertex_id2, vertex_id1)
            edge_2 = SVector(vertex_id2, vertex_id3)
            edge_2_ = SVector(vertex_id3, vertex_id2)
            edge_3 = SVector(vertex_id3, vertex_id1)
            edge_3_ = SVector(vertex_id1, vertex_id3)

            # 6 possible variations
            edge_id1 = findfirst(x -> (edge_1 == x || edge_1_ == x), edges)
            edge_id2 = findfirst(x -> (edge_2 == x || edge_2_ == x), edges)
            edge_id3 = findfirst(x -> (edge_3 == x || edge_3_ == x), edges)

            if isnothing(edge_id1)
                push!(edges, edge_1)
                push!(normals_edge, n)
            else
                normals_edge[edge_id1] += n
            end
            if isnothing(edge_id2)
                push!(edges, edge_2)
                push!(normals_edge, n)
            else
                normals_edge[edge_id2] += n
            end
            if isnothing(edge_id3)
                push!(edges, edge_3)
                push!(normals_edge, n)
            else
                normals_edge[edge_id3] += n
            end
        end

        return new{NDIMS, ELTYPE}(vertices, face_vertices, normalize.(normals_vertex),
                                  normalize.(normals_edge), edges,
                                  normals_face, min_box, max_box)
    end
end

@inline nfaces(mesh::TriangleMesh) = length(mesh.normals_face)

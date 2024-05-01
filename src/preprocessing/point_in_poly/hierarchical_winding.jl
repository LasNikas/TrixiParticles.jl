mutable struct BoundingBoxTree{MC}
    faces         :: Vector{Int}
    min_corner    :: MC
    max_corner    :: MC
    is_leaf       :: Bool
    closing_faces :: Vector{NTuple{3, Int}}
    left          :: BoundingBoxTree
    right         :: BoundingBoxTree

    function BoundingBoxTree(faces, min_corner, max_corner)
        return new{typeof(min_corner)}(faces, min_corner, max_corner, false)
    end
end

function construct_hierarchy!(bounding_box, mesh, directed_edges)
    (; max_corner, min_corner, faces) = bounding_box

    if length(faces) < 5
        bounding_box.is_leaf = true

        return bounding_box
    end

    bounding_box.closing_faces = determine_exterior(mesh, faces, directed_edges)

    if length(bounding_box.closing_faces) >= length(faces)
        bounding_box.is_leaf = true

        return bounding_box
    end

    # Bisect the box splitting its longest side
    box_edges = max_corner - min_corner

    split_direction = findfirst(x -> maximum(box_edges) == x, box_edges)

    uvec = (1:3) .== split_direction

    max_corner_left = max_corner - 0.5box_edges[split_direction] * uvec
    min_corner_right = min_corner + 0.5box_edges[split_direction] * uvec

    faces_left = in_bbox(mesh, faces, min_corner, max_corner_left)
    faces_right = in_bbox(mesh, faces, min_corner_right, max_corner)

    bbox_left = BoundingBoxTree(faces_left, min_corner, max_corner_left)
    bbox_right = BoundingBoxTree(faces_right, min_corner_right, max_corner)

    bounding_box.left = bbox_left
    bounding_box.right = bbox_right

    construct_hierarchy!(bbox_left, mesh, directed_edges)
    construct_hierarchy!(bbox_right, mesh, directed_edges)

    return bounding_box
end

function determine_exterior(mesh::Shapes{3}, faces, count_directed_edge)
    (; face_vertices_ids, face_edges_ids) = mesh

    count_directed_edge .= 0

    for face in faces
        v1 = face_vertices_ids[face][1]
        v2 = face_vertices_ids[face][2]
        v3 = face_vertices_ids[face][3]

        edge1 = face_edges_ids[face][1]
        edge2 = face_edges_ids[face][2]
        edge3 = face_edges_ids[face][3]

        # Keep track how many extra time each edge is seen
        # in the forward (`1`) and backward direction (`-1`)
        count_directed_edge[edge1] += v1 < v2 ? 1 : -1
        count_directed_edge[edge2] += v2 < v3 ? 1 : -1
        count_directed_edge[edge3] += v3 < v1 ? 1 : -1
    end

    closing_faces = Vector{Tuple{Int, Int, Int}}()
    closing_vertex = nothing

    # Determine vertex which defines an interior face
    for face in faces
        v1 = face_vertices_ids[face][1]

        edge1 = face_edges_ids[face][1]
        edge2 = face_edges_ids[face][2]
        edge3 = face_edges_ids[face][3]

        if count_directed_edge[edge1] == 0 &&
           count_directed_edge[edge2] == 0 &&
           count_directed_edge[edge3] == 0

            # Arbitrary vertex which lies on an interior face
            closing_vertex = v1

            break
        end
    end

    for face in faces
        v1 = face_vertices_ids[face][1]
        v2 = face_vertices_ids[face][2]
        v3 = face_vertices_ids[face][3]

        edge1 = face_edges_ids[face][1]
        edge2 = face_edges_ids[face][2]
        edge3 = face_edges_ids[face][3]

        if count_directed_edge[edge1] != 0
            # `edge1` is an exterior edge
            if count_directed_edge[edge1] > 0
                # These triangles are repeated |count| times to account for possible
                # multiple coverage of the same exterior edge.
                @inbounds for _ in 1:abs(count_directed_edge[edge1])
                    push!(closing_faces, (v1, v2, closing_vertex))
                end
            else
                @inbounds for _ in 1:abs(count_directed_edge[edge1])
                    push!(closing_faces, (v2, v1, closing_vertex))
                end
            end
        end

        if count_directed_edge[edge2] != 0
            # `edge2` is an exterior edge
            if count_directed_edge[edge2] > 0
                @inbounds for _ in 1:abs(count_directed_edge[edge2])
                    push!(closing_faces, (v2, v3, closing_vertex))
                end
            else
                @inbounds for _ in 1:abs(count_directed_edge[edge2])
                    push!(closing_faces, (v3, v2, closing_vertex))
                end
            end
        end

        if count_directed_edge[edge3] != 0
            # `edge3` is an exterior edge
            if count_directed_edge[edge3] > 0
                @inbounds for _ in 1:abs(count_directed_edge[edge3])
                    push!(closing_faces, (v3, v1, closing_vertex))
                end
            else
                @inbounds for _ in 1:abs(count_directed_edge[edge3])
                    push!(closing_faces, (v1, v3, closing_vertex))
                end
            end
        end
    end

    return closing_faces
end

function in_bbox(mesh, faces, min_corner, max_corner)
    faces_in_bbox = Int[]

    for face in faces
        if region_bit_outcode(barycenter(mesh, face), min_corner, max_corner) == 0
            push!(faces_in_bbox, face)
        end
    end

    return faces_in_bbox
end

@inline function barycenter(mesh::Shapes{2}, edge)
    (; edge_vertices) = mesh

    v1 = edge_vertices[edge][1]
    v2 = edge_vertices[edge][2]

    return 0.5(v1 + v2)
end

@inline function barycenter(mesh::Shapes{3}, face)
    (; face_vertices) = mesh

    v1 = face_vertices[face][1]
    v2 = face_vertices[face][2]
    v3 = face_vertices[face][3]

    return (v1 + v2 + v3) / 3
end

@inline function region_bit_outcode(p::SVector{2}, min_corner, max_corner)
    pos = 0

    # Left `0001`
    if p[1] < min_corner[1]
        pos |= 1

        # Right `0010`
    elseif p[1] > max_corner[1]
        pos |= 2
    end

    # Bottom `0100`
    if p[2] < min_corner[2]
        pos |= 4

        # Top `1000`
    elseif p[2] > max_corner[2]
        pos |= 8
    end

    # Inside `0000`
    return pos
end

@inline function region_bit_outcode(p::SVector{3}, min_corner, max_corner)
    pos = 0

    # Left `000001`
    if p[1] < min_corner[1]
        pos |= 1

        # Right `000010`
    elseif p[1] > max_corner[1]
        pos |= 2
    end

    # Bottom `000100`
    if p[2] < min_corner[2]
        pos |= 4

        # Top `001000`
    elseif p[2] > max_corner[2]
        pos |= 8
    end

    # Behind `010000`
    if p[3] < min_corner[3]
        pos |= 16
        # Front `100000`
    elseif p[3] > max_corner[3]
        pos |= 32
    end

    # Inside `000000`
    return pos
end

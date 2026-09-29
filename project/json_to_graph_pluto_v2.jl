### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 2cdaf3dd-0a25-4ce2-9da9-4ee017ca9a1b
md"""
# JSON network file → Graphs.jl graph

This Pluto notebook reads a JSON network file with the same structure as
`setA-09-net.json` and converts it into a `Graphs.jl` graph.

Expected JSON structure:

- `directed`: `true` or `false`
- `multigraph`: must be `false`
- `nodes`: objects containing `id` and `name`
- `links`: objects containing `id`, `from`, `to`, `metric`, and `capacity`

The JSON node identifiers may start at 0 (as in the example file). They are
mapped explicitly to Julia/Graphs.jl vertex numbers `1:n`.
"""

# ╔═╡ 7fdb9545-2175-4cf3-862c-ccc4ebd1d627
begin
    using JSON3
    using Graphs
    using SparseArrays
end

# ╔═╡ 705b5e90-5ef4-4285-a46a-4ef4fb0dff01
md"""
## Data structure

`graph` is the actual `Graphs.jl` graph.

The other fields keep the information that is not stored by a
`SimpleGraph` or `SimpleDiGraph`.
"""

# ╔═╡ 7e06044c-4f17-4753-8d29-54fea385e727
struct NetworkData{G<:Graphs.AbstractGraph}
    graph::G

    # Node information
    id_to_vertex::Dict{Int, Int}
    vertex_to_id::Vector{Int}
    vertex_name::Vector{String}

    # Link information, indexed by Julia endpoints (u,v)
    link_id::Dict{Tuple{Int, Int}, Int}
    metric::Dict{Tuple{Int, Int}, Float64}
    capacity::Dict{Tuple{Int, Int}, Float64}
end

# ╔═╡ fee9dcd1-c6f5-46e7-9546-24ba4209efb2
# Canonical key for an edge.
# For a directed graph:   (u,v)
# For an undirected graph: (min(u,v), max(u,v))
edge_key(directed::Bool, u::Int, v::Int) =
    directed ? (u, v) : (u <= v ? (u, v) : (v, u))

# ╔═╡ ae531c1a-a653-41bf-879c-799723f2a086
md"""
## Conversion function

The function checks the schema, creates the graph, and preserves the
node/link attributes.
"""

# ╔═╡ 7ba74f02-ab39-4cce-a008-e456bccf8332
function json_to_graph(filename::AbstractString)
    data = JSON3.read(read(filename, String))

    # ---------- Check top-level JSON structure ----------
    required_top = ("directed", "multigraph", "nodes", "links")
    missing_top = [k for k in required_top if !haskey(data, k)]

    isempty(missing_top) ||
        error("Missing top-level JSON field(s): $(join(missing_top, ", "))")

    directed = Bool(data["directed"])
    multigraph = Bool(data["multigraph"])

    multigraph &&
        error("This converter uses Graphs.jl simple graphs and does not accept multigraph=true.")

    nodes = data["nodes"]
    links = data["links"]

    # ---------- Nodes ----------
    isempty(nodes) && error("The JSON file contains no nodes.")

    for (i, node) in enumerate(nodes)
        haskey(node, "id") ||
            error("Node $i has no 'id' field.")
        haskey(node, "name") ||
            error("Node $i has no 'name' field.")
    end

    vertex_to_id = Int[Int(node["id"]) for node in nodes]

    length(unique(vertex_to_id)) == length(vertex_to_id) ||
        error("Node IDs must be unique.")

    id_to_vertex = Dict(
        json_id => julia_vertex
        for (julia_vertex, json_id) in enumerate(vertex_to_id)
    )

    vertex_name = String[String(node["name"]) for node in nodes]

    # ---------- Graph ----------
    n = length(nodes)

    g = if directed
        SimpleDiGraph(n)
    else
        SimpleGraph(n)
    end

    # ---------- Link attributes ----------
    link_id = Dict{Tuple{Int, Int}, Int}()
    metric = Dict{Tuple{Int, Int}, Float64}()
    capacity = Dict{Tuple{Int, Int}, Float64}()

    seen_link_ids = Set{Int}()

    for (i, link) in enumerate(links)
        for field in ("id", "from", "to", "metric", "capacity")
            haskey(link, field) ||
                error("Link $i has no '$field' field.")
        end

        e_id = Int(link["id"])
        from_id = Int(link["from"])
        to_id = Int(link["to"])

        e_id in seen_link_ids &&
            error("Duplicate link ID: $e_id")
        push!(seen_link_ids, e_id)

        haskey(id_to_vertex, from_id) ||
            error("Link $e_id refers to unknown node ID $from_id.")

        haskey(id_to_vertex, to_id) ||
            error("Link $e_id refers to unknown node ID $to_id.")

        u = id_to_vertex[from_id]
        v = id_to_vertex[to_id]

        # Graphs.jl SimpleGraph/SimpleDiGraph does not allow parallel edges.
        add_edge!(g, u, v) ||
            error("Duplicate/parallel link between node IDs $from_id and $to_id.")

        key = edge_key(directed, u, v)

        link_id[key] = e_id
        metric[key] = Float64(link["metric"])
        capacity[key] = Float64(link["capacity"])
    end

    return NetworkData(
        g,
        id_to_vertex,
        vertex_to_id,
        vertex_name,
        link_id,
        metric,
        capacity,
    )
end

# ╔═╡ f09af42c-d81b-4f31-8b7b-9f240c81be6b
md"""
## Helper functions

`edge_attributes(network,u,v)` returns the JSON attributes of a link,
where `u` and `v` are Julia vertex numbers.

`metric_matrix(network)` creates a sparse matrix that can, for example,
be supplied as the distance matrix to shortest-path algorithms.
"""

# ╔═╡ 41836f74-0f1f-4e99-a4fc-7be3c2f2cc55
function edge_attributes(network::NetworkData, u::Int, v::Int)
    key = edge_key(is_directed(network.graph), u, v)

    haskey(network.link_id, key) ||
        error("There is no edge $u → $v in the graph.")

    return (
        id = network.link_id[key],
        metric = network.metric[key],
        capacity = network.capacity[key],
    )
end

# ╔═╡ aa5cf923-9654-4b58-8085-2b66ea3aca47
function attribute_matrix(network::NetworkData, attribute::Symbol)
    values = if attribute === :metric
        network.metric
    elseif attribute === :capacity
        network.capacity
    else
        error("attribute must be :metric or :capacity")
    end

    directed = is_directed(network.graph)
    n = nv(network.graph)

    I = Int[]
    J = Int[]
    V = Float64[]

    for ((u, v), value) in values
        push!(I, u)
        push!(J, v)
        push!(V, value)

        if !directed && u != v
            push!(I, v)
            push!(J, u)
            push!(V, value)
        end
    end

    return sparse(I, J, V, n, n)
end

metric_matrix(network::NetworkData) =
    attribute_matrix(network, :metric)

capacity_matrix(network::NetworkData) =
    attribute_matrix(network, :capacity)

# ╔═╡ 63d45e27-6c5c-4755-9b94-4a7531c85599
md"""
## Read a file

Put the JSON file in the same directory as this notebook and change only
the filename below when needed.
"""

# ╔═╡ b58aa673-0039-483f-abd0-337c9b7e0a68
json_file = joinpath(@__DIR__, "setA-09-net.json")

# ╔═╡ 72cbbb3d-6002-491b-9559-4d0955e077fb
network = json_to_graph(json_file)

# ╔═╡ caa9cb28-0c1a-43be-97be-0b54fb06489c
g = network.graph

# ╔═╡ db1c3609-d65a-4e95-a215-815052cc27a1
(
    graph_type = typeof(g),
    directed = is_directed(g),
    number_of_vertices = nv(g),
    number_of_edges = ne(g),
)

# ╔═╡ ee9d2d3e-6d88-4bf0-8038-1e0f9a2cdbee
md"""
## Examples

The JSON node ID `0` is not a Graphs.jl vertex number. Use
`network.id_to_vertex` to convert JSON IDs to Julia vertices.
"""

# ╔═╡ de136f32-2133-4e18-abef-fbb42cffd537
begin
    json_node_id = 0
    julia_vertex = network.id_to_vertex[json_node_id]

    (
        json_id = json_node_id,
        julia_vertex = julia_vertex,
        name = network.vertex_name[julia_vertex],
    )
end

# ╔═╡ a8bfaf8b-a886-48db-b27b-f68fbdfcc871
begin
    u = network.id_to_vertex[0]
    v = network.id_to_vertex[24]

    edge_attributes(network, u, v)
end

# ╔═╡ 0286942e-2224-4bad-b162-f3db59b674cb
W = metric_matrix(network)

# ╔═╡ 670cc8d4-0867-4838-a7a3-321eab3ce633
md"""
For example, a shortest-path computation using `metric` as the arc length is:

```julia
source = network.id_to_vertex[0]
sp = dijkstra_shortest_paths(g, source, W)
```

The resulting `sp.dists[v]` gives the shortest-path distance from `source`
to Julia vertex `v`.
"""

# ╔═╡ Cell order:
# ╟─2cdaf3dd-0a25-4ce2-9da9-4ee017ca9a1b
# ╠═7fdb9545-2175-4cf3-862c-ccc4ebd1d627
# ╟─705b5e90-5ef4-4285-a46a-4ef4fb0dff01
# ╠═7e06044c-4f17-4753-8d29-54fea385e727
# ╠═fee9dcd1-c6f5-46e7-9546-24ba4209efb2
# ╟─ae531c1a-a653-41bf-879c-799723f2a086
# ╠═7ba74f02-ab39-4cce-a008-e456bccf8332
# ╟─f09af42c-d81b-4f31-8b7b-9f240c81be6b
# ╠═41836f74-0f1f-4e99-a4fc-7be3c2f2cc55
# ╠═aa5cf923-9654-4b58-8085-2b66ea3aca47
# ╟─63d45e27-6c5c-4755-9b94-4a7531c85599
# ╠═b58aa673-0039-483f-abd0-337c9b7e0a68
# ╠═72cbbb3d-6002-491b-9559-4d0955e077fb
# ╠═caa9cb28-0c1a-43be-97be-0b54fb06489c
# ╠═db1c3609-d65a-4e95-a215-815052cc27a1
# ╟─ee9d2d3e-6d88-4bf0-8038-1e0f9a2cdbee
# ╠═de136f32-2133-4e18-abef-fbb42cffd537
# ╠═a8bfaf8b-a886-48db-b27b-f68fbdfcc871
# ╠═0286942e-2224-4bad-b162-f3db59b674cb
# ╟─670cc8d4-0867-4838-a7a3-321eab3ce633
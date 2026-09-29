### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ f9b0f7ce-a589-4495-938f-33f03be74d1b
md"""
# JSON multigraph → Julia `DiMultigraph`

This Pluto notebook reads a JSON network file with the same structure as
`setA-14-net.json` and converts it into a directed multigraph.

Expected JSON structure:

- `directed`
- `multigraph`
- `graph`
- `nodes`
- `links`

Each node contains:

- `id`
- `name`

Each link contains:

- `id`
- `from`
- `to`
- `metric`
- `capacity`

Unlike `SimpleDiGraph`, a `DiMultigraph` can represent several parallel
arcs having the same ordered pair of endpoints.
"""

# ╔═╡ 5f2d823b-eee0-49dc-b29f-46269dc516f5
begin
    using JSON3
    using Graphs
    using Multigraphs
    using SparseArrays
end

# ╔═╡ 7e277083-f31e-4ddd-891c-1657be9f5ab1
md"""
## Representation

`Multigraphs.jl` stores the multiplicity of an ordered pair of vertices.

The JSON file, however, gives every arc its own `id`, `metric`, and
`capacity`. Therefore these attributes are stored separately, one entry
per JSON arc.

This is important when two parallel arcs have different attributes.
"""

# ╔═╡ 2cda6636-29e9-488a-b3cd-1710aa69af0a
struct MultiNetworkData{G}
    graph::G

    # Node information
    id_to_vertex::Dict{Int, Int}
    vertex_to_id::Vector{Int}
    vertex_name::Vector{String}

    # Arc information: position a = 1,...,m
    link_id::Vector{Int}
    arc_from::Vector{Int}
    arc_to::Vector{Int}
    metric::Vector{Float64}
    capacity::Vector{Float64}

    # Look-up tables
    id_to_arc::Dict{Int, Int}
    pair_to_arcs::Dict{Tuple{Int, Int}, Vector{Int}}
end

# ╔═╡ e68aa8c2-3269-4b3a-8eca-a36fcf2b10a1
md"""
## Conversion function
"""

# ╔═╡ c5650c13-2860-4605-9b84-d0392335053d
function json_to_multigraph(filename::AbstractString)
    data = JSON3.read(read(filename, String))

    # ---------- Check top-level structure ----------
    required_top = ("directed", "multigraph", "nodes", "links")
    missing_top = [k for k in required_top if !haskey(data, k)]

    isempty(missing_top) ||
        error("Missing top-level JSON field(s): $(join(missing_top, ", "))")

    Bool(data["directed"]) ||
        error("This notebook expects a directed multigraph.")

    Bool(data["multigraph"]) ||
        error("The JSON file does not declare multigraph=true.")

    nodes = data["nodes"]
    links = data["links"]

    isempty(nodes) && error("The JSON file contains no nodes.")

    # ---------- Nodes ----------
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

    # ---------- Directed multigraph ----------
    n = length(nodes)
    g = DiMultigraph(n)

    # ---------- One record per JSON arc ----------
    m = length(links)

    link_id = Vector{Int}(undef, m)
    arc_from = Vector{Int}(undef, m)
    arc_to = Vector{Int}(undef, m)
    metric = Vector{Float64}(undef, m)
    capacity = Vector{Float64}(undef, m)

    id_to_arc = Dict{Int, Int}()
    pair_to_arcs = Dict{Tuple{Int, Int}, Vector{Int}}()

    for (a, link) in enumerate(links)
        for field in ("id", "from", "to", "metric", "capacity")
            haskey(link, field) ||
                error("Link $a has no '$field' field.")
        end

        json_link_id = Int(link["id"])
        from_id = Int(link["from"])
        to_id = Int(link["to"])

        haskey(id_to_arc, json_link_id) &&
            error("Duplicate link ID: $json_link_id")

        haskey(id_to_vertex, from_id) ||
            error("Link $json_link_id refers to unknown node ID $from_id.")

        haskey(id_to_vertex, to_id) ||
            error("Link $json_link_id refers to unknown node ID $to_id.")

        u = id_to_vertex[from_id]
        v = id_to_vertex[to_id]

        # Every call adds one copy of the arc.
        add_edge!(g, u, v)

        link_id[a] = json_link_id
        arc_from[a] = u
        arc_to[a] = v
        metric[a] = Float64(link["metric"])
        capacity[a] = Float64(link["capacity"])

        id_to_arc[json_link_id] = a
        push!(get!(pair_to_arcs, (u, v), Int[]), a)
    end

    return MultiNetworkData(
        g,
        id_to_vertex,
        vertex_to_id,
        vertex_name,
        link_id,
        arc_from,
        arc_to,
        metric,
        capacity,
        id_to_arc,
        pair_to_arcs,
    )
end

# ╔═╡ dbfa58a1-795c-479a-8d81-1aeef7eb3fad
md"""
## Helper functions
"""

# ╔═╡ feadcc8d-0049-4211-aac0-b55c029bd6b8
function arc_attributes(network::MultiNetworkData, a::Int)
    1 <= a <= length(network.link_id) ||
        error("Arc index $a is outside 1:$(length(network.link_id)).")

    u = network.arc_from[a]
    v = network.arc_to[a]

    return (
        arc_index = a,
        id = network.link_id[a],
        from_vertex = u,
        to_vertex = v,
        from_json_id = network.vertex_to_id[u],
        to_json_id = network.vertex_to_id[v],
        metric = network.metric[a],
        capacity = network.capacity[a],
    )
end

# ╔═╡ 51c767b3-ca14-4ce9-97e0-92c0bff724ae
function arc_by_id(network::MultiNetworkData, json_link_id::Int)
    haskey(network.id_to_arc, json_link_id) ||
        error("Unknown JSON link ID $json_link_id.")

    return arc_attributes(
        network,
        network.id_to_arc[json_link_id],
    )
end

# ╔═╡ 3a674046-dfbb-4ca4-9dab-50fa509929d8
function arcs_between(
    network::MultiNetworkData,
    u::Int,
    v::Int,
)
    a = get(network.pair_to_arcs, (u, v), Int[])
    return [arc_attributes(network, k) for k in a]
end

# ╔═╡ dc7846d6-932d-4b3b-ad7e-b956192c3edc
function multiplicity(
    network::MultiNetworkData,
    u::Int,
    v::Int,
)
    return length(get(network.pair_to_arcs, (u, v), Int[]))
end

# ╔═╡ 679d5686-e249-4028-9634-53bbd9b441df
md"""
## Simple Graphs.jl projection

Some Graphs.jl algorithms are naturally formulated on a simple graph.
The following function keeps one directed edge `u → v` whenever at least
one multigraph arc exists from `u` to `v`.
"""

# ╔═╡ 3469e52f-adca-48da-a84f-25d0fbc86a09
function simple_projection(network::MultiNetworkData)
    g = SimpleDiGraph(length(network.vertex_to_id))

    for (u, v) in keys(network.pair_to_arcs)
        add_edge!(g, u, v)
    end

    return g
end

# ╔═╡ 16dd473f-01fb-4b9d-9462-a76bebc3a7f7
md"""
## Matrices for algorithms on the simple projection

For parallel arcs:

- the shortest-path `metric` between `u` and `v` is the minimum metric;
- the total `capacity` between `u` and `v` is the sum of capacities.
"""

# ╔═╡ eac7e670-e435-4ce4-a6c4-4e418962e339
function metric_matrix(network::MultiNetworkData)
    n = length(network.vertex_to_id)

    I = Int[]
    J = Int[]
    V = Float64[]

    for ((u, v), arcs) in network.pair_to_arcs
        push!(I, u)
        push!(J, v)
        push!(V, minimum(network.metric[a] for a in arcs))
    end

    return sparse(I, J, V, n, n)
end

# ╔═╡ eec77268-b7dd-420f-8cc8-6c825805daa5
function capacity_matrix(network::MultiNetworkData)
    n = length(network.vertex_to_id)

    I = Int[]
    J = Int[]
    V = Float64[]

    for ((u, v), arcs) in network.pair_to_arcs
        push!(I, u)
        push!(J, v)
        push!(V, sum(network.capacity[a] for a in arcs))
    end

    return sparse(I, J, V, n, n)
end

# ╔═╡ 42e9e132-3232-4921-9288-bd68be51ee1e
md"""
## Read a JSON file

Put the JSON file in the same directory as this Pluto notebook.
Only the filename below needs to be changed for another instance.
"""

# ╔═╡ d23567dd-d73d-46ef-b87a-669ba42b6f5a
json_file = joinpath(@__DIR__, "setA-14-net.json")

# ╔═╡ af14cfa8-5f98-43cd-9b75-301d9c1a9b53
network = json_to_multigraph(json_file)

# ╔═╡ 8b0f96b8-1cf7-4ff3-82e3-fd12541698e2
mg = network.graph

# ╔═╡ 298a2ab8-7b61-4ad1-841b-faa8e10d6824
begin
    multiplicities = length.(values(network.pair_to_arcs))

    (
        graph_type = typeof(mg),
        number_of_vertices = length(network.vertex_to_id),
        number_of_json_arcs = length(network.link_id),
        number_of_distinct_ordered_pairs = length(network.pair_to_arcs),
        maximum_multiplicity = maximum(multiplicities),
        number_of_pairs_with_parallel_arcs =
            count(>(1), multiplicities),
    )
end

# ╔═╡ f4c2b4da-a72f-45eb-b243-5583652c0ab1
md"""
## Node-ID conversion

The JSON node IDs are not assumed to coincide with Julia vertex numbers.

For the supplied file the JSON nodes appear in descending order
(`249, 248, …, 0`), so this explicit mapping is essential.
"""

# ╔═╡ 4cfebd27-408d-4798-8ee1-8e1594675170
begin
    json_node_id = 248
    julia_vertex = network.id_to_vertex[json_node_id]

    (
        json_id = json_node_id,
        julia_vertex = julia_vertex,
        name = network.vertex_name[julia_vertex],
    )
end

# ╔═╡ 993ff416-ebe8-43cd-8671-ca7fd80ad9f8
md"""
## Inspect an arc by its JSON link ID
"""

# ╔═╡ 18c46f3e-a92d-466f-a5f6-e4f0c3a257ab
arc_by_id(network, 0)

# ╔═╡ a02c2d97-4bc4-4a5b-9a4f-b7b403b20b9c
md"""
## Inspect all parallel arcs between two nodes

The endpoints below are given as JSON node IDs and then converted to
Julia vertex numbers.
"""

# ╔═╡ 995ddd1c-c2bd-453a-896a-f7ef6e591d17
begin
    from_id = 248
    to_id = 239

    u = network.id_to_vertex[from_id]
    v = network.id_to_vertex[to_id]

    arcs_between(network, u, v)
end

# ╔═╡ 801a3391-ee64-4ffe-ac65-e17200889884
md"""
## Shortest paths using `metric`

When parallel arcs exist, Dijkstra only needs the smallest metric among
parallel arcs joining the same ordered pair.

Therefore we use the simple projection together with `metric_matrix`.
"""

# ╔═╡ a8c28b02-2d0f-4563-a4c0-8bd27bf12ce6
begin
    g_simple = simple_projection(network)
    W = metric_matrix(network)

    source_json_id = 248
    source = network.id_to_vertex[source_json_id]

    sp = dijkstra_shortest_paths(g_simple, source, W)
end

# ╔═╡ 60fe50e2-5d7c-47d1-8aea-86f12cc15012
md"""
`sp.dists[v]` is the shortest-path distance from JSON node `248`
to Julia vertex `v`.

To recover a JSON node ID from a Julia vertex `v`, use

```julia
network.vertex_to_id[v]
```
"""

# ╔═╡ Cell order:
# ╟─f9b0f7ce-a589-4495-938f-33f03be74d1b
# ╠═5f2d823b-eee0-49dc-b29f-46269dc516f5
# ╟─7e277083-f31e-4ddd-891c-1657be9f5ab1
# ╠═2cda6636-29e9-488a-b3cd-1710aa69af0a
# ╟─e68aa8c2-3269-4b3a-8eca-a36fcf2b10a1
# ╠═c5650c13-2860-4605-9b84-d0392335053d
# ╟─dbfa58a1-795c-479a-8d81-1aeef7eb3fad
# ╠═feadcc8d-0049-4211-aac0-b55c029bd6b8
# ╠═51c767b3-ca14-4ce9-97e0-92c0bff724ae
# ╠═3a674046-dfbb-4ca4-9dab-50fa509929d8
# ╠═dc7846d6-932d-4b3b-ad7e-b956192c3edc
# ╟─679d5686-e249-4028-9634-53bbd9b441df
# ╠═3469e52f-adca-48da-a84f-25d0fbc86a09
# ╟─16dd473f-01fb-4b9d-9462-a76bebc3a7f7
# ╠═eac7e670-e435-4ce4-a6c4-4e418962e339
# ╠═eec77268-b7dd-420f-8cc8-6c825805daa5
# ╟─42e9e132-3232-4921-9288-bd68be51ee1e
# ╠═d23567dd-d73d-46ef-b87a-669ba42b6f5a
# ╠═af14cfa8-5f98-43cd-9b75-301d9c1a9b53
# ╠═8b0f96b8-1cf7-4ff3-82e3-fd12541698e2
# ╠═298a2ab8-7b61-4ad1-841b-faa8e10d6824
# ╟─f4c2b4da-a72f-45eb-b243-5583652c0ab1
# ╠═4cfebd27-408d-4798-8ee1-8e1594675170
# ╟─993ff416-ebe8-43cd-8671-ca7fd80ad9f8
# ╠═18c46f3e-a92d-466f-a5f6-e4f0c3a257ab
# ╟─a02c2d97-4bc4-4a5b-9a4f-b7b403b20b9c
# ╠═995ddd1c-c2bd-453a-896a-f7ef6e591d17
# ╟─801a3391-ee64-4ffe-ac65-e17200889884
# ╠═a8c28b02-2d0f-4563-a4c0-8bd27bf12ce6
# ╟─60fe50e2-5d7c-47d1-8aea-86f12cc15012
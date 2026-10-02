# =============================================================================
#  TSP: Miller–Tucker–Zemlin (MTZ) formulation vs subtour elimination (SEC)
# =============================================================================
#
#  MTZ (directed, arc variables x_ij, order variables u_i):
#
#     min  Σ c_ij x_ij
#     s.t. Σ_j x_ij = 1                               for every i     (out-degree)
#          Σ_i x_ij = 1                               for every j     (in-degree)
#          u_i - u_j + (n-1) x_ij ≤ n-2               i ≠ j, i,j ≥ 2  (MTZ)
#          2 ≤ u_i ≤ n                                i ≥ 2
#          x_ij ∈ {0,1}
#
#  If x_ij = 1 the constraint forces u_j ≥ u_i + 1: the u_i number the cities
#  along the tour starting from city 1, so no cycle can avoid city 1.
#  If x_ij = 0 it is inactive (a "big-M" constraint) — which is exactly why
#  its LP relaxation is weak.
#
#  Lifted MTZ (Desrochers & Laporte, 1991) adds the reverse arc:
#          u_i - u_j + (n-1) x_ij + (n-3) x_ji ≤ n-2
#
#  Contrast with the SEC model of tsp_bnc_solution.jl:
#     MTZ : O(n²) constraints, compact, no separation, any MIP solver can take it,
#           weak LP bound  -> large trees
#     SEC : O(2ⁿ) constraints, needs lazy constraints / cuts,
#           strong LP bound (subtour LP)  -> tiny trees
#
#  Needs tsp_bnc_solution.jl in the same folder (instance, SEC branch-and-cut,
#  separation routines are reused).
#  Run:  julia tsp_mtz_compare.jl
# =============================================================================

include(joinpath(@__DIR__, "tsp_bnc_solution.jl"))

# -----------------------------------------------------------------------------
# 1. The MTZ model
# -----------------------------------------------------------------------------

function cost_matrix(n, edges, cost)
    C = zeros(n, n)
    for (e, (i, j)) in enumerate(edges)
        C[i, j] = cost[e]
        C[j, i] = cost[e]
    end
    return C
end

"""
Build the MTZ model. `binary = false` gives the LP relaxation (used by Bonobo),
`binary = true` the MIP (used by HiGHS directly). `lifted = true` uses the
Desrochers–Laporte lifted constraints.
"""
function build_mtz(n, edges, cost; binary = false, lifted = false)
    C = cost_matrix(n, edges, cost)
    arcs = [(i, j) for i in 1:n for j in 1:n if i != j]
    arc_index = Dict(a => k for (k, a) in enumerate(arcs))
    out_arcs = [findall(a -> a[1] == v, arcs) for v in 1:n]
    in_arcs = [findall(a -> a[2] == v, arcs) for v in 1:n]

    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, 0 <= x[1:length(arcs)] <= 1)
    binary && set_binary.(x)
    @variable(m, 2 <= u[2:n] <= n)
    @objective(m, Min, sum(C[arcs[k][1], arcs[k][2]] * x[k] for k in eachindex(arcs)))
    for v in 1:n
        @constraint(m, sum(x[k] for k in out_arcs[v]) == 1)
        @constraint(m, sum(x[k] for k in in_arcs[v]) == 1)
    end
    for (k, (i, j)) in enumerate(arcs)
        (i == 1 || j == 1) && continue
        if lifted
            r = arc_index[(j, i)]
            @constraint(m, u[i] - u[j] + (n - 1) * x[k] + (n - 3) * x[r] <= n - 2)
        else
            @constraint(m, u[i] - u[j] + (n - 1) * x[k] <= n - 2)
        end
    end
    return m, x, arcs
end

# -----------------------------------------------------------------------------
# 2. MTZ solved by plain LP-based branch-and-bound with Bonobo (no cuts)
# -----------------------------------------------------------------------------

mutable struct MTZRoot
    model::JuMP.Model
    x::Vector{VariableRef}
    nb_nodes::Int
    node_limit::Int
end

mutable struct MTZNode <: BB.AbstractNode
    std::BB.BnBNodeInfo
    lbs::Vector{Float64}
    ubs::Vector{Float64}
end

# we branch on the x variables only; u stays continuous
BB.get_branching_indices(root::MTZRoot) = collect(1:length(root.x))

BB.get_relaxed_values(tree::BB.BnBTree{MTZNode,MTZRoot}, node::MTZNode) = value.(tree.root.x)

function BB.get_branching_nodes_info(tree::BB.BnBTree{MTZNode,MTZRoot}, node::MTZNode, vidx::Int)
    lbs0, ubs0 = copy(node.lbs), copy(node.ubs)
    ubs0[vidx] = 0.0
    lbs1, ubs1 = copy(node.lbs), copy(node.ubs)
    lbs1[vidx] = 1.0
    return [(lbs = lbs0, ubs = ubs0), (lbs = lbs1, ubs = ubs1)]
end

function BB.evaluate_node!(tree::BB.BnBTree{MTZNode,MTZRoot}, node::MTZNode)
    R = tree.root
    R.nb_nodes += 1
    set_lower_bound.(R.x, node.lbs)
    set_upper_bound.(R.x, node.ubs)
    optimize!(R.model)
    termination_status(R.model) == MOI.OPTIMAL || return NaN, NaN
    obj = objective_value(R.model)
    if all(BB.is_approx_feasible(tree, v) for v in value.(R.x))
        # The model is complete: an integer x is a tour. No lazy check needed —
        # compare with evaluate_node! in tsp_bnc_solution.jl.
        return obj, obj
    end
    return obj, NaN
end

"Stop at a node limit (MTZ trees explode quickly), otherwise Bonobo's default rule."
function BB.terminated(tree::BB.BnBTree{MTZNode,MTZRoot})
    tree.root.nb_nodes >= tree.root.node_limit && return true
    return invoke(BB.terminated, Tuple{BB.BnBTree}, tree)
end

"Best proven lower bound when the search stops (open nodes carry their parent's bound)."
function final_bound(tree)
    isempty(tree.nodes) && return tree.incumbent
    return min(tree.incumbent, minimum(nd.lb for nd in values(tree.nodes)))
end

function solve_mtz_bonobo(n, edges, cost; node_limit = 20_000, lifted = false)
    m, x, _ = build_mtz(n, edges, cost; lifted)
    root = MTZRoot(m, x, 0, node_limit)
    tree = BB.initialize(;
        traverse_strategy = BB.BestFirstSearch(),
        branch_strategy = BB.MOST_INFEASIBLE(),
        Node = MTZNode,
        root = root,
        sense = :Min,
    )
    BB.set_root!(tree, (lbs = zeros(length(x)), ubs = ones(length(x))))
    t = @elapsed BB.optimize!(tree)
    obj = isempty(tree.solutions) ? Inf : BB.get_objective_value(tree)
    return (obj = obj, bound = final_bound(tree), nodes = root.nb_nodes, time = t)
end

# -----------------------------------------------------------------------------
# 3. MTZ solved by HiGHS as a MIP (presolve, its own cuts, heuristics...)
# -----------------------------------------------------------------------------

function solve_mtz_highs(n, edges, cost; time_limit = 120.0, lifted = false)
    m, _, _ = build_mtz(n, edges, cost; binary = true, lifted)
    set_time_limit_sec(m, time_limit)
    t = @elapsed optimize!(m)
    obj = has_values(m) ? objective_value(m) : Inf
    nodes = try
        Int(MOI.get(m, MOI.NodeCount()))
    catch
        -1
    end
    return (obj = obj, bound = objective_bound(m), nodes = nodes, time = t)
end

# -----------------------------------------------------------------------------
# 4. SEC branch-and-cut (from tsp_bnc_solution.jl), same result format
# -----------------------------------------------------------------------------

function solve_sec(n, edges, cost; use_user_cuts = true)
    obj, _, root, t = solve_bnc(n, edges, cost; use_user_cuts)
    return (obj = obj, bound = obj, nodes = root.nb_nodes, time = t,
            lazy = root.nb_lazy, user = root.nb_user)
end

# -----------------------------------------------------------------------------
# 5. Root LP bounds
# -----------------------------------------------------------------------------

"""
Return the LP bounds of: MTZ, lifted MTZ, degree constraints only
(2-matching LP) and the subtour LP (all SECs, separated exactly).
"""
function root_bounds(n, edges, cost)
    m1, _, _ = build_mtz(n, edges, cost)
    optimize!(m1)
    m2, _, _ = build_mtz(n, edges, cost; lifted = true)
    optimize!(m2)

    m3, x3 = build_lp(n, edges, cost)
    optimize!(m3)
    deg = objective_value(m3)
    while true   # cutting-plane loop until no SEC is violated
        xv = value.(x3)
        sets = sec_from_components(n, edges, xv; tol = 1e-6)
        if isempty(sets)
            val, S = min_cut_stoer_wagner(n, edges, xv)
            val < 2 - 1e-6 || break
            sets = [S]
        end
        for S in sets
            add_sec!(m3, x3, n, edges, S)
        end
        optimize!(m3)
    end
    return (mtz = objective_value(m1), lifted = objective_value(m2),
            deg = deg, subtour = objective_value(m3))
end

gap(bound, opt) = 100 * (opt - bound) / opt

# -----------------------------------------------------------------------------
# 6. Experiments
# -----------------------------------------------------------------------------

fmt(v) = isfinite(v) ? @sprintf("%.1f", v) : "—"

function compare(; seed = 1)
    println("="^88)
    println("A. Root LP bounds (gap to the optimum, in %)")
    println("="^88)
    @printf("%4s %8s | %16s %16s %16s %16s\n",
            "n", "opt", "MTZ", "lifted MTZ", "degree only", "subtour LP")
    for n in (10, 15, 20, 30, 40)
        _, edges, cost = random_instance(n; seed)
        opt = solve_sec(n, edges, cost).obj
        b = root_bounds(n, edges, cost)
        @printf("%4d %8.1f | %8.1f (%4.1f%%) %8.1f (%4.1f%%) %8.1f (%4.1f%%) %8.1f (%4.1f%%)\n",
                n, opt, b.mtz, gap(b.mtz, opt), b.lifted, gap(b.lifted, opt),
                b.deg, gap(b.deg, opt), b.subtour, gap(b.subtour, opt))
    end

    println()
    println("="^88)
    println("B. Same tree search (Bonobo, best-first, most-infeasible branching)")
    println("="^88)
    @printf("%4s %-22s %8s %8s %8s %8s\n", "n", "method", "obj", "bound", "nodes", "time(s)")
    for n in (8, 10, 12, 15)
        _, edges, cost = random_instance(n; seed)
        for (name, r) in (("MTZ (B&B)", solve_mtz_bonobo(n, edges, cost)),
                          ("lifted MTZ (B&B)", solve_mtz_bonobo(n, edges, cost; lifted = true)),
                          ("SEC (B&C)", solve_sec(n, edges, cost)))
            @printf("%4d %-22s %8s %8s %8d %8.2f\n", n, name, fmt(r.obj), fmt(r.bound), r.nodes, r.time)
        end
        println()
    end

    println("="^88)
    println("C. MTZ given to a full MIP solver (HiGHS, 120 s limit) vs our SEC branch-and-cut")
    println("="^88)
    @printf("%4s %-22s %8s %8s %8s %8s\n", "n", "method", "obj", "bound", "nodes", "time(s)")
    for n in (15, 25, 40)
        _, edges, cost = random_instance(n; seed)
        for (name, r) in (("MTZ (HiGHS MIP)", solve_mtz_highs(n, edges, cost)),
                          ("lifted MTZ (HiGHS MIP)", solve_mtz_highs(n, edges, cost; lifted = true)),
                          ("SEC (Bonobo B&C)", solve_sec(n, edges, cost)))
            @printf("%4d %-22s %8s %8s %8d %8.2f\n", n, name, fmt(r.obj), fmt(r.bound), r.nodes, r.time)
        end
        println()
    end
end

compare()

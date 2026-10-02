### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ cba540a6-51a7-41f3-bb1b-08ea87609ce2
begin
    using Bonobo
    using JuMP
    using HiGHS
    using Combinatorics
    using DataFrames
    using PlutoUI

    import MathOptInterface as MOI

    const BB = Bonobo
end

# ╔═╡ ad0c1770-319e-4a36-91ad-6a5151153d29
md"""
# Branch-and-Cut with Bonobo.jl

This Pluto notebook implements two Branch-and-Cut algorithms for the same binary
knapsack instance:

1. Chvátal-Gomory cuts generated from the residual knapsack at each node;
2. cover and extended-cover inequalities generated from the residual knapsack.

The instance has 27 items: the original 9 items plus 18 additional items.
The capacity is three times the original capacity. The cut-node tables show
where each separator becomes active in the Branch-and-Bound tree.
"""

# ╔═╡ 54a8acc0-132d-455f-bf65-9fb9dd1405fd
TableOfContents()

# ╔═╡ 5f6acef5-fe13-4f49-a134-1b43588d8431
md"""
## 1. Larger knapsack instance

We solve

`max sum(p[i] * x[i])`

subject to

`sum(w[i] * x[i]) <= 69`, `x[i] in {0,1}`.

The optimal integer value is 212 (independently verified by dynamic programming).
The additional items were generated once with seed 20261002 and stored explicitly
so that every run uses the same instance.
"""

# ╔═╡ d7415c61-6cd9-4553-a17b-18b00655052f
begin
    profit = [
        12.0, 5.0, 9.0, 10.0, 9.0, 13.0, 14.0, 18.0, 7.0,
        7.0, 15.0, 7.0, 18.0, 13.0, 10.0, 11.0, 16.0, 10.0,
        6.0, 15.0, 12.0, 10.0, 9.0, 12.0, 14.0, 16.0, 18.0,
    ]
    weight = [
        3.0, 6.0, 10.0, 4.0, 6.0, 3.0, 3.0, 7.0, 9.0,
        7.0, 3.0, 10.0, 6.0, 7.0, 9.0, 4.0, 4.0, 3.0,
        6.0, 4.0, 5.0, 8.0, 4.0, 3.0, 5.0, 10.0, 9.0,
    ]
    capacity = 69.0
    n = length(profit)

    DataFrame(
        item = 1:n,
        profit = profit,
        weight = weight,
    )
end

# ╔═╡ af4d25a6-39bc-4042-bee1-fd4fc57439ca
md"""
The root LP relaxation has objective value 215. The cell below computes
its solution. `MOST_INFEASIBLE` selects a variable closest to 0.5
among the fractional variables at each node.
"""

# ╔═╡ a0dd4fab-b811-42ee-a79d-a358f4ed3db2
function solve_root_lp(profit, weight, capacity)
    n = length(profit)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, 0 <= x[1:n] <= 1)
    @constraint(model, sum(weight[i] * x[i] for i in 1:n) <= capacity)
    @objective(model, Max, sum(profit[i] * x[i] for i in 1:n))

    optimize!(model)

    return (
        objective = objective_value(model),
        solution = value.(x),
    )
end

# ╔═╡ ab49c6b7-0a72-47bc-abfb-9b5c0c86da66
root_lp = solve_root_lp(profit, weight, capacity)

# ╔═╡ cc916f27-a1a1-4842-97f3-db326ece5fb5
md"""
## 2. Problem and cut structures

`KnapsackBCProblem` is stored as the Bonobo root object. Unlike the earlier
Branch-and-Bound notebook, the root is not a JuMP model: the LP relaxation is
rebuilt at every node. This makes local cuts easier to handle.
"""

# ╔═╡ a01a6f73-a2ee-4804-8a03-edd7900f20a8
struct LinearCut
    coeff::Vector{Float64}
    rhs::Float64
    name::String
end

# ╔═╡ 8df4be2f-527c-49d1-b4c8-8c6e6ec8c9d1
struct KnapsackBCProblem
    profit::Vector{Float64}
    weight::Vector{Float64}
    capacity::Float64
    cut_family::Symbol
    theta_values::Vector{Float64}
    max_cover_size::Int
    max_cut_rounds::Int
    max_cuts_per_round::Int
end

# ╔═╡ 1220962e-d905-4407-91a4-8b36aef61d46
mutable struct BCNode <: BB.AbstractNode
    std::BB.BnBNodeInfo

    lbs::Vector{Float64}
    ubs::Vector{Float64}
    local_cuts::Vector{LinearCut}

    status::MOI.TerminationStatusCode

    parent_id::Int
    branch_text::String

    lp_objective::Float64
    lp_solution::Vector{Float64}
    integral::Bool

    cuts_added_at_node::Int
    cut_rounds::Int
    new_cut_names::Vector{String}
end

# ╔═╡ 52d70135-58d5-4f1b-95c3-17315c10303d
function BB.get_branching_indices(problem::KnapsackBCProblem)
    return collect(eachindex(problem.profit))
end

# ╔═╡ 5518249f-d936-4202-88cb-b99105dc7bce
md"""
## 3. Node relaxation

A node stores variable bounds and all local cuts inherited from its ancestors.
The JuMP relaxation is rebuilt from this information.
"""

# ╔═╡ 78a2c083-21a6-4953-afa8-ce196cc00155
function build_node_model(problem::KnapsackBCProblem, node::BCNode)
    n = length(problem.profit)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:n])

    for i in 1:n
        set_lower_bound(x[i], node.lbs[i])
        set_upper_bound(x[i], node.ubs[i])
    end

    @constraint(
        model,
        sum(problem.weight[i] * x[i] for i in 1:n) <= problem.capacity
    )

    for cut in node.local_cuts
        @constraint(
            model,
            sum(cut.coeff[i] * x[i] for i in 1:n) <= cut.rhs
        )
    end

    @objective(
        model,
        Max,
        sum(problem.profit[i] * x[i] for i in 1:n)
    )

    return model, x
end

# ╔═╡ e87f921c-af24-4220-85d9-c45d65a30fbb
function fixed_one_indices(node::BCNode; atol = 1e-9)
    return [
        i for i in eachindex(node.lbs)
        if abs(node.lbs[i] - 1.0) <= atol &&
           abs(node.ubs[i] - 1.0) <= atol
    ]
end

# ╔═╡ 4c505906-178e-40c0-bae4-b0158793b736
function free_indices(node::BCNode; atol = 1e-9)
    return [
        i for i in eachindex(node.lbs)
        if node.ubs[i] - node.lbs[i] > atol
    ]
end

# ╔═╡ a5b50658-6568-412c-a97a-b8d5b3b24446
function residual_capacity(problem::KnapsackBCProblem, node::BCNode)
    fixed_one = fixed_one_indices(node)

    used_capacity =
        isempty(fixed_one) ? 0.0 :
        sum(problem.weight[i] for i in fixed_one)

    return problem.capacity - used_capacity
end

# ╔═╡ fb92e385-f39e-485a-a583-72cd17464d91
md"""
## 4. Chvátal-Gomory separator

For a residual knapsack

`sum(w[i] * x[i]) <= bN`

and a multiplier `theta > 0`, the cut

`sum(floor(theta*w[i]) * x[i]) <= floor(theta*bN)`

is valid for the free binary variables. Several multiplier values are tested.
"""

# ╔═╡ 23ce4917-abf6-4c3c-b9af-2337316e2372
function cg_candidates(
    problem::KnapsackBCProblem,
    node::BCNode,
    xval::Vector{Float64};
    atol = 1e-6,
)
    F = free_indices(node)
    bN = residual_capacity(problem, node)
    n = length(problem.profit)

    cuts = LinearCut[]

    for theta in problem.theta_values
        coeff = zeros(n)

        for i in F
            coeff[i] =
                floor(Int, theta * problem.weight[i] + 1e-9)
        end

        rhs = floor(Int, theta * bN + 1e-9)

        all(iszero, coeff) && continue

        lhs = sum(coeff[i] * xval[i] for i in 1:n)

        if lhs > rhs + atol
            push!(
                cuts,
                LinearCut(
                    coeff,
                    Float64(rhs),
                    "CG theta=$(round(theta, digits=4))",
                ),
            )
        end
    end

    return cuts
end

# ╔═╡ 532f1295-e063-4581-aa85-a8989a7fa634
md"""
## 5. Cover and extended-cover separator

The residual capacity is reduced by variables already fixed to one.

A minimal cover `C` satisfies

`sum(w[i], i in C) > bN`

but removing any item destroys the cover property.

The extended cover adds every free item whose weight is at least the largest
weight in the minimal cover.
"""

# ╔═╡ 4a9996e7-3f7d-4d68-acaa-6d59af9bbfe8
function is_minimal_cover(
    problem::KnapsackBCProblem,
    C,
    bN;
    atol = 1e-9,
)
    total = sum(problem.weight[i] for i in C)

    total > bN + atol || return false

    return all(
        total - problem.weight[i] <= bN + atol
        for i in C
    )
end

# ╔═╡ a3130393-e50d-4a2f-9f30-3c348c1cc15a
function extend_cover(
    problem::KnapsackBCProblem,
    node::BCNode,
    C,
)
    isempty(C) && return Int[]

    F = free_indices(node)
    threshold = maximum(problem.weight[i] for i in C)

    E = collect(C)

    for j in F
        if !(j in C) && problem.weight[j] >= threshold
            push!(E, j)
        end
    end

    return sort(unique(E))
end

# ╔═╡ ffeed9e5-f18a-48ac-8ce8-ad8ba31682b3
function extended_cover_candidates(
    problem::KnapsackBCProblem,
    node::BCNode,
    xval::Vector{Float64};
    atol = 1e-6,
)
    F = free_indices(node)
    bN = residual_capacity(problem, node)
    n = length(problem.profit)

    cuts = LinearCut[]

    largest_size = min(problem.max_cover_size, length(F))

    for k in 2:largest_size
        for C_tuple in combinations(F, k)
            C = collect(C_tuple)

            is_minimal_cover(problem, C, bN) || continue

            E = extend_cover(problem, node, C)
            rhs = k - 1

            coeff = zeros(n)
            coeff[E] .= 1.0

            lhs = sum(coeff[i] * xval[i] for i in 1:n)

            if lhs > rhs + atol
                push!(
                    cuts,
                    LinearCut(
                        coeff,
                        Float64(rhs),
                        "Extended cover C=$(C), E=$(E)",
                    ),
                )
            end
        end
    end

    return cuts
end

# ╔═╡ 31d6dd81-13a5-44e9-88a0-5f5934c1171e
md"""
## 6. Cut filtering

Only violated cuts that are not already present in the node are kept.
When several cuts are available, the most violated ones are added first.
"""

# ╔═╡ 7126d8e0-b2bb-49e6-bb1c-ca340b14b19d
cut_key(cut::LinearCut) = (Tuple(cut.coeff), cut.rhs)

# ╔═╡ 59a384c5-69b3-419e-9ac3-e1b9bccf2929
function cut_violation(cut::LinearCut, xval)
    return sum(cut.coeff[i] * xval[i] for i in eachindex(xval)) -
           cut.rhs
end

# ╔═╡ e6a5a16a-f639-40dc-b4b4-8a410c67fdb1
function new_cuts(
    problem::KnapsackBCProblem,
    node::BCNode,
    xval::Vector{Float64},
)
    candidates =
        problem.cut_family == :CG ?
        cg_candidates(problem, node, xval) :
        extended_cover_candidates(problem, node, xval)

    existing = Set(cut_key(cut) for cut in node.local_cuts)

    unique_new = LinearCut[]
    new_keys = Set{Any}()

    for cut in candidates
        key = cut_key(cut)

        if !(key in existing) && !(key in new_keys)
            push!(unique_new, cut)
            push!(new_keys, key)
        end
    end

    sort!(
        unique_new;
        by = cut -> cut_violation(cut, xval),
        rev = true,
    )

    k = min(length(unique_new), problem.max_cuts_per_round)

    return unique_new[1:k]
end

# ╔═╡ 34f6cfd6-7358-4bcb-b1fc-803e707db554
md"""
## 7. Branch-and-Cut node evaluation

The cutting-plane loop is executed inside `BB.evaluate_node!`.

The returned first value is the node relaxation value. In a maximization problem
Bonobo internally changes signs when storing its B&B bounds.
"""

# ╔═╡ e3a6c5f7-75e5-4434-8a98-c05f05e7d768
function BB.evaluate_node!(
    tree::BB.BnBTree{BCNode,KnapsackBCProblem},
    node::BCNode,
)
    problem = tree.root

    node.cuts_added_at_node = 0
    node.cut_rounds = 0
    empty!(node.new_cut_names)

    while true
        model, x = build_node_model(problem, node)
        optimize!(model)

        node.status = termination_status(model)

        if node.status != MOI.OPTIMAL
            node.lp_objective = NaN
            node.lp_solution .= NaN
            node.integral = false

            return NaN, NaN
        end

        node.lp_objective = objective_value(model)
        node.lp_solution = value.(x)

        if node.cut_rounds >= problem.max_cut_rounds
            break
        end

        cuts = new_cuts(problem, node, node.lp_solution)

        isempty(cuts) && break

        append!(node.local_cuts, cuts)
        append!(node.new_cut_names, getfield.(cuts, :name))

        node.cuts_added_at_node += length(cuts)
        node.cut_rounds += 1
    end

    node.integral = all(
        BB.is_approx_feasible.(tree, node.lp_solution)
    )

    if node.integral
        return node.lp_objective, node.lp_objective
    else
        return node.lp_objective, NaN
    end
end

# ╔═╡ fcaffbae-36e4-4c70-a1b0-a6fedf48010a
function BB.get_relaxed_values(
    tree::BB.BnBTree{BCNode,KnapsackBCProblem},
    node::BCNode,
)
    return copy(node.lp_solution)
end

# ╔═╡ 12a43db7-dfdd-4b06-9c60-6e90a7109db7
md"""
## 8. Branching

Children inherit every local cut generated at the parent node.
"""

# ╔═╡ be85ab9b-b9cc-4c5b-bb7a-06c44fc2a601
function child_info(
    node::BCNode,
    lbs,
    ubs,
    branch_text,
)
    n = length(lbs)

    return (
        lbs = lbs,
        ubs = ubs,
        local_cuts = deepcopy(node.local_cuts),
        status = MOI.OPTIMIZE_NOT_CALLED,
        parent_id = node.std.id,
        branch_text = branch_text,
        lp_objective = NaN,
        lp_solution = fill(NaN, n),
        integral = false,
        cuts_added_at_node = 0,
        cut_rounds = 0,
        new_cut_names = String[],
    )
end

# ╔═╡ 6dcb9d8a-df7b-41cd-9afa-6babb13dbdf7
function BB.get_branching_nodes_info(
    tree::BB.BnBTree{BCNode,KnapsackBCProblem},
    node::BCNode,
    vidx::Int,
)
    value_at_node = node.lp_solution[vidx]

    left_lbs = copy(node.lbs)
    left_ubs = copy(node.ubs)
    left_ubs[vidx] = floor(Int, value_at_node)

    right_lbs = copy(node.lbs)
    right_ubs = copy(node.ubs)
    right_lbs[vidx] = ceil(Int, value_at_node)

    return [
        child_info(
            node,
            left_lbs,
            left_ubs,
            "x[$vidx] <= $(floor(Int, value_at_node))",
        ),
        child_info(
            node,
            right_lbs,
            right_ubs,
            "x[$vidx] >= $(ceil(Int, value_at_node))",
        ),
    ]
end

# ╔═╡ b9690efc-2197-4305-b68e-e41fcf0d0a1d
md"""
## 9. Logging

The callback records node bounds, incumbent values, and cut activity.
"""

# ╔═╡ 0c3f6e2c-a5eb-4bd8-ac01-dce854dacedb
function current_incumbent(tree)
    if BB.get_num_solutions(tree) == 0
        return missing
    end

    return BB.get_objective_value(tree)
end

# ╔═╡ db34bf7d-7d43-435d-9eb9-afe18035d9f7
function run_branch_and_cut(cut_family::Symbol)
    problem = KnapsackBCProblem(
        copy(profit),
        copy(weight),
        capacity,
        cut_family,
        [1/2, 1/3, 1/4, 1/5, 1/6, 2/5, 2/7, 3/7],
        4,      # maximum size of enumerated minimal covers
        10,     # maximum cut rounds at one B&B node
        4,      # maximum cuts added in one cut round
    )

    tree = BB.initialize(
        branch_strategy = BB.MOST_INFEASIBLE(),
        traverse_strategy = BB.BestFirstSearch(),
        Node = BCNode,
        root = problem,
        sense = :Max,
    )

    BB.set_root!(
        tree,
        (
            lbs = zeros(n),
            ubs = ones(n),
            local_cuts = LinearCut[],
            status = MOI.OPTIMIZE_NOT_CALLED,
            parent_id = 0,
            branch_text = "root",
            lp_objective = NaN,
            lp_solution = fill(NaN, n),
            integral = false,
            cuts_added_at_node = 0,
            cut_rounds = 0,
            new_cut_names = String[],
        ),
    )

    log = NamedTuple[]

    callback = function(tree, node; kwargs...)
        status =
            get(kwargs, :node_infeasible, false) ? :infeasible :
            get(kwargs, :worse_than_incumbent, false) ? :bound :
            node.integral ? :integer :
            :branched

        push!(
            log,
            (
                step = length(log) + 1,
                id = node.std.id,
                parent = node.parent_id,
                depth = node.std.depth,
                restriction = node.branch_text,
                status = status,
                local_bound = node.lp_objective,
                incumbent = current_incumbent(tree),
                cuts_added = node.cuts_added_at_node,
                cut_rounds = node.cut_rounds,
                new_cuts = join(node.new_cut_names, " | "),
                solution = copy(node.lp_solution),
            ),
        )
    end

    BB.optimize!(tree; callback = callback)

    return (
        tree = tree,
        objective = BB.get_objective_value(tree),
        solution = BB.get_solution(tree),
        log = DataFrame(log),
    )
end

# ╔═╡ bcf42f00-7445-4c1b-b6ef-c0d5038197bc
function process_cpu_seconds()
    # Total CPU consumed by the Julia process (all threads), in seconds.
    if Sys.iswindows()
        creation = Ref{UInt64}(0)
        exit_time = Ref{UInt64}(0)
        kernel = Ref{UInt64}(0)
        user = Ref{UInt64}(0)
        handle = ccall((:GetCurrentProcess, "kernel32"), stdcall, Ptr{Cvoid}, ())
        ok = ccall(
            (:GetProcessTimes, "kernel32"), stdcall, Cint,
            (Ptr{Cvoid}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}),
            handle, creation, exit_time, kernel, user,
        )
        ok == 0 && error("GetProcessTimes failed")
        return (Float64(kernel[]) + Float64(user[])) * 1e-7
    else
        # POSIX clock() uses 1,000,000 ticks per second (Linux/macOS).
        ticks = ccall(:clock, Clong, ())
        ticks == -1 && error("CPU clock unavailable")
        return Float64(ticks) / 1_000_000
    end
end

# ╔═╡ 70327c88-275a-4098-8270-41e3a43b49cb
function timed_branch_and_cut(cut_family::Symbol)
    # Untimed full run warms up Julia compilation and the solver for this variant.
    run_branch_and_cut(cut_family)
    GC.gc()
    process_cpu_seconds()  # compile the timing helper before measurement

    cpu_start = process_cpu_seconds()
    wall_start = time_ns()
    result = run_branch_and_cut(cut_family)
    wall_time_seconds = Float64(time_ns() - wall_start) / 1e9
    cpu_time_seconds = process_cpu_seconds() - cpu_start

    return merge(result, (
        cpu_time_seconds = cpu_time_seconds,
        wall_time_seconds = wall_time_seconds,
    ))
end

# ╔═╡ 2bb25b81-aaae-455c-b5c4-99f3b2cfba4d
md"""
## 10. Branch-and-Cut with Chvátal-Gomory cuts
"""

# ╔═╡ 9461ebcd-4f6a-41db-914a-135464f522bd
cg_result = timed_branch_and_cut(:CG)

# ╔═╡ b6a6a8f6-6428-4f11-a272-17557c4c0be6
cg_result.objective, cg_result.solution

# ╔═╡ a7cb0379-fbed-49c1-b072-f89c091c55f2
cg_result.log

# ╔═╡ 58bfd2a0-0c3b-457e-a6ea-1db46bce7c8b
cg_cut_nodes = filter(
    row -> row.cuts_added > 0,
    cg_result.log,
)

# ╔═╡ 1fa0cefa-b056-476a-9650-09e02a9814c4
md"""
The table above identifies all evaluated nodes where new CG cuts were added,
including the number of cutting-plane rounds and the cut descriptions.
"""

# ╔═╡ e35bafcc-0a9e-4ecf-a940-0b3cbc70b24d
md"""
## 11. Branch-and-Cut with extended-cover inequalities
"""

# ╔═╡ 6e7f409b-8a5b-42e7-9bf8-29ebfa8fbd94
cover_result = timed_branch_and_cut(:ExtendedCover)

# ╔═╡ bfbac4c1-1b46-4990-af77-9b7a2536c591
cover_result.objective, cover_result.solution

# ╔═╡ 7f36af45-8942-4f6e-9345-2b9ca304499b
cover_result.log

# ╔═╡ 288c6724-ec14-4b3d-9eae-19f92b3ff1a0
cover_cut_nodes = filter(
    row -> row.cuts_added > 0,
    cover_result.log,
)

# ╔═╡ b05681bf-5bc0-4cbf-b32e-156319c296a1
md"""
The table above identifies nodes where new extended-cover cuts were added.
The separator still enumerates minimal covers of size at most 4. For this
larger-capacity instance, it can become active only after enough capacity
has been consumed by variables fixed to one.
"""

# ╔═╡ c189da3c-1c83-4662-b74b-c2aa0838bca6
md"""
## 12. Direct MIP verification
"""

# ╔═╡ 857391c7-4dfe-404b-bac2-16b3bfb44bf5
function solve_direct_mip(profit, weight, capacity)
    n = length(profit)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:n], Bin)
    @constraint(model, sum(weight[i] * x[i] for i in 1:n) <= capacity)
    @objective(model, Max, sum(profit[i] * x[i] for i in 1:n))

    optimize!(model)

    return (
        objective = objective_value(model),
        solution = value.(x),
    )
end

# ╔═╡ 69b8a4d2-de15-4778-8816-0188d5188f23
direct_mip = solve_direct_mip(profit, weight, capacity)

# ╔═╡ ade7644b-75a1-4c45-a8f0-32229f2be4b3
md"""
The two Branch-and-Cut runs and the direct HiGHS MIP solve should all return
the optimal objective value 212.
"""

# ╔═╡ c5ad9404-8e94-4b26-bdf1-695190e21302
comparison_results = let
    # Counts refer to evaluated nodes recorded by the callback.
    # Each evaluation solves one LP, plus one LP per cut round.
    # Timing covers a full run after an untimed warm-up of the same variant.
    # CPU time includes all Julia-process threads; elapsed time is also reported.
    variants = ("Chvatal-Gomory", "Extended cover")
    results = (cg_result, cover_result)

    DataFrame([
        (
            variant = variant,
            objective = result.objective,
            cpu_time_seconds = result.cpu_time_seconds,
            wall_time_seconds = result.wall_time_seconds,
            solution = round.(result.solution; digits = 6),
            matches_direct_mip = isapprox(
                result.objective, direct_mip.objective;
                atol = 1e-6, rtol = 0,
            ),
            nodes_evaluated = nrow(result.log),
            nodes_branched = count(==(:branched), result.log.status),
            nodes_integer = count(==(:integer), result.log.status),
            nodes_pruned_by_bound = count(==(:bound), result.log.status),
            nodes_infeasible = count(==(:infeasible), result.log.status),
            maximum_depth = maximum(result.log.depth),
            cuts_added = sum(result.log.cuts_added),
            nodes_with_cuts = count(>(0), result.log.cuts_added),
            cut_rounds = sum(result.log.cut_rounds),
            lp_solves = nrow(result.log) + sum(result.log.cut_rounds),
        )
        for (variant, result) in zip(variants, results)
    ])
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Bonobo = "f7b14807-3d4d-461a-888a-05dd4bca8bc3"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Bonobo = "~0.1.5"
Combinatorics = "~1.1.0"
DataFrames = "~1.8.2"
HiGHS = "~1.24.1"
JuMP = "~1.31.1"
MathOptInterface = "~1.52.0"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "2557648dd4429ac0bbfd2870ea385b0aaca928fb"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bonobo]]
deps = ["DataStructures", "NamedTupleTools"]
git-tree-sha1 = "17878aad72be16fecd899699c5a0b26711b46a7d"
uuid = "f7b14807-3d4d-461a-888a-05dd4bca8bc3"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Combinatorics]]
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "1b86cca764a61dcac4fef4c5e16e378e5ed6953c"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.5"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HiGHS]]
deps = ["HiGHS_jll", "LinearAlgebra", "MathOptIIS", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "01a5241985559c08a5baadbcebd6d87daaf84a84"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.24.1"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Zlib_jll", "libblastrampoline_jll"]
git-tree-sha1 = "2d9747b79d17c4320fe48048a3a768fe6d6d82de"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.15.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c7345ab1a7ca4dc8a02c9f6510da0d9857bbe513"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.7.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "614b22ff014355192982b1f9a12c61298ce6a908"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.31.1"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptIIS]]
deps = ["MathOptInterface"]
git-tree-sha1 = "3b3d69130d8ab8c39d5fa4d30e20a8e6428c9d37"
uuid = "8c4f8055-bd93-4160-a86b-a0c04941dbff"
version = "0.2.0"

[[deps.MathOptInterface]]
deps = ["CodecBzip2", "CodecZlib", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "f1ccd9ffcb8577e207deb9aaebeb3f961de70380"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.52.0"

    [deps.MathOptInterface.extensions]
    MathOptInterfaceBenchmarkToolsExt = "BenchmarkTools"
    MathOptInterfaceCliqueTreesExt = "CliqueTrees"

    [deps.MathOptInterface.weakdeps]
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "dc5b2c4c111c46bc79ac4405eeb563523b39c004"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.8.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "30870d0f2dc0b2dba76b10df1c58c7f018413e56"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "3de8f5e6e90ebfa8d6d1f86997d6cdcd6a912ff3"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "4ac881f5432bd93463a41767a814a45245be22b6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.4.6"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "8a90c1d77c3277a5d43b83927b3cbe2c70a37484"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.7"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "0f38a06c83f0007bbab3cf911262841c9a0f07e0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.13.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"
"""

# ╔═╡ Cell order:
# ╟─ad0c1770-319e-4a36-91ad-6a5151153d29
# ╠═cba540a6-51a7-41f3-bb1b-08ea87609ce2
# ╠═54a8acc0-132d-455f-bf65-9fb9dd1405fd
# ╟─5f6acef5-fe13-4f49-a134-1b43588d8431
# ╠═d7415c61-6cd9-4553-a17b-18b00655052f
# ╟─af4d25a6-39bc-4042-bee1-fd4fc57439ca
# ╠═a0dd4fab-b811-42ee-a79d-a358f4ed3db2
# ╠═ab49c6b7-0a72-47bc-abfb-9b5c0c86da66
# ╟─cc916f27-a1a1-4842-97f3-db326ece5fb5
# ╠═a01a6f73-a2ee-4804-8a03-edd7900f20a8
# ╠═8df4be2f-527c-49d1-b4c8-8c6e6ec8c9d1
# ╠═1220962e-d905-4407-91a4-8b36aef61d46
# ╠═52d70135-58d5-4f1b-95c3-17315c10303d
# ╟─5518249f-d936-4202-88cb-b99105dc7bce
# ╠═78a2c083-21a6-4953-afa8-ce196cc00155
# ╠═e87f921c-af24-4220-85d9-c45d65a30fbb
# ╠═4c505906-178e-40c0-bae4-b0158793b736
# ╠═a5b50658-6568-412c-a97a-b8d5b3b24446
# ╟─fb92e385-f39e-485a-a583-72cd17464d91
# ╠═23ce4917-abf6-4c3c-b9af-2337316e2372
# ╟─532f1295-e063-4581-aa85-a8989a7fa634
# ╠═4a9996e7-3f7d-4d68-acaa-6d59af9bbfe8
# ╠═a3130393-e50d-4a2f-9f30-3c348c1cc15a
# ╠═ffeed9e5-f18a-48ac-8ce8-ad8ba31682b3
# ╟─31d6dd81-13a5-44e9-88a0-5f5934c1171e
# ╠═7126d8e0-b2bb-49e6-bb1c-ca340b14b19d
# ╠═59a384c5-69b3-419e-9ac3-e1b9bccf2929
# ╠═e6a5a16a-f639-40dc-b4b4-8a410c67fdb1
# ╟─34f6cfd6-7358-4bcb-b1fc-803e707db554
# ╠═e3a6c5f7-75e5-4434-8a98-c05f05e7d768
# ╠═fcaffbae-36e4-4c70-a1b0-a6fedf48010a
# ╟─12a43db7-dfdd-4b06-9c60-6e90a7109db7
# ╠═be85ab9b-b9cc-4c5b-bb7a-06c44fc2a601
# ╠═6dcb9d8a-df7b-41cd-9afa-6babb13dbdf7
# ╟─b9690efc-2197-4305-b68e-e41fcf0d0a1d
# ╠═0c3f6e2c-a5eb-4bd8-ac01-dce854dacedb
# ╠═db34bf7d-7d43-435d-9eb9-afe18035d9f7
# ╠═bcf42f00-7445-4c1b-b6ef-c0d5038197bc
# ╠═70327c88-275a-4098-8270-41e3a43b49cb
# ╟─2bb25b81-aaae-455c-b5c4-99f3b2cfba4d
# ╠═9461ebcd-4f6a-41db-914a-135464f522bd
# ╠═b6a6a8f6-6428-4f11-a272-17557c4c0be6
# ╠═a7cb0379-fbed-49c1-b072-f89c091c55f2
# ╠═58bfd2a0-0c3b-457e-a6ea-1db46bce7c8b
# ╟─1fa0cefa-b056-476a-9650-09e02a9814c4
# ╟─e35bafcc-0a9e-4ecf-a940-0b3cbc70b24d
# ╠═6e7f409b-8a5b-42e7-9bf8-29ebfa8fbd94
# ╠═bfbac4c1-1b46-4990-af77-9b7a2536c591
# ╠═7f36af45-8942-4f6e-9345-2b9ca304499b
# ╠═288c6724-ec14-4b3d-9eae-19f92b3ff1a0
# ╟─b05681bf-5bc0-4cbf-b32e-156319c296a1
# ╟─c189da3c-1c83-4662-b74b-c2aa0838bca6
# ╠═857391c7-4dfe-404b-bac2-16b3bfb44bf5
# ╠═69b8a4d2-de15-4778-8816-0188d5188f23
# ╟─ade7644b-75a1-4c45-a8f0-32229f2be4b3
# ╠═c5ad9404-8e94-4b26-bdf1-695190e21302
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

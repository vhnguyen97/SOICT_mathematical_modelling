
using Statistics
using Random
using Plots

const goldenratio = (1 + sqrt(5)) / 2

"""
    fp(p, L, a, b)

Compute the penalty-based cost function at subscribed power `p`, for the
consumption record `L` and the constants `a` (fixed cost) and `b` (penalty
weight).
"""
function fp(p, L, a, b)
    s = 0.0
    for i in eachindex(L)
        s += max(0, L[i] - p)^2
    end
    return a * p + b * sqrt(s)
end

"""
    minifp_dicho(a, b, L, eps)

Find the minimizer of `fp` with precision `eps`, using the bisection search
method. Returns a tuple `(pstar, nbeval)` with the approximate minimizer and
the total number of evaluations of `fp`.
"""
function minifp_dicho(a, b, L, eps)
    ak = 0.0
    bk = maximum(L)
    xkG = 0.25 * bk
    xkC = 0.5 * bk
    xkD = 0.75 * bk
    fg = fp(xkG, L, a, b)
    fc = fp(xkC, L, a, b)
    fd = fp(xkD, L, a, b)
    nbeval = 3

    while (bk - ak) > eps  # stopping criterion not yet satisfied
        if fc > fd  # case 1
            ak = xkC
            xkC = xkD
            fc = fd
        elseif fc > fg  # case 2
            bk = xkC
            xkC = xkG
            fc = fg
        else  # case 3
            ak = xkG
            bk = xkD
        end

        # compute the new points
        xkD = ak + 0.75 * (bk - ak)
        xkG = ak + 0.25 * (bk - ak)
        fd = fp(xkD, L, a, b)
        fg = fp(xkG, L, a, b)
        nbeval += 2
    end

    return ak, nbeval
end

# Question 15:

"""
    minifp_nbor(a, b, L, eps)

Find the minimizer of `fp` with precision `eps`, using the golden section
method. Returns a tuple `(pstar, nbeval)` with the approximate minimizer and
the total number of evaluations of `fp`.
"""
function minifp_nbor(a, b, L, eps)
    ak = 0.0
    bk = maximum(L)
    xkG = bk - (1 / goldenratio) * (bk - ak)
    xkD = ak + (1 / goldenratio) * (bk - ak)
    fg = fp(xkG, L, a, b)
    fd = fp(xkD, L, a, b)
    nbeval = 2

    while (bk - ak) > eps  # stopping criterion not yet satisfied
        if fg > fd  # case 1
            ak = xkG
            xkG = xkD
            fg = fd
            # compute the new point
            xkD = ak + (1 / goldenratio) * (bk - ak)
            fd = fp(xkD, L, a, b)
            nbeval += 1
        elseif fd > fg  # case 2
            bk = xkD
            xkD = xkG
            fd = fg
            # compute the new point
            xkG = bk - (1 / goldenratio) * (bk - ak)
            fg = fp(xkG, L, a, b)
            nbeval += 1
        else  # case 3
            ak = xkG
            bk = xkD
            # compute the new points
            xkG = bk - (1 / goldenratio) * (bk - ak)
            xkD = ak + (1 / goldenratio) * (bk - ak)
            fg = fp(xkG, L, a, b)
            fd = fp(xkD, L, a, b)
            nbeval += 2
        end
    end

    return ak, nbeval
end

# Estimate of the average electrical consumption and of its standard
# deviation, depending on the electrical equipment present in a home, for
# several configurations (house/apartment, gas/electric heating, ...).
# For the comparisons we take 20 random samples per configuration.
a = 1
b = 0.5
L = [2500, 3500, 4380, 4389, 4725, 4800, 3700, 3500, 7000, 7500, 2000, 1200]

L_ect = [
    [mean(L), std(L, corrected = false)],
    [2320, 960],
    [17100, 3200],
    [11930, 3870],
    [7000, 1600],
]

"""
    crea_listes(L)

Build a random sample from a mean electrical consumption and a standard
deviation. These two values are given in the vector `L` passed as a
parameter, the mean being the first element and the standard deviation the
second element.
"""
function crea_listes(L)
    res = Int[]  # random sample
    moy = L[1]
    ecart_type = L[2]

    for i in 1:10000
        push!(res, rand(trunc(Int, moy - ecart_type):trunc(Int, moy + ecart_type)))
    end

    return res
end

"""
    compare_methodes(a, b, eps, L_ect)

Compare the running time and the number of evaluations of `fp` of the two
methods (bisection search and golden section).
"""
function compare_methodes(a, b, eps, L_ect)
    Temps_or = Float64[]
    Temps_dicho = Float64[]
    NbEvalOr = Int[]
    NbEvalDicho = Int[]

    for i in 1:5
        for j in 1:20
            L = crea_listes(L_ect[i])  # build the random sample

            # running time of the golden section method
            tps_av = time()
            nbeval = minifp_nbor(a, b, L, eps)[2]
            tps_ap = time()

            push!(Temps_or, tps_ap - tps_av)
            push!(NbEvalOr, nbeval)

            # running time of the bisection method
            tps_av = time()
            nbeval = minifp_dicho(a, b, L, eps)[2]
            tps_ap = time()

            push!(Temps_dicho, tps_ap - tps_av)
            push!(NbEvalDicho, nbeval)
        end
    end

    # average running time and average number of evaluations of f,
    # for each method
    moyenne_eval_dicho = sum(NbEvalDicho) / 100
    moyenne_temps_dicho = sum(Temps_dicho) / 100
    moyenne_eval_or = sum(NbEvalOr) / 100
    moyenne_temps_or = sum(Temps_or) / 100

    println("Bisection method:")
    println("\taverage running time: ", moyenne_temps_dicho)
    println("\taverage number of evaluations of f: ", moyenne_eval_dicho)

    println("Golden section method:")
    println("\taverage running time: ", moyenne_temps_or)
    println("\taverage number of evaluations of f: ", moyenne_eval_or)
end

compare_methodes(a, b, 1e-7, L_ect)

"""
    trace_minifp_dicho(a, b, L, eps)

Build the vectors containing the number of evaluations of `fp` (vector `X`)
and the log10 of the difference between `bk` and `ak` (vector `Y`), for the
bisection search method.
"""
function trace_minifp_dicho(a, b, L, eps)
    ak = 0.0
    bk = maximum(L)
    xkG = 0.25 * bk
    xkC = 0.5 * bk
    xkD = 0.75 * bk
    fg = fp(xkG, L, a, b)
    fc = fp(xkC, L, a, b)
    fd = fp(xkD, L, a, b)
    nbeval = 3
    X, Y = Int[], Float64[]

    while (bk - ak) > eps  # stopping criterion not yet satisfied
        push!(X, nbeval)
        push!(Y, log10(bk - ak))
        if fc > fd  # case 1
            ak = xkC
            xkC = xkD
            fc = fd
        elseif fc > fg  # case 2
            bk = xkC
            xkC = xkG
            fc = fg
        else  # case 3
            ak = xkG
            bk = xkD
        end

        # compute the new points
        xkD = ak + 0.75 * (bk - ak)
        xkG = ak + 0.25 * (bk - ak)
        fd = fp(xkD, L, a, b)
        fg = fp(xkG, L, a, b)
        nbeval += 2
    end

    return X, Y
end

"""
    trace_minifp_nbor(a, b, L, eps)

Build the vectors containing the number of evaluations of `fp` (vector `X`)
and the log10 of the difference between `bk` and `ak` (vector `Y`), for the
golden section method.
"""
function trace_minifp_nbor(a, b, L, eps)
    ak = 0.0
    bk = maximum(L)
    xkG = bk - (1 / goldenratio) * (bk - ak)
    xkD = ak + (1 / goldenratio) * (bk - ak)
    fg = fp(xkG, L, a, b)
    fd = fp(xkD, L, a, b)
    nbeval = 2
    X, Y = Int[], Float64[]

    while (bk - ak) > eps  # stopping criterion not yet satisfied
        push!(X, nbeval)
        push!(Y, log10(bk - ak))
        if fg > fd  # case 1
            ak = xkG
            xkG = xkD
            fg = fd
            # compute the new point
            xkD = ak + (1 / goldenratio) * (bk - ak)
            fd = fp(xkD, L, a, b)
            nbeval += 1
        elseif fd > fg  # case 2
            bk = xkD
            xkD = xkG
            fd = fg
            # compute the new point
            xkG = bk - (1 / goldenratio) * (bk - ak)
            fg = fp(xkG, L, a, b)
            nbeval += 1
        else  # case 3
            ak = xkG
            bk = xkD
            # compute the new points
            xkG = bk - (1 / goldenratio) * (bk - ak)
            xkD = ak + (1 / goldenratio) * (bk - ak)
            fg = fp(xkG, L, a, b)
            fd = fp(xkD, L, a, b)
            nbeval += 2
        end
    end

    return X, Y
end

"""
    compare_convergence(a, b, L, eps)

Compare the convergence of the two methods as a function of the number of
evaluations of `fp`.
"""
function compare_convergence(a, b, L, eps)
    xdicho, ydicho = trace_minifp_dicho(a, b, L, eps)
    xgold, ygold = trace_minifp_nbor(a, b, L, eps)

    plt = plot(xdicho, ydicho, label = "Bisection method convergence")
    plot!(plt, xgold, ygold, label = "Golden section method convergence")
    xlabel!(plt, "Number of evaluations of f")
    ylabel!(plt, "log10(bk - ak)")
    display(plt)
end
compare_convergence(a, b, L_ect[1], 1e-7)
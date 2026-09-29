# =============================================================================
# EXERCICE : Comparaison de performance pour le test de colinéarité en Julia
# =============================================================================
# Objectif : Comparer l'approche par "recherche de composante et division" (Méthode A)
#            avec l'approche par "Inégalité de Cauchy-Schwarz" (Méthode B)
# =============================================================================
using LinearAlgebra
using BenchmarkTools
# -----------------------------------------------------------------------------
# Méthode A : Votre approche (Recherche linéaire de la 1ère composante non nulle)
# -----------------------------------------------------------------------------
function colineaire_recherche(v1::AbstractVector, v2::AbstractVector; atol::Real=1e-7)
    n = length(v1)
    idx = 0
    
    # Étape 1 : Recherche linéaire du premier pivot non nul
    @inbounds for i in 1:n
        if abs(v2[i]) > atol
            idx = i
            break
        end
    end
    
    # Si v2 est entièrement nul
    if idx == 0
        @inbounds for i in 1:n
            if abs(v1[i]) > atol return false end
        end
        return true
    end
    
    # Étape 2 : Calcul de lambda
    lambda = v1[idx] / v2[idx]
    
    # Étape 3 : Vérification du reste des composantes
    @inbounds for i in 1:n
        # Forme optimisée sans division répétée : |v1 - lambda*v2| <= atol
        if abs(v1[i] - lambda * v2[i]) > atol
            return false
        end
    end
    return true
end
# -----------------------------------------------------------------------------
# Méthode B : L'approche purement mathématique (Cauchy-Schwarz vectorisée)
# -----------------------------------------------------------------------------
function colineaire_cauchy(v1::AbstractVector, v2::AbstractVector; atol::Real=1e-7)
    # Calcule 3 produits scalaires (hautement optimisés via SIMD / BLAS)
    dot_12 = dot(v1, v2)
    dot_11 = dot(v1, v1)
    dot_22 = dot(v2, v2)
    
    # Comparaison des carrés
    return abs(dot_12^2 - dot_11 * dot_22) <= atol^2
end

# =============================================================================
# PHASE DE TEST ET BENCHMARK
# =============================================================================
println("="^60)
println("SITUATION 1 : Petits vecteurs (Dim 3) - Cas Colinéaire")
println("="^60)
vec1_small = [1.0, 2.0, 3.0]
vec2_small = [2.0, 4.0, 6.0]
print("Méthode A (Recherche) -> "); @btime colineaire_recherche($vec1_small, $vec2_small)
print("Méthode B (Cauchy)    -> "); @btime colineaire_cauchy($vec1_small, $vec2_small)

println("\n" * "="^60)
println("SITUATION 2 : Grands vecteurs (Dim 5000) - Cas NON Colinéaire (Erreur au début)")
println("="^60)
# Les vecteurs divergent dès la 3ème composante
vec1_large_non = [1.0, 2.0, 99.0, rand(4997)...]
vec2_large_non = [2.0, 4.0,  6.0, rand(4997)...]
print("Méthode A (Recherche) -> "); @btime colineaire_recherche($vec1_large_non, $vec2_large_non)
print("Méthode B (Cauchy)    -> "); @btime colineaire_cauchy($vec1_large_non, $vec2_large_non)

println("\n" * "="^60)
println("SITUATION 3 : Grands vecteurs (Dim 5000) - Cas strictement Colinéaire")
println("="^60)
vec1_large_oui = rand(5000)
vec2_large_oui = vec1_large_oui .* 3.5
print("Méthode A (Recherche) -> "); @btime colineaire_recherche($vec1_large_oui, $vec2_large_oui)
print("Méthode B (Cauchy)    -> "); @btime colineaire_cauchy($vec1_large_oui, $vec2_large_oui)
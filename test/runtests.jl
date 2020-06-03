using Test

using BlockDavidson: davidson!, CPU, State
using LinearAlgebra

function setup(n = 1000; min_dim = 12, max_dim = 24, block_size = 4, evals = 4)
    A = rand(n, n)
    A = (A + A') ./ 2 + Diagonal(1:n)
    B = Diagonal(fill(2.0, n))

    s = State(CPU(), n = n, min_dimension = min_dim, max_dimension = max_dim, evals = evals)

    P = Diagonal(A)

    return s, A, B, P
end

@testset "Davidson" begin
    n = 2000
    min_dim = 12
    max_dim = 36
    block_size = 6
    evals = 6

    s, A, B, P = setup(n, min_dim = min_dim, max_dim = max_dim, block_size = block_size, evals = evals)

    davidson!(s, A, B, P, evals = evals, curr_dim = block_size, block_size = block_size, min_dimension = min_dim, max_dimension = max_dim)

    Φ = s.Φ[:, 1:evals]
    Λ = Diagonal(s.Λ[1:evals])
    @test norm(A * Φ - B * Φ * Λ) < 1e-6
end
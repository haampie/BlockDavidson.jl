using Test

using BlockDavidson
using BlockDavidson: davidson!, CPU, State, DiagonalPreconditioner
using LinearAlgebra

function setup(n = 1000; min_dim = 12, max_dim = 24, block_size = 4, evals = 4)
    A = rand(n, n)
    A = (A + A') ./ 2 + Diagonal(1:n)
    B = Diagonal(fill(2.0, n))

    s = State(CPU(), n = n, min_dimension = min_dim, max_dimension = max_dim, evals = evals)

    P = DiagonalPreconditioner(A, B)

    return s, A, B, P
end

@testset "Davidson" begin
    n = 6000
    min_dim = 40
    max_dim = 200
    block_size = 40
    evals = 100

    s, A, B, P = setup(n, min_dim = min_dim, max_dim = max_dim, block_size = block_size, evals = evals)

    davidson!(s, A, B, P, evals = evals, curr_dim = block_size, block_size = block_size, min_dimension = min_dim, max_dimension = max_dim)

    Φ = s.Φ[:, 1:evals]
    Λ = Diagonal(s.Λ[1:evals])
    R = A * Φ - B * Φ * Λ

    resnorms = map(norm, eachcol(R))

    @test all(x -> x < 1e-4, resnorms)
end
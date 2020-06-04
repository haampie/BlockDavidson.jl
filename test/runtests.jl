using Test

using BlockDavidson
using BlockDavidson: davidson!, CPU, State
using LinearAlgebra

import BlockDavidson: apply_preconditioner!

"""
A and B are supposed to be the diagonal entries of the matrices A and B
"""
struct DiagonalPreconditioner{TA,TB}
    A::TA
    B::TB
end

DiagonalPreconditioner(A::AbstractMatrix, B::AbstractMatrix) = DiagonalPreconditioner(diag(A), diag(B))

smooth(p) = (1 + p + sqrt(1 + (p - 1)^2)) / 2

function BlockDavidson.apply_preconditioner!(y, P::DiagonalPreconditioner, Λ)
    @inbounds for j = axes(y, 2)
        λ = Λ[j]
        for i = axes(y, 1)
            y[i, j] /= smooth(P.A[i] - P.B[i] * λ)
        end
    end

    return y
end

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
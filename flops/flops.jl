using BlockDavidson
using BlockDavidson: davidson!, CPU, State, DiagonalPreconditioner, OpCounter
using LinearAlgebra
using UnicodePlots

import Base: show
using UnicodePlots

function show(io::IO, a::OpCounter) 
    println(io, "Matrix-vector products (#): ", a.matrix_product)
    println(io, "Locks (#): ", a.nlocks)
    println(io, "Restarts (#): ", a.nrestarts)

    bar = barplot(["Locking", "Restarting", "Orthogonalization", "Eigenproblem", "Residual"], 
                  [a.locking, a.restarting, a.orthogonalization, a.eigenproblem, a.residual], 
                  title = "Flop count")

    orth = lineplot(log10.(a.orthogonality), title = "‖Φ'SΦ - I‖₂", xlabel = "iteration", ylabel = "log10(norm)")

    eigen = histogram(a.eigenproblemsize, title = "Eigenproblem dimension")

    println(io, bar)
    println(io, orth)
    println(io, eigen)
end

function setup(n = 1000; min_dim = 12, max_dim = 24, block_size = 4, evals = 4)
    A = randn(n, n)
    A = (A + A') ./ 2 + Diagonal(n+1:2n)
    B = Diagonal(fill(2.0, n))

    s = State(CPU(), n = n, min_dimension = min_dim, max_dimension = max_dim, evals = evals)

    P = DiagonalPreconditioner(A, B)

    return s, A, B, P
end

n = 6000
evals = 150
block_size = 12
min_dimension, max_dimension = block_size, evals + 2block_size
count = OpCounter()
# count = nothing

s, A, B, P = setup(n, min_dim = min_dimension, max_dim = max_dimension, block_size = block_size, evals = evals)

@time davidson!(s, A, B, P, counter = count, evals = evals, curr_dim = block_size, block_size = block_size, min_dimension = min_dimension, max_dimension = max_dimension, locking = true, tolerance = 1e-6)

Φ = s.Φ[:, 1:evals]
Λ = Diagonal(s.Λ[1:evals])
R = A * Φ - B * Φ * Λ

println(histogram(map(x -> log10(norm(x)), eachcol(R)), title = "log₁₀(residual norm)"))
println(histogram(s.Λ[1:evals], title = "Eigenvalue distribution"))

count !== nothing && println(count)
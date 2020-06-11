"""
All state variables
"""
struct State{TV,TAV,TBV,TVAV,TR,TAX,TBX,TRes,TΛ}
    Φ::TV       # Search subspace Φ
    AΦ::TAV     # A * Φ
    BΦ::TBV     # B * Φ
    ΦᵀAΦ::TVAV  # Galerkin projection of A onto Φ
    Ψ::TR       # Ritz vectors Ψ
    AΨ::TAX     # A * Ψ
    BΨ::TBX     # B * Ψ
    R::TRes     # R = A * Ψ - B * Ψ * Λ
    Λ::TΛ       # Ritz values
end

struct CPU end
struct GPU end

"""
Constructor for CPU
"""
State(::CPU; n = 100, block_size = 10, max_dimension = 20, evals = 4) = State(
    rand(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, max_dimension),
    zeros(max_dimension, max_dimension),
    zeros(n, max_dimension),
    zeros(n, block_size),
    zeros(n, block_size),
    zeros(n, block_size),
    zeros(evals)
)

mutable struct OpCounter
    matrix_product::Int
    locking::Int
    nlocks::Int
    restarting::Int
    nrestarts::Int
    orthogonalization::Int
    eigenproblem::Int
    residual::Int
    orthogonality::Vector{Float64}
    eigenproblemsize::Vector{Int}

    OpCounter() = new(0, 0, 0, 0, 0, 0, 0, 0, Float64[], Int[])
end

macro maybe(x, y)
    esc(quote
        if $x !== nothing
            $y
        end
    end)
end

"""
Run the Davidson method to find `evals` eigenvalues of the stencil (A, B).

A preconditioner P is used to approximately solve (A - Bσ)X = R the expansion
of the search subspace. The user has to implement apply_preconditioner! for this.

To start with an initial guess, set curr_dim to the amount of vectors and put the
vectors in s.Φ[:, 1:curr_dim]. They will be reorthogonalized.
"""
function davidson!(s::State, A, B, P; counter::Union{Nothing,OpCounter} = nothing, evals = 4, block_size = 4, num_locked = 0, curr_dim = 4, min_dimension = evals + 2, max_dimension = 10, tolerance = 1e-6, max_iter = size(A, 1) ÷ 10, locking = true)
    mA, nA = size(A)
    mB, nB = size(B)
    T = eltype(s.Φ)

    @assert mA == nA == mB == nB
    @assert size(s.Ψ, 2) ≥ min_dimension
    @assert size(s.AΨ, 2) ≥ block_size
    @assert size(s.BΨ, 2) ≥ block_size
    @assert num_locked ≤ curr_dim
    @assert max_dimension ≤ nA

    block_start = num_locked + 1

    iter = 1

    @views while curr_dim ≤ max_dimension && iter ≤ max_iter
        Φ_b = s.Φ[:, block_start:curr_dim]
        AΦ_b = s.AΦ[:, block_start:curr_dim]
        BΦ_b = s.BΦ[:, block_start:curr_dim]

        # Compute AΦ and BΦ for the new block
        @timeit to "matrix product" begin
            mul!(AΦ_b, A, Φ_b)
            mul!(BΦ_b, B, Φ_b)
        end

        @maybe counter counter.matrix_product += size(Φ_b, 2)

        # Orthogonalization with respect to previous vectors in the search subspace
        Φ_prev = s.Φ[:, 1:block_start - 1]
        AΦ_prev = s.AΦ[:, 1:block_start - 1]
        BΦ_prev = s.BΦ[:, 1:block_start - 1]

        # Orthogonalize a second time.

        # proj = Φ_prev' * BΦ_b

        # @maybe counter counter.orthogonalization += 2 * size(Φ_prev, 1) * size(Φ_prev, 2) * size(BΦ_b, 2)

        # # Gemm
        # mul!(Φ_b, Φ_prev, proj, -one(T), one(T))
        # mul!(AΦ_b, AΦ_prev, proj, -one(T), one(T))
        # mul!(BΦ_b, BΦ_prev, proj, -one(T), one(T))

        # @maybe counter counter.orthogonalization += 3 * 2 * size(Φ_prev, 1) * size(Φ_prev, 2) * size(proj, 2)

        @timeit to "Orthogonalization" begin
            @timeit to "project" proj = Φ_prev' * BΦ_b

            @maybe counter counter.orthogonalization += 2 * size(Φ_prev, 1) * size(Φ_prev, 2) * size(BΦ_b, 2)

            # Gemm
            @timeit to "remove" begin
                mul!(Φ_b, Φ_prev, proj, -one(T), one(T))
                mul!(AΦ_b, AΦ_prev, proj, -one(T), one(T))
                mul!(BΦ_b, BΦ_prev, proj, -one(T), one(T))
            end

            @maybe counter counter.orthogonalization += 3 * 2 * size(Φ_prev, 1) * size(Φ_prev, 2) * size(proj, 2)

            # Interior orthogonalization
            @timeit to "project block" int_proj = Φ_b' * BΦ_b
            @timeit to "cholesky" C = cholesky!(Symmetric(int_proj))

            @timeit to "orth block" begin
                rdiv!(Φ_b, C.L')
                rdiv!(AΦ_b, C.L')
                rdiv!(BΦ_b, C.L')
            end
        end

        @show round.(diag(C.L), sigdigits = 2)

        @maybe counter counter.orthogonalization += 2 * size(Φ_b, 1) * size(Φ_b, 2) * size(BΦ_b, 2) # int proj
        @maybe counter counter.orthogonalization += size(int_proj, 1)^3 ÷ 3 # cholesky
        @maybe counter counter.orthogonalization += 3 * size(Φ_b, 1) * size(Φ_b, 2) * size(C.L', 2) # trmm 3x

        # Update the low-dimensional problem
        @timeit to "low dimensional update" begin
            mul!(s.ΦᵀAΦ[num_locked+1:curr_dim, block_start:curr_dim], s.Φ[:, num_locked+1:curr_dim]', AΦ_b)
            copyto!(s.ΦᵀAΦ[block_start:curr_dim, num_locked+1:block_start-1], s.ΦᵀAΦ[num_locked+1:block_start-1, block_start:curr_dim]')
        end

        # Copy the matrix over and solve the eigenvalue problem
        ΦᵀAΦ_search_subspace = copy(s.ΦᵀAΦ[num_locked+1:curr_dim, num_locked+1:curr_dim])
        @timeit to "solve eigenproblem" eigen = eigen!(Symmetric(ΦᵀAΦ_search_subspace))

        # Approximately 7n^3 flops for all eigenvalues and eigenvectors...
        @maybe counter counter.eigenproblem += 7 * size(ΦᵀAΦ_search_subspace, 1)^3
        @maybe counter push!(counter.eigenproblemsize, size(ΦᵀAΦ_search_subspace, 1))

        # Number of new Ritz vectors to compute.
        num_ritz = block_size
        num_needed = min(num_ritz, evals - num_locked)

        # Compute the Ritz vectors Ψ and AΨ and BΨ.
        @timeit to "compute residual" begin
            @timeit to "compute Ψ"  mul!( s.Ψ[:, 1:num_ritz],  s.Φ[:, num_locked+1:curr_dim], eigen.vectors[:, 1:num_ritz])
            @timeit to "compute AΨ" mul!(s.AΨ[:, 1:num_ritz], s.AΦ[:, num_locked+1:curr_dim], eigen.vectors[:, 1:num_ritz])
            @timeit to "compute BΨ" mul!(s.BΨ[:, 1:num_ritz], s.BΦ[:, num_locked+1:curr_dim], eigen.vectors[:, 1:num_ritz])

            @maybe counter counter.residual += 3 * 2 * size(s.Φ[:, num_locked+1:curr_dim], 1) * size(s.Φ[:, num_locked+1:curr_dim], 2) * size(eigen.vectors[:, 1:num_ritz], 2)

            # Save the Ritz values / eigenvalues
            copyto!(s.Λ[num_locked+1:num_locked+num_needed], eigen.values[1:num_needed])

            # Copy the residual R = B * Ψ * Λ - A * Ψ
            @timeit to "compute R" begin
                copyto!(s.R[:, 1:num_ritz], s.BΨ[:, 1:num_ritz])
                rmul!(s.R[:, 1:num_ritz], Diagonal(eigen.values[1:num_ritz]))
                s.R[:, 1:num_ritz] .-= s.AΨ[:, 1:num_ritz]
            end

            @maybe counter counter.residual += 2 * size(s.R[:, 1:num_ritz], 1) * size(s.R[:, 1:num_ritz], 2)

            # Compute residual norms
            @timeit to "compute norm(R)" residual_norms = [norm(s.R[:, i]) for i = 1 : num_ritz]
        end

        @show curr_dim iter round.(residual_norms, sigdigits = 2)

        @maybe counter counter.residual += 2 * size(s.R[:, 1:num_ritz], 1) * size(s.R[:, 1:num_ritz], 2)

        # Count the number of converged vectors and number of lockable vectors (with stricter tolerance)
        search_converged = findlast(x -> x ≤ tolerance, residual_norms)
        num_converged = search_converged === nothing ? 0 : search_converged
        num_unconverged = num_ritz - num_converged

        should_restart = curr_dim + num_unconverged > max_dimension
        everything_converged = num_converged + num_locked ≥ evals
        should_lock = locking && num_converged ≥ block_size ÷ 2

        should_change_basis = should_lock || everything_converged || should_restart
    
        # Lock converged vectors and shrink the subspace when necessary        
        if should_change_basis
            # Search subspace is divided into [locked][newly_locked][search space][free space]
            # `1:new_curr_dim` is the range we want to keep including already locked vectors.
            # - In case everything has converged, we should just keep 1:evals vecs.
            # - In case of just a restart, keep min_dimension more vecs than the number of locked ones, including newly locked guys
            # - In case of just locking, keep the full search subspace as is, but just do a change of basis.
            new_curr_dim = if everything_converged
                evals
            elseif should_restart
                num_locked + num_converged + min_dimension
            else # only lock
                curr_dim
            end

            # We have already computed AΨ, BΨ and Ψ for `num_ritz` vectors and `1:num_locked` don't have to be recomputed,
            # so here we compute just the remainder `num_ritz+1:keep`
            keep = new_curr_dim - num_locked

            @timeit to "lock/restart" begin
                # Copy the num_ritz vectors already computed
                mul!(s.Ψ[:, num_ritz+1:keep], s.Φ[:, num_locked+1:curr_dim], eigen.vectors[:, num_ritz+1:keep])
                copyto!(s.Φ[:, num_locked+1:num_locked+keep], s.Ψ[:, 1:keep])

                # Make a change of basis for the rest
                mul!(s.Ψ[:, num_ritz+1:keep], s.AΦ[:, num_locked+1:curr_dim], eigen.vectors[:, num_ritz+1:keep])
                copyto!(s.AΦ[:, num_locked+1:num_locked+num_ritz], s.AΨ[:, 1:num_ritz])
                copyto!(s.AΦ[:, num_locked+num_ritz+1:num_locked+keep], s.Ψ[:, num_ritz+1:keep])

                mul!(s.Ψ[:, num_ritz+1:keep], s.BΦ[:, num_locked+1:curr_dim], eigen.vectors[:, num_ritz+1:keep])
                copyto!(s.BΦ[:, num_locked+1:num_locked+num_ritz], s.BΨ[:, 1:num_ritz])
                copyto!(s.BΦ[:, num_locked+num_ritz+1:num_locked+keep], s.Ψ[:, num_ritz+1:keep])

                # "Compute" the new Galerkin projection, just a diagonal matrix of Ritz values
                ΦᵀAΦ = s.ΦᵀAΦ[num_locked+1:num_locked+keep, num_locked+1:num_locked+keep]
                fill!(ΦᵀAΦ, 0)
                copyto!(ΦᵀAΦ[diagind(ΦᵀAΦ)], eigen.values[1:keep])
            end

            if should_restart || everything_converged
                @maybe counter counter.restarting += 3 * 2 * size(s.AΦ[:, num_locked+1:curr_dim], 1) * size(s.AΦ[:, num_locked+1:curr_dim], 2) * size(eigen.vectors[:, num_ritz+1:keep], 2)
                @maybe counter counter.nrestarts += 1
            else
                @maybe counter counter.locking += 3 * 2 * size(s.AΦ[:, num_locked+1:curr_dim], 1) * size(s.AΦ[:, num_locked+1:curr_dim], 2) * size(eigen.vectors[:, num_ritz+1:keep], 2)
                @maybe counter counter.nlocks += 1
            end

            if locking
                num_locked += num_converged
            end

            curr_dim = new_curr_dim

            everything_converged && break
        end

        # println("Converged: ", locking ? num_locked : num_converged, "/", evals)

        # @maybe counter push!(counter.orthogonality, norm(s.Φ[:, 1:curr_dim]' * s.BΦ[:, 1:curr_dim] - I))

        # Copy the non-converged residual vectors over to the search subspace
        Φ_next = s.Φ[:, curr_dim+1:curr_dim+num_unconverged]
        @timeit to "prep new block" copyto!(Φ_next, s.R[:, num_converged+1:num_ritz])

        # Apply the preconditioner
        @timeit to "precondition" apply_preconditioner!(Φ_next, P, eigen.values[num_converged+1:num_ritz])

        # And increment the pointers
        block_start = curr_dim + 1
        curr_dim += num_unconverged

        iter += 1
    end

    return counter === nothing ? (A, B, s) : (A, B, s, counter)
end
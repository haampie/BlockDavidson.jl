module BlockDavidson

using LinearAlgebra

"""
By default the preconditioner is identity
"""
apply_preconditioner!(y, P, Λ) = y

"""
All state variables
"""
struct State{TV,TAV,TBV,TVAV,TVBV,TR,TAX,TBX,TRes,TΛ}
    Φ::TV       # Search subspace Φ
    AΦ::TAV     # A * Φ
    BΦ::TBV     # B * Φ
    ΦᵀAΦ::TVAV  # Galerkin projection of A onto Φ
    ΦᵀBΦ::TVBV  # temporary
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
State(::CPU; n = 100, min_dimension = 10, max_dimension = 20, evals = 4) = State(
    rand(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, max_dimension),
    zeros(n, min_dimension),
    zeros(evals)
)

"""
Enforce A' == A
"""
function force_symmetry!(A)
    n, m = size(A)
    @assert n == m
    for i = 1 : n, j = i + 1 : n
        @inbounds A[i, j] = A[j, i]
    end
    return A
end

"""
Run the Davidson method to find `evals` eigenvalues of the stencil (A, B).

A preconditioner P is used to approximately solve (A - Bσ)X = R the expansion
of the search subspace. The user has to implement apply_preconditioner! for this.

To start with an initial guess, set curr_dim to the amount of vectors and put the
vectors in s.Φ[:, 1:curr_dim]. They will be reorthogonalized.
"""
function davidson!(s::State, A, B, P; evals = 4, block_size = 4, num_locked = 0, curr_dim = 4, min_dimension = evals + 2, max_dimension = 10, tolerance = 1e-6)
    mA, nA = size(A)
    mB, nB = size(B)
    T = eltype(s.Φ)

    @assert mA == nA == mB == nB
    @assert size(s.Ψ, 2) ≥ min_dimension
    @assert size(s.AΨ, 2) ≥ evals
    @assert size(s.BΨ, 2) ≥ evals
    @assert num_locked ≤ curr_dim
    @assert max_dimension ≤ nA

    block_start = num_locked + 1

    @views while curr_dim ≤ max_dimension
        Φ_b = s.Φ[:, block_start:curr_dim]
        AΦ_b = s.AΦ[:, block_start:curr_dim]
        BΦ_b = s.BΦ[:, block_start:curr_dim]

        # Compute AΦ and BΦ for the new block
        mul!(AΦ_b, A, Φ_b)
        mul!(BΦ_b, B, Φ_b)

        # Orthogonalization with respect to previous vectors in the search subspace
        Φ_prev = s.Φ[:, 1:block_start - 1]
        AΦ_prev = s.AΦ[:, 1:block_start - 1]
        BΦ_prev = s.BΦ[:, 1:block_start - 1]

        # Orthogonalization: Φ_b := (I - Φ_prev * Φ_prev' * B) * Φ_b.
        # Computed as: Φ_b := Φ_b - Φ_prev * (Φ_prev' * (BΦ_b))
        #              AΦ_b := AΦ_b - (AΦ_prev)(Φ_prev' * (BΦ_b))
        #              BΦ_b := BΦ_b - (BΦ_prev)(Φ_prev' * (BΦ_b))

        proj = Φ_prev' * BΦ_b

        # Gemm
        mul!(Φ_b, Φ_prev, proj, -one(T), one(T))
        mul!(AΦ_b, AΦ_prev, proj, -one(T), one(T))
        mul!(BΦ_b, BΦ_prev, proj, -one(T), one(T))

        # Interior orthogonalization
        int_proj = Φ_b' * BΦ_b
        force_symmetry!(int_proj)
        C = cholesky!(int_proj)
        rdiv!(Φ_b, C.L')
        rdiv!(AΦ_b, C.L')
        rdiv!(BΦ_b, C.L')

        # Update the low-dimensional problem
        mul!(s.ΦᵀAΦ[num_locked+1:curr_dim, block_start:curr_dim], s.Φ[:, num_locked+1:curr_dim]', AΦ_b)
        copyto!(s.ΦᵀAΦ[block_start:curr_dim, num_locked+1:block_start-1], s.ΦᵀAΦ[num_locked+1:block_start-1, block_start:curr_dim]')

        # Copy the matrix over and solve the eigenvalue problem
        copyto!(s.ΦᵀBΦ, s.ΦᵀAΦ)
        eigen = eigen!(Symmetric(s.ΦᵀBΦ[num_locked+1:curr_dim, num_locked+1:curr_dim]))

        # Number of new Ritz vectors to compute.
        num_ritz = min(block_size, evals - num_locked)

        # Compute the Ritz vectors
        mul!(s.Ψ[:, 1:num_ritz], s.Φ[:, num_locked+1:curr_dim], eigen.vectors[:, 1:num_ritz])

        # Save the Ritz values / eigenvalues
        copyto!(s.Λ[num_locked+1:num_locked+num_ritz], eigen.values[1:num_ritz])

        # Compute ingredients for the residual block
        mul!(s.AΨ[:, 1:num_ritz], s.AΦ[:, num_locked+1:curr_dim], eigen.vectors[:, 1:num_ritz])
        mul!(s.BΨ[:, 1:num_ritz], s.BΦ[:, num_locked+1:curr_dim], eigen.vectors[:, 1:num_ritz])

        # Copy the residual R = B * Ψ * Λ - A * Ψ
        copyto!(s.R[:, 1:num_ritz], s.BΨ[:, 1:num_ritz])
        rmul!(s.R[:, 1:num_ritz], Diagonal(eigen.values[1:num_ritz]))
        s.R[:, 1:num_ritz] .-= s.AΨ[:, 1:num_ritz]

        # Compute residual norms
        residual_norms = [norm(s.R[:, i]) for i = 1 : num_ritz]
        @show residual_norms

        # Count the number of converged vectors
        search_converged = findlast(x -> x ≤ tolerance, residual_norms)
        num_converged = search_converged === nothing ? 0 : search_converged
        num_unconverged = num_ritz - num_converged

        # Lock converged vectors and shrink the subspace when necessary
        if num_converged > 0 || curr_dim + num_unconverged > max_dimension
            # We have to remove the converged Ritz vectors from the search subspace.
            # Let's assume it's feasible to compute all eigenvectors. If not, we can maybe
            # just QR a random matrix or so with the first few columns the pre-Ritz vectors.

            # When the search subspace has to be shrunken, we keep min_dimension of them, excluding converged ones
            new_curr_dim = curr_dim + num_unconverged > max_dimension ? num_locked + num_converged + min_dimension : curr_dim

            keep = new_curr_dim - num_locked - num_ritz

            # Update the search subspace
            range = num_locked+1:num_locked+num_ritz+keep

            mul!(s.AΨ[:, num_ritz+1:num_ritz+keep], s.AΦ[:, num_locked+1:curr_dim], eigen.vectors[:, num_ritz+1:num_ritz+keep])
            copyto!(s.AΦ[:, range], s.AΨ[:, 1:num_ritz+keep])

            mul!(s.BΨ[:, num_ritz+1:num_ritz+keep], s.BΦ[:, num_locked+1 : curr_dim], eigen.vectors[:, num_ritz+1:num_ritz+keep])
            copyto!(s.BΦ[:, range], s.BΨ[:, 1:num_ritz+keep])

            mul!(s.Ψ[:, num_ritz+1:num_ritz+keep], s.Φ[:, num_locked+1 : curr_dim], eigen.vectors[:, num_ritz+1:num_ritz+keep])
            copyto!(s.Φ[:, range], s.Ψ[:, 1:num_ritz+keep])

            # "Compute" the new Galerkin projection, just a diagonal matrix of Ritz values
            ΦᵀAΦ = s.ΦᵀAΦ[num_locked+1:num_locked+num_ritz+keep, num_locked+1:num_locked+num_ritz+keep]
            fill!(ΦᵀAΦ, 0)
            copyto!(ΦᵀAΦ[diagind(ΦᵀAΦ)], eigen.values[1:num_ritz+keep])

            num_locked += num_converged
            curr_dim = new_curr_dim
        end

        # Copy the non-converged residual vectors over to the search subspace
        Φ_next = s.Φ[:, curr_dim+1:curr_dim+num_unconverged]
        copyto!(Φ_next, s.R[:, num_converged+1:num_ritz])

        num_locked ≥ evals && break

        # Apply the preconditioner
        apply_preconditioner!(Φ_next, P, s.Λ[num_locked + 1])

        # And increment the pointers
        block_start = curr_dim + 1
        curr_dim += num_unconverged
    end

    return A, B, s
end

end

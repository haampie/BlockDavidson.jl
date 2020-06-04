"""
Identity preconditioner for the Davidson method.
Implement your own preconditioner type and this function
to approximately solve (A - Bσ)X = R
"""
apply_preconditioner!(y, P, Λ) = y

"""
A and B are supposed to be the diagonal entries of the matrices A and B
"""
struct DiagonalPreconditioner{TA,TB}
    A::TA
    B::TB
end

DiagonalPreconditioner(A::AbstractMatrix, B::AbstractMatrix) = DiagonalPreconditioner(diag(A), diag(B))

smooth(p) = (1 + p + sqrt(1 + (p - 1)^2)) / 2

function apply_preconditioner!(y, P::DiagonalPreconditioner, Λ)
    @inbounds for j = axes(y, 2)
        λ = Λ[j]
        for i = axes(y, 1)
            y[i, j] /= smooth(P.A[i] - P.B[i] * λ)
        end
    end

    return y
end
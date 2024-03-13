"""
    spblkdiag(A)

Construct a block diagonal sparse matrix from a vector of dense or sparse matrices.

# Inputs
- `A::Vector{<:AbstractMatrix}`: Vector of matrices to be stacked.

# Output
- `M::SparseMatrixCSC`: Block diagonal matrix.

# Example
```julia-repl
julia> A = [rand(3, 3) for i in 1:3]
julia> M = spblkdiag(A)
```
"""
function spblkdiag(A)
    p = length(A)
    m, n = size(A[1])
    M = spzeros(eltype(A[1]), m*p, n*p)
    @inbounds for i in 1:p
        M[(i-1)*m+1:i*m, (i-1)*n+1:i*n] .= A[i]
    end
    return M
end

function spblkdiag2(A)
    p = length(A)
    m, n = size(A[1])
    mn = m*n
    ix = Vector{Int64}(undef, mn*p)
    iy = Vector{Int64}(undef, mn*p)
    vM = Vector{eltype(A[1])}(undef, mn*p)
    @views @inbounds for i in 1:p
        idM = (1:mn) .+ (i-1)*mn
        idx = (1:m) .+ (i-1)*m
        idy = (1:n) .+ (i-1)*n

        idi, idj = ndgrid(idx, idy)
        ix[idM] .= idi[:]
        iy[idM] .= idj[:]
        vM[idM] .= A[i][:]
    end
    return sparse(ix, iy, vM)
end

"""
    condest(A)

Estimate the condition number of a matrix `A` using the 2-norm.

# Input
- `A::AbstractMatrix`: Matrix to be analyzed.

# Output
- `κ::Float64`: Condition number of `A`.

# Example
```julia-repl
julia> A = rand(3, 3)
julia> κ = condest(A)
```
"""
function condest(A)
    # Initialization
    n, m = size(A)

    if n != m
        error("Matrix must be square")
    end

    F = lu(A)
    k = findall(abs.(diag(F.U)) == 0.)

    if !isempty(k)
        return Inf
    else
        x₀ = sum(abs, A, dims = 1)'
        Ainv_norm = normestinv(F, x₀)
        A_norm = norm(A)
    end

    return A_norm*Ainv_norm
end

function normestinv(F, x)

    maxiter = 100
    tol = 1e-6
    cnt = 0
    est = norm(x)

    f(x) = F\x
    f_t(x) = F'\x

    if est == 0.
        return est
    end

    x /= est

    est0 = 0.
    while abs(est - est0) > tol*est && cnt < maxiter
        est0 = est
        Sx = f(x)

        if iszero(Sx)
            Sx = rand(eltype(Sx), size(Sx))
        end

        x = f_t(Sx)
        normx = norm(x)
        est = normx/norm(Sx)
        x /= normx

        cnt += 1
    end

    return est
end

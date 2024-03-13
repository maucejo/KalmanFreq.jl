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

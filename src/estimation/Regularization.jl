@with_kw struct RVRProblem
    A
    b
    R
end

function solve(prob :: RVRProblem)
    return RVR(prob.A, prob.b, prob.R)
end

@with_kw struct LqRegProblem
    A
    b
    q
    R
    type
    method

    function LqRegProblem(A, b, q, R; type = :mult, method = :lc)
        return new(A, b, q, R, type, method)
    end
end

function solve(prob :: LqRegProblem)
    return lq_reg(prob.A, prob.b, prob.q, prob.R, type = prob.type, method = prob.method)
end

@with_kw struct LpqRegProblem
    A
    b
    p
    q
    R
end

function solve(prob :: LpqRegProblem)
    return lpq_reg(prob.A, prob.b, prob.p, prob.q, prob.R)
end


"""
    RVR_(A, b)

Computes the solution of the system `Ax = b` using the Reelevant Vector Regression method.

# Inputs
- `A`: matrix of the system - Matrix{Complex{Float64}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]

# Output
- `X`: solution of the system - Vector{Complex{Float64}}
- `Λ`: regularization matrix - Matrix{Complex{Float64}}

# Example
```julia-repl
julia> X, Λ = RVR_(A, b)
```
"""
function RVR_(A, b)
    # Initialization
    α = 1.
    β = 1e-18
    tol = 1e-8
    maxit = 200

    Atb = A'*b
    # Initial solution
    λ = norm(2Atb, Inf)
    X₀ = tikhonov(A, b, λ)

    # Preallocation
    nᵤ = size(A, 2)
    X = Vector{eltype(b)}(undef, nᵤ)
    Λ = Diagonal{Float64}(undef, nᵤ)
    indm = diagind(Λ)
    τ = Vector{Float64}(undef, nᵤ)

    # Precomputation
    AtA = A'*A

    # Compute solution
    crit = 1.
    ee = 0
    while (crit > tol) && (ee ≤ maxit)
        @. τ = α/(β + abs2(X₀))
        Λ[indm] .= τ

        X .= (AtA + Λ)\Atb

        crit = norm(X - X₀, 1)/norm(X₀, 1)
        ee += 1
        X₀ .= X
    end

    return X, Λ
end

function RVR(A, b, R)
    S = Diagonal(sqrt.(1 ./diag(R)))

    if isa(b, Vector)
        nx = size(A, 2)
        X = typeof(b)(undef, nx)
        X .= RVR_(S*A, S*b)[1]
    else
        nx = size(A[1], 2)
        nf = length(A)
        X = typeof(b)(undef, nx, nf)

        SA = Ref(S).*A # Equivalent to [S*a for a in A]
        Sb = S*b

        p = Progress(nf, desc = "RVR...", showspeed = true)
        @inbounds @views for f in eachindex(A)
            next!(p)
            X[:, f] .= RVR_(SA[f], Sb[:, f])[1]
        end
    end

    return X
end

"""
    lq_reg_(A, b, q[, R])

Computes the solution of the system `Ax = b` using the additive lq-regularization method.

# Inputs
- `A`: matrix of the system - Matrix{Complex{Float64}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]
- `q`: value of desired norm - Float64
- `R`: noise covariance matrix - Matrix{Float64}

# Output
- `X`: solution of the system - Vector{Complex{Float64}}

# Example
```julia-repl
julia> X = lq_reg_(A, b, q)
```
"""
function lq_reg_(A, b, q)
    # Initialisation
    Atb = A'*b
    λ = norm(2Atb, Inf)
    X₀ = tikhonov(A, b, λ)

    # Precomputation
    nx = size(A, 2)
    X = Vector{eltype(b)}(undef, nx)
    AtA = A'*A
    epsr = calc_eps(X₀, 0.05)

    # Preallocation
    W = Diagonal{Float64}(undef, nx)
    indm = diagind(W)

    # Compute solution
    tol = 1e-8
    maxit = 200
    crit = 1
    ee = 1
    while (crit > tol) && (ee ≤ maxit)
        W[indm] .= @. (q/2.)*max(abs(X₀), epsr)^(q - 2.)

        λ = norm(b - A*X₀)^2/norm(X₀, q)^q
        X .= (AtA + λ*W)\Atb

        crit = norm(X - X₀, 1)/norm(X₀, 1)
        ee += 1
        X₀ .= X
    end

    return X
end

"""
    lq_reg(A, b, q[, R])

Computes the solution of the system `Ax = b` using the additive lq-regularization method.

# Inputs
- `A`: matrix of the system - Matrix{Complex{Float64}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]
- `q`: value of desired norm - Float64
- `R`: noise covariance matrix - Matrix{Float64}

# Output
- `X`: solution of the system - Vector{Complex{Float64}}

# Example
```julia-repl
julia> X = lq_reg(A, b, q)
"""
function lq_reg(A, b, q, R = I(size(b, 1)); type = :mult, method = :lc)
    S = Diagonal(sqrt.(1 ./diag(R)))

    if isa(b, Vector)
        nx = size(A, 2)
        X = typeof(b)(undef, nx)
        X .= lq_reg_(S*A, S*b, q, type = type, method = method)
    else
        # Preallocation
        nx = size(A[1], 2)
        nf = length(A)
        X = typeof(b)(undef, nx, nf)

        SA = Ref(S).*A # Equivalent to [S*a for a in A]
        Sb = S*b

        p = Progress(nf, desc = "lq-regularization...", showspeed = true)
        @inbounds @views for f in eachindex(A)
            next!(p)
            X[:, f] .= lq_reg_(SA[f], Sb[:, f], q)
        end
    end

    return X
end

"""
    lpq_reg(A, b, p, q[, R])

Computes the solution of the system `Ax = b` using the additive lpq-regularization method.

# Inputs
- `A`: matrix of the system - Vector{Matrix{Complex{Float64}}}
- `b`: right-hand side of the system - Matrix{Complex{Float64}]
- `p, q`: values of desired mixed norm - Float64
- `R`: noise covariance matrix - Matrix{Float64}

# Output
- `X`: solution of the system - Matrix{Complex{Float64}}

# Example
```julia-repl
julia> X = lpq_reg(A, b, p, q)
julia> X = lpq_reg(A, b, p, q, R)
```
"""
function lpq_reg(A, b, p, q)
    # Precomputation
    nf = length(A)
    nb, nx = size(A[1])
    if nb != nx
        Af = spblkdiag2(A)
    else
        Af = BlockDiagonal(A)
    end

    bf = b[:]
    AtA = Af'*Af
    Atb = Af'*bf

    # Initialisation
    # λ = init_lambda(Af, 200)
    λ = norm(2Atb, Inf)
    X₀ = tikhonov(Af, bf, λ)
    epsr = calc_eps(X₀, 0.05)

    # Preallocation
    X = Vector{eltype(b)}(undef, nx*nf)
    W = Diagonal{Float64}(undef, nx*nf)
    indm = diagind(W)

    # Compute solution
    tol = 1e-8
    maxit = 50
    crit = 1
    ee = 1
    while (crit > tol) && (ee ≤ maxit)
        w, s = fw(X₀, p, q, epsr, nx, nf)
        W[indm] .= w

        λ = norm(bf - Af*X₀)^2/s
        X .= (AtA + λ*W)\Atb

        crit = norm(X - X₀, 1)/norm(X₀, 1)
        ee += 1
        X₀ .= X
    end

    return reshape(X, nx, nf)
end

function lpq_reg(A, b, p, q, R)
    S = Diagonal(sqrt.(1 ./diag(R)))

    SA = Ref(S).*A # Equivalent to [S*a for a in A]
    Sb = S*b

    return lpq_reg(SA, Sb, p, q)
end

"""
    tikhonov(A, b, λ)

Computes the solution of the system `Ax = b` using the Tikhonov regularization method.

# Inputs
- `A`: matrix of the system - Matrix{Complex{Float64}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]
- `λ`: regularization parameter - Float64

# Output
- `X`: solution of the system - Vector{Complex{Float64}}

# Example
```julia-repl
julia> X = tikhonov(A, b, λ)
```
"""
function tikhonov(A, b, λ)
    # Calculation of the solution
    X = (A'*A + λ*I)\(A'*b)

    return X
end

"""
    calc_eps(X, percentage)

Computes the damping parameter `eps` of the problem for avoiding infinite weights.

# Inputs
- `X`: Solution of the system - Vector{Complex{Float64}}
- `percentage`: percentage of the cumulative histogram - Float64

# Output
- `eps`: damping parameter - Float64

# Example
```julia-repl
julia> eps = calc_eps(X, 0.05)
```
"""
function calc_eps(X, percentage)
    # Initialization
    absx = abs.(X)

    minbin = minimum(absx)
    maxbin = maximum(absx)
    bins = range(minbin, maxbin, length = 1000)

    h = fit(Histogram, absx, bins)
    edges = h.edges[1]
    hacc = cumsum(Float64.(h.weights))
    hacc ./= hacc[end]

    center_bins = edges[1:end-1] .+ (diff(edges) ./ 2)
    # Compute eps
    eps = center_bins[findfirst(x -> x ≥ percentage, hacc)]

    return eps
end

"""
    fw(X, p, q, epsr, nx, nf)

Computes the weights for the mixed-norm regularization.

# Inputs
- `X`: solution of the system - Vector{Complex{Float64}}
- `p, q`: values of desired mixed norm - Float64
- `epsr`: damping parameter - Float64
- `nx, nf`: number of locations and frequencies - Int64

# Output
- `W`: weights for the mixed-norm regularization - Vector{Float64}
- `s`: value of the mixed-norm - Float64

# Example
```julia-repl
julia> W, s = fw(X, p, q, epsr, nx, nf)
```
"""
function fw(X, p, q, epsr, nx, nf)
    Xtemp = reshape(abs.(X), nx, nf)

    Wf = @. max(epsr, Xtemp)^(p - 2.)

    normp = sum(Xtemp.^p, dims = 2)

    Ws = @. (q/2.)*max(epsr, normp)^(q/p - 1.)

    W = (Ws.*Wf)[:]

    s = sum(normp.^(q/p))

    return W, s
end
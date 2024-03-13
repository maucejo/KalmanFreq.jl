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
    # F = svd(A)
    # λ = reg_param(F, b, :lc)
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

        p = Progress(nf, desc = "RVR...", showspeed = true, color = :black)
        @inbounds @views for f in eachindex(A)
            next!(p)
            X[:, f] .= RVR_(SA[f], Sb[:, f])[1]
        end
    end

    return X
end

"""
    lq_reg_(A, b, q[, R]; type, method)

Computes the solution of the system `Ax = b` using the additive lq-regularization method.

# Inputs
- `A`: matrix of the system - Matrix{Complex{Float64}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]
- `q`: value of desired norm - Float64
- `R`: noise covariance matrix - Matrix{Float64}
- `type`: type of regularization - Symbol
    - `:add`: additive regularization
    - `:mult`: multiplicative regularization
- `method`: method used to compute the regularization parameter - Symbol
    - `:bayes`: Bayesian method
    - `:gcv`: Generalized Cross-Validation method
    - `:lc`: L-curve method

# Output
- `X`: solution of the system - Vector{Complex{Float64}}

# Example
```julia-repl
julia> X = lq_reg_(A, b, q)
```
"""
function lq_reg_(A, b, q; type = :mult, method = :lc)
    # Initialisation
    Atb = A'*b
    if type == :mult
        # λ = init_lambda(A, 200)
        λ = norm(2Atb, Inf)
        X₀ = tikhonov(A, b, λ)
    else
        F = svd(A)
        λ = reg_param(F, b, method)
        X₀ = tikhonov(F, b, λ)
    end

    if q == 2 && type != :mult
        return X₀
    end

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

        if type == :mult
            λ = norm(b - A*X₀)^2/norm(X₀, q)^q
        else
            F = svd(A, sqrt.(W))
            λ = reg_param(F, b, method)
        end
        X .= (AtA + λ*W)\Atb

        crit = norm(X - X₀, 1)/norm(X₀, 1)
        ee += 1
        X₀ .= X
    end

    return X
end

"""
    lq_reg(A, b, q[, R]; type, method)

Computes the solution of the system `Ax = b` using the additive lq-regularization method.

# Inputs
- `A`: matrix of the system - Matrix{Complex{Float64}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]
- `q`: value of desired norm - Float64
- `R`: noise covariance matrix - Matrix{Float64}
- `type`: type of regularization - Symbol
    - `:add`: additive regularization
    - `:mult`: multiplicative regularization
- `method`: method used to compute the regularization parameter - Symbol
    - `:bayes`: Bayesian method
    - `:gcv`: Generalized Cross-Validation method
    - `:lc`: L-curve method

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

        p = Progress(nf, desc = "Lq-regularization...", showspeed = true, color = :black)
        @inbounds @views for f in eachindex(A)
            next!(p)
            X[:, f] .= lq_reg_(SA[f], Sb[:, f], q, type = type, method = method)
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
- `F`: SVD of the system matrix `A` - SVD{Complex{Float64}, Matrix{Complex{Float64}}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]
- `λ`: regularization parameter - Float64

# Output
- `X`: solution of the system - Vector{Complex{Float64}}

# Example
```julia-repl
julia> X = tikhonov(A, b, λ)
```
"""
function tikhonov(F::SVD, b, λ)
    # Calculation of the solution
    U, σ, V = F
    y = U'*b
    Σ = Diagonal(@. σ/(σ^2 + λ))

    return V*Σ*y
end

function tikhonov(A, b, λ)
    # Calculation of the solution
    X = (A'*A + λ*I)\(A'*b)

    return X
end

"""
    reg_param(F, b, method = :lc)

Computes the regularization parameter of the problem.

# Inputs
- `F`: singular value decomposition of the matrix `A` - SVD{Complex{Float64}, Matrix{Complex{Float64}}}
- `b`: right-hand side of the system - Vector{Complex{Float64}]
- `method`: method used to compute the regularization parameter - Symbol
    - `:bayes`: Bayesian method
    - `:gcv`: Generalized Cross-Validation method
    - `:lc`: L-curve method

# Output
- `λc`: regularization parameter - Float64

# Example
```julia-repl
julia> λc = reg_param(F, b, :lc)
```
"""
function reg_param(F, b, method = :lc) :: Float64
    # Initialization
    npoints = 200
    σₘ = 16eps()
    m, n = size(F.U)
    σ = Vector{Float64}(undef, n)
    β = F.U'*b
    if hasproperty(F, :D1)
        m = minimum(size(F.D1))
        @views σ .= real(diag(F.D1[1:m, 1:m])./diag(F.D2[1:m, 1:m]))
    else
        σ .= F.S
    end

    # Estimation
    σ₁ = maximum(σ)
    vmax = max(minimum(σ), σ₁*σₘ)
    ratio = (σ₁/vmax)^(1/(npoints - 1))
    Λ = vmax*ratio.^(0:(npoints - 1))

    g = Vector{Float64}(undef, npoints)
    if method == :bayes
        for (i, λ) in enumerate(Λ)
            g[i] = bayes_func(λ, σ, β)
        end
        pos_min = argmin(g)
        res = optimize(L -> bayes_func(L, σ, β), Λ[max(pos_min - 1, 1)], Λ[min(pos_min + 1, npoints)])

    elseif method == :gcv
        δ₀ = 0.
        y2 = norm(b)^2 - norm(β)^2;
        mn = m - n
        if (m > n) && (y2 > 0.)
            δ₀ = y2
        end
        for (i, λ) in enumerate(Λ)
            g[i] = gcv_func(λ, σ, β, δ₀, mn)
        end
        pos_min = argmin(g)
        res = optimize(L -> gcv_func(L, σ, β, δ₀, mn), Λ[max(pos_min - 1, 1)], Λ[min(pos_min + 1, npoints)])

    elseif method == :lc
        ξ = β./σ

        for (i, λ) in enumerate(Λ)
            g[i] = lc_func(λ, σ, β, ξ)
        end

        pos_min = argmin(g)
        res = optimize(L -> lc_func(L, σ, β, ξ), Λ[max(pos_min - 1, 1)], Λ[min(pos_min + 1, npoints)])
    end

    return Optim.minimizer(res)^2
end

"""
    bayes_func(λ, σ, β)

Computes the functional to be minizimed using the `bayes` method.

# Inputs
- `λ`: regularization parameter - Float64
- `σ`: singular values of the matrix `A` - Vector{Float64}
- `β`: Projection of right-hand side of the system on the left singular vectors - Vector{Complex{Float64}}

# Output
- `J`: Bayesian functional - Float64

# Example
```julia-repl
julia> J = bayes_func(λ, σ, β)
```
"""
function bayes_func(λ, σ, β)
    M = length(β)
    γ = @. σ^2 + λ^2
    α² = mean(abs2.(β)./γ)

    return sum(log, γ) + (M - 2.)*log(α²)
end

"""
    gcv_func(λ, σ, β, δ₀)

Computes the functional to be minizimed using the `gcv` method.

# Inputs
- `λ`: regularization parameter - Float64
- `σ`: singular values of the matrix `A` - Vector{Float64}
- `β`: right-hand side of the system - Vector{Complex{Float64}}
- `δ₀`: Intrinsic residual - Float64
- `mn`: difference of the number of columns and rows of the left singular vectors - Int64

# Output
- `J`: GCV functional - Float64

# Example
```julia-repl
julia> J = gcv_func(λ, σ, β)
```
"""
function gcv_func(λ, σ, β, δ₀, mn)
    # Square systems
    f = @. λ^2/(σ^2 + λ^2)

    return (norm(f.*β)^2 + δ₀)/(mn + sum(f))^2
end

"""
    lc_func(λ, σ, β, ξ)

Computes the functional to be minizimed using the `lcurve` method.

# Inputs
- `λ`: regularization parameter - Float64
- `σ`: singular values of the matrix `A` - Vector{Float64}
- `β`: right-hand side of the system - Vector{Complex{Float64}}
- `ξ`: right-hand side of the system - Vector{Complex{Float64}}

# Output
- `J`: L-curve functional - Float64

# Example
```julia-repl
julia> J = lc_func(λ, σ, β, ξ)
```
"""
function lc_func(λ, σ, β, ξ)
    f = @. σ^2/(σ^2 + λ^2)
    cf  = 1. .- f
    η = norm(f.*ξ)
    ρ = norm(cf.*β)
    f₁ = @. -2f*cf/λ
    f₂ = @. -f₁*(3. - 4f)/λ
    ϕ = sum(@. f*f₁*abs(ξ)^2)
    ψ = sum(@. cf*f₁*abs(β)^2)
    dϕ = sum(@. (f₁^2 + f*f₂)*abs(ξ)^2)
    dψ = sum(@. (cf*f₂ - f₁^2)*abs(β)^2)

    # Compute the first and second derivatives of η and ρ w.r.t. λ
    dη = ϕ/η
    dρ = -ψ/ρ
    d²η = (dϕ/η) - dη*(dη/η)
    d²ρ = -(dψ/ρ) - dρ*(dρ/ρ)

    # Convert to derivatives of log(η) and log(ρ)
    dlogη = dη/η
    dlogρ = dρ/ρ
    d²logη = (d²η/η) - dlogη^2
    d²logρ = (d²ρ/ρ) - dlogρ^2

    # Let g = curvature
    g = -(dlogρ*d²logη - d²logρ*dlogη)/(dlogρ^2 + dlogη^2)^(1.5)

    return g
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
    init_lambda(A, npoints)

Computes the initial value of the regularization parameter `λ` of the problem.

# Inputs
- `A`: matrix of the system - Matrix{Complex{Float64}}
- `npoints`: number of points for the discretization of the regularization parameter - Int64

# Output
- `λ`: initial value of the regularization parameter - Float64

# Example
```julia-repl
julia> λ = init_lambda(A, npoints)
```
"""
function init_lambda(A, npoints)
    # Initialization
    σₘ = 16eps()
    B = A'*A

    # Approximation of the singular values
    σ₁ = sqrt(norm(B, 1)*norm(B, Inf)) # Largest singular value
    σₚ = σ₁/condest(B) # Smallest singular value

    vmax = max(σₚ, σ₁*σₘ)
    ratio = (σ₁/vmax)^(1/(npoints - 1))
    Λ = vmax*ratio.^(0:(npoints - 1))

    return median(Λ)
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

function RE(xref, xid)
    return 100norm(xref - xid, 1)/norm(xref, 1)
end

function Corr(xref, xid)
    return 100abs(dot(xref, xid))/(norm(xref)*norm(xid))
end
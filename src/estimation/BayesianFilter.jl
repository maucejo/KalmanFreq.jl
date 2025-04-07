"""
    InitialConditions

Structure for storing the initial conditions of the Bayesian filter

# Inputs
- `x`: initial state - Vector{ComplexF64}
- `u`: initial input - Vector{ComplexF64}
- `Px`: initial covariance of the state - Matrix{ComplexF64}
- `Pu`: initial covariance of the input - Matrix{ComplexF64}
- `Pxu`: initial cross-covariance of the state and the input - Matrix{ComplexF64}

```julia-repl
julia> ic = InitialConditions(x, u, Px, Pu, Pxu)
```
"""
@with_kw struct InitialConditions
    x₀ :: Vector{ComplexF64}
    u₀ :: Vector{ComplexF64}
    P₀ˣ :: Matrix{ComplexF64}
    P₀ᵘ :: Matrix{ComplexF64}
    P₀ˣᵘ :: Matrix{ComplexF64}
end

@with_kw struct InputEstimation
    ûₖ :: Vector{ComplexF64}
    Kₖᵘ :: Matrix{ComplexF64}
    P̃ₖᵘ :: Matrix{ComplexF64}
end

"""
    BayesianFilterProblem

Structure for storing the problem of Bayesian filtering

# Inputs
- `H`: transfer function - Array{Matrix{ComplexF64}}
- `y`: measurement - Matrix{ComplexF64}
- `Q`: covariance matrix of the process noise - Matrix{ComplexF64}
- `R`: covariance matrix of the measurement noise - Matrix{ComplexF64}
- `C`: output matrix - Matrix{ComplexF64}

# Outputs
- `y`: measurement - Matrix{ComplexF64}
- `ss`: state space model - StateSpace
- `ic`: initial conditions - InitialConditions

```julia-repl
julia> prob = BayesianFilterProblem(H, y, Q, R, C)
```
"""
@with_kw struct BayesianFilterProblem
    y
    ss :: StateSpace
    ic :: InitialConditions

    function BayesianFilterProblem(H, y, Q, R, C = I(size(H[1], 1)))
        # Construction of the state space model
        ss = StateSpace(H, Q, R, C)

        # Calculation of the initial input solution
        u₀, P₀ᵘ = init_input(C*H[2], y[:, 1], R)

        # Calculation of the reduced state
        x̄₀, P̄₀ˣ, P̄₀ˣᵘ = init_reduced_state(u₀, P₀ᵘ, H[2], H[1])

        # Conditions initiales
        ic = InitialConditions(x̄₀, u₀, P̄₀ˣ, P₀ᵘ, P̄₀ˣᵘ)

        return new(y, ss, ic)
    end
end

"""
    BayesianFilterSolution

Structure for storing the results of Bayesian filtering

# Inputs
- `x`: estimated state - Matrix{ComplexF64}
- `u`: estimated input - Matrix{ComplexF64}
- `Pˣ`: estimated covariance of the state - Array{Matrix{ComplexF64}}
- `Pᵘ`: estimated covariance of the input - Array{Matrix{ComplexF64}}
- `Pˣᵘ`: estimated cross-covariance of the state and the input - Array{Matrix{ComplexF64}}

```julia-repl
julia> bfsol = BayesianFilterSolution(prob)
```
"""
@with_kw struct BayesianFilterSolution
    x :: Matrix{ComplexF64}
    u :: Matrix{ComplexF64}
    Pˣ :: Vector{Matrix{ComplexF64}}
    Pᵘ :: Vector{Matrix{ComplexF64}}
    Pˣᵘ :: Vector{Matrix{ComplexF64}}
end

function solve(prob :: BayesianFilterProblem)
    return bfilter(prob.y, prob.ss, prob.ic)
end

"""
    bfilter(y, ss, ic)

Bayesian filtering of a linear system.

# Inputs
- `y`: measurement - Matrix{ComplexF64}
- `ss`: state space model - StateSpace
- `ic`: initial conditions - InitialConditions

# Output
- `bf`: BayesianFilter

# Example
```julia-repl
julia> bf = bfilter(y, ss, ic)
```
"""
function bfilter(y, ss, ic)
    # Initialisation
    nx = length(ic.x₀) # Nombre de variables d'état
    nu = length(ic.u₀) # Nombre de variables d'entrée
    ny, nf = size(y) # Nombre de fréquences
    (; B, D, Q, R, C) = ss
    Iₓ = I(nx)
    Iᵤ = I(nu)

    # Pre-allocations

    # Output
    ûₖ = Matrix{ComplexF64}(undef, nu, nf)
    x̂ₖ = Matrix{ComplexF64}(undef, nx, nf)
    Pₖᵘ = [Matrix{ComplexF64}(undef, nu, nu) for _ in 1:nf]
    Pₖˣ = [Matrix{ComplexF64}(undef, nx, nx) for _ in 1:nf]
    Pₖˣᵘ = [Matrix{ComplexF64}(undef, nx, nu) for _ in 1:nf]

    # Intermediate variables
    Sₖ = Matrix{ComplexF64}(undef, ny, ny)
    iₖᵘ = Vector{ComplexF64}(undef, ny)
    iₖˣ = Vector{ComplexF64}(undef, ny)
    Tu = Matrix{ComplexF64}(undef, nu, nu)
    Tx = Matrix{ComplexF64}(undef, nx, nx)
    PC = Matrix{ComplexF64}(undef, nx, ny)

    # Application of the initial conditions
    ûₖ[:, 1] .= ic.u₀
    x̂ₖ[:, 1] .= ic.x₀
    Pₖᵘ[1] .= ic.P₀ᵘ
    Pₖˣ[1] .= ic.P₀ˣ
    Pₖˣᵘ[1] .= ic.P₀ˣᵘ

    Bₖ = B[1]
    x̃ₖ = ic.x₀ .+ Bₖ*ic.u₀
    BP = Bₖ*ic.P₀ˣᵘ'
    P̃ₖ = ic.P₀ˣ .+ BP .+ BP' .+ Bₖ*ic.P₀ᵘ*Bₖ' + Q

    # Filtering loop
    p = Progress(nf - 1, desc = "Bayesian filtering...", showspeed = true)
    @views @inbounds for k ∈ 2:nf
        next!(p)
        Bₖ .= B[k]
        Dₖ = D[k]

        # 1 - Estimation de l'entrée
        iₖᵘ .= y[:, k] .- C*x̃ₖ
        ie = input_estimation(Dₖ, iₖᵘ, R, ûₖ[:, k - 1])
        ûₖ[:, k] .= ie.ûₖ
        Tu .= Iᵤ .- ie.Kₖᵘ*Dₖ
        PC .= P̃ₖ*C'
        Sₖ .= C*PC .+ R
        Pₖᵘ[k] .= Tu*ie.P̃ₖᵘ*Tu' .+ ie.Kₖᵘ*Sₖ*ie.Kₖᵘ'

        # 2 - Estimation de l'état
        iₖˣ .= iₖᵘ .- Dₖ*ie.ûₖ
        Kₖˣ = PC/Sₖ
        x̂ₖ[:, k] .= x̃ₖ .+ Kₖˣ*iₖˣ
        Tx .= Iₓ .- Kₖˣ*C
        Sₖ .= Dₖ*ie.P̃ₖᵘ*Dₖ' .+ R
        Pₖˣ[k] .= Tx*P̃ₖ*Tx' .+ Kₖˣ*Sₖ*Kₖˣ'
        Pₖˣᵘ[k] .= -Kₖˣ*Dₖ*Pₖᵘ[k]

        # 3 - Prédiction de l'état
        x̃ₖ .= x̂ₖ[:, k] .+ Bₖ*ûₖ[:, k]
        BP .= Bₖ*Pₖˣᵘ[k]'
        P̃ₖ .= Pₖˣ[k] .+ BP .+ BP'.+ Bₖ*Pₖᵘ[k]*Bₖ' + Q
    end

    return BayesianFilterSolution(x̂ₖ, ûₖ, Pₖˣ, Pₖᵘ, Pₖˣᵘ)
end

"""
    input_estimation(D, iₖ, R, u₀)

Estimate the input `u` of a linear system from the measurement `iₖ` using Bayesian denoising.

# Inputs
- `D`: feedthrough matrix - Matrix{ComplexF64}
- `iₖ`: innovation vector - Vector{ComplexF64}
- `R`: measurement noise covariance matrix - Matrix{ComplexF64}
- `u₀`: initial guess of the input - Vector{ComplexF64}

# Output
- `û`: estimated input - Vector{ComplexF64}
- `Kᵤ`: Kalman gain - Matrix{ComplexF64}
- `P̃ᵤ`: prior covariance of the input estimation error - Matrix{ComplexF64}

# Example
```julia-repl
julia> û, Kᵤ, P̃ᵤ = input_estimation(D, iₖ, R, u₀)
```
"""
function input_estimation(D, iₖ, R, u₀)
    # Initialisation
    ny = length(iₖ)
    nu = length(u₀)

    tol = 1e-8
    maxit = 100
    nu = length(u₀)

    αₜ = 1.
    βₜ = 1e-18

    S = Diagonal{Float64}(undef, ny)
    inds = diagind(S)

    S[inds] .= 1 ./sqrt.(diag(R))
    Dᵣ = S*D
    y = S*iₖ

    û₀ = Vector{eltype(iₖ)}(undef, nu)
    û = Vector{eltype(iₖ)}(undef, nu)
    τ = Vector{Float64}(undef, nu)
    Λ = Diagonal{Float64}(undef, nu)
    indm = diagind(Λ)

    # Mise en cache
    û₀ .= u₀
    Dᴴ = Dᵣ'
    DᴴD = Dᴴ*Dᵣ
    Dᴴy = Dᴴ*y

    crit = 1
    ee = 1
    while (crit > tol) && (ee ≤ maxit)
        # Calcul de τ
         @. τ = αₜ/(βₜ + abs2(û₀))
         Λ[indm] .= τ

        # Calcul de û
        û .= (DᴴD + Λ)\Dᴴy

        # Calcul du critère d'arrêt
        crit = norm(û - û₀, 1)/norm(û₀, 1)
        û₀ .= û
        ee += 1
    end

    Kᵤ = (DᴴD + Λ)\(Dᴴ*S)
    P̃ᵤ = Diagonal(1 ./τ)

    return InputEstimation(û, Kᵤ, P̃ᵤ)
end
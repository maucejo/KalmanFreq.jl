"""
    KalmanFilterProblem

Structure for storing the data of a Kalman filtering problem

# Inputs
- `H`: transfer function - Array{Matrix{ComplexF64}}
- `y`: measurement - Matrix{ComplexF64}
- `Q`: covariance matrix of the process noise - Matrix{ComplexF64}
- `R`: covariance matrix of the measurement noise - Matrix{ComplexF64}

# Outputs
- ss: state space model - StateSpace
- y: measurement - Matrix{ComplexF64}
- ic: initial conditions - InitialConditions


```julia-repl
julia> prob = KalmanFilterProblem(H, y, Q, R)
```
"""
@with_kw struct KalmanFilterProblem
    y
    ss
    ic

    function KalmanFilterProblem(H, y, Q, R)
        # Construction of the state space model
        ss = (C = H, R = R)

        # Calculation of the initial input solution
        x₀, P₀ = init_input(H[1], y[:, 1], R)

        # Conditions initiales
        ic = (x₀ = x₀, P₀ = P₀, Q = Q)

        return new(y, ss, ic)
    end
end

"""
    KalmanFilterSolution

Structure for storing the results of Kalman filtering

# Inputs
- `x`: estimated state - Matrix{ComplexF64}
- `P`: estimated covariance of the state - Array{Matrix{ComplexF64}}
- `Q`: estimated covariance matrix of the process noise - Array{Matrix{ComplexF64}}

```julia-repl
julia> kfsol = KalmanFilter(u)
```
"""
@with_kw struct KalmanFilterSolution
    x :: Matrix{ComplexF64}
    P :: Vector{Matrix{ComplexF64}}
    Q :: Vector{Matrix{ComplexF64}}
end

"""
    KalmanSmootherSolution

Structure for storing the results of Kalman smoothing

# Inputs
- `x`: smoothed state - Matrix{ComplexF64}
- `P`: smoothed covariance of the state - Array{Matrix{ComplexF64}}

```julia-repl
julia> kssol = KalmanSmoother(x, P)
```
"""
@with_kw struct KalmanSmootherSolution
    x :: Matrix{ComplexF64}
    P :: Vector{Matrix{ComplexF64}}
end

"""
    kfilter(y, ss, ic)

Computes the Kalman filter of the state-space model `ss` using the measurement `y` and the initial conditions `ic`.

# Inputs
- `y`: measurement - Matrix{ComplexF64}
- `ss`: state-space model - StateSpace
- `ic`: initial conditions - InitialConditions

# Output
- `kfsol`: Kalman filter - KalmanFilterSolution

# Example
```julia-repl
julia> kfsol = kfilter(y, ss, ic)
```
"""
function kfilter(y, ss, ic)
    # Initialisation
    nx = length(ic.x₀)  # Dimension of the state
    ny, nf = size(y)    # Dimension of the measurement matrix
    R = ss.R            # State space model
    Iₓ = I(nx)

    # Pre-allocations
    xₖ = Matrix{ComplexF64}(undef, nx, nf)
    Pₖ = [Matrix{ComplexF64}(undef, nx, nx) for _ in 1:nf]
    Qₖ = [Matrix{ComplexF64}(undef, nx, nx) for _ in 1:nf]

    # Intermediate variables
    x̃ₖ = Vector{ComplexF64}(undef, nx)
    P̃ₖ = Matrix{ComplexF64}(undef, nx, nx)
    Cₖ = Matrix{ComplexF64}(undef, ny, nx)
    Sₖ = Matrix{ComplexF64}(undef, ny, ny)
    iₖ = Vector{ComplexF64}(undef, ny)
    Kₖ = Matrix{ComplexF64}(undef, nx, ny)
    T = Matrix{ComplexF64}(undef, nx, nx)
    PC = Matrix{ComplexF64}(undef, nx, ny)
    Ωₖ = Matrix{ComplexF64}(undef, nx, nx)
    εₖ = Vector{ComplexF64}(undef, nx)

    # Application of the initial conditions
    xₖ[:, 1] .= ic.x₀
    Pₖ[1] .= ic.P₀
    Qₖ[1] .= ic.Q
    Ωₖ .= ic.Q
    ρₖ = 0.

    # Filtering loop
    p = Progress(nf - 1, desc = "Kalman filtering...", showspeed = true, color = :black)
    @views @inbounds for k ∈ 2:nf
        next!(p)
        Cₖ .= ss.C[k]

        # 1 - State prediction
        x̃ₖ .= xₖ[:, k-1]
        @. P̃ₖ = Pₖ[k-1] + Qₖ[k-1]

        # 2 - State estimation
        PC .= P̃ₖ*Cₖ'
        iₖ .= y[:, k] .- Cₖ*x̃ₖ        # Innovation
        Sₖ .= Cₖ*PC .+ R             # Innovation covariance

        Kₖ .= PC/Sₖ                  # Kalman gain
        T .= Iₓ .- Kₖ*Cₖ
        εₖ .= Kₖ*iₖ                  # State estimation error
        @. xₖ[:, k] = x̃ₖ + εₖ        # State estimation
        Pₖ[k] .= T*P̃ₖ*T' .+ Kₖ*R*Kₖ' # Covariance estimation

        # 3 - Process noise estimation
        Ωₖ .= Ωₖ .+ εₖ*εₖ'
        ρₖ += 1.
        @. Qₖ[k] = Ωₖ/(nx + ρₖ + 1.)
    end

    return KalmanFilterSolution(xₖ, Pₖ, Qₖ)
end

function solve(prob :: KalmanFilterProblem)
    return kfilter(prob.y, prob.ss, prob.ic)
end

function ksmoother(kf :: KalmanFilterSolution)
    # Initialisation
    nx, nf = size(kf.x)

    # Pre-allocations
    x̂ₖₛ = Matrix{ComplexF64}(undef, nx, nf)
    Pₖₛ = [Matrix{ComplexF64}(undef, nx, nx) for _ in 1:nf]

    # Intermediate variables
    xₖ  = Vector{ComplexF64}(undef, nx)
    Pₖ  = Matrix{ComplexF64}(undef, nx, nx)
    x̃ₖₛ = Vector{ComplexF64}(undef, nx)
    P̃ₖₛ = Matrix{ComplexF64}(undef, nx, nx)
    Gₖ = Matrix{ComplexF64}(undef, nx, nx)

    # Initialisation
    x̂ₖₛ[:, end] .= kf.x[:, end]
    Pₖₛ[end] .= kf.P[end]

    p = Progress(nf - 1, desc = "Bayesian smoother", showspeed = true, color = :black)
    @views @inbounds for k ∈ (nf - 1):-1:1
        next!(p)

        xₖ .= kf.x[:, k]
        Pₖ .= kf.P[k]

        # 1 - State prediction
        x̃ₖₛ .= xₖ
        @. P̃ₖₛ = Pₖ + kf.Q[k]

        # 2 - State smoothing
        Gₖ .= Pₖ/P̃ₖₛ
        x̂ₖₛ[:, k] .= xₖ .+ Gₖ*(x̂ₖₛ[:, k + 1] .- x̃ₖₛ)
        Pₖₛ[k] .= Pₖ .+ Gₖ*(Pₖₛ[k + 1] .- P̃ₖₛ)*Gₖ'
    end

    return KalmanSmootherSolution(x̂ₖₛ, Pₖₛ)
end

function flsmoother(prob :: KalmanFilterProblem, lag = 10.)
    # Initialisation
    # Problem unpacking
    y = prob.y
    (; C, R) = prob.ss
    (; x₀, P₀, Q) = prob.ic
    nx = length(prob.ic.x₀)  # Dimension of the state
    ny, nf = size(y)    # Dimension of the measurement matrix
    Iₓ = I(nx)

    # Pre-allocations
    x̂ₖₛ = Matrix{ComplexF64}(undef, nx, nf)
    Pₖₛ = [Matrix{ComplexF64}(undef, nx, nx) for _ in 1:nf]

    # Intermediate variables
    x̃ₖ = Vector{ComplexF64}(undef, nx)
    P̃ₖ = Matrix{ComplexF64}(undef, nx, nx)
    Cₖ = Matrix{ComplexF64}(undef, ny, nx)
    Sₖ = Matrix{ComplexF64}(undef, ny, ny)
    iₖ = Vector{ComplexF64}(undef, ny)
    Kₖ = Matrix{ComplexF64}(undef, nx, ny)
    T = Matrix{ComplexF64}(undef, nx, nx)
    PC = Matrix{ComplexF64}(undef, nx, ny)
    Ωₖ = Matrix{ComplexF64}(undef, nx, nx)
    εₖ = Vector{ComplexF64}(undef, nx)
    xₛ = Vector{ComplexF64}(undef, nx)
    Pₛ = Matrix{ComplexF64}(undef, nx, nx)
    Pᵢ = Matrix{ComplexF64}(undef, nx, nx)

    # Initialization
    x̂ₖₛ[:, 1] = x₀
    Pₖₛ[1] .= P₀
    xₖ = x₀
    Pₖ = P₀
    Qₖ = Matrix{eltype(y)}(Q)
    Ωₖ .= Q
    ρₖ = 0.

    # Filtering step
    p = Progress(nf - 1, desc = "Fixed-lag smoothing...", showspeed = true, color = :black)
    @views @inbounds for k ∈ 2:nf
        next!(p)
        Cₖ .= C[k]

        # 1 - State prediction
        x̃ₖ .= xₖ
        @. P̃ₖ = Pₖ + Qₖ

        # 2 - State estimation
        PC .= P̃ₖ*Cₖ'
        iₖ .= y[:, k] .- Cₖ*x̃ₖ        # Innovation
        Sₖ .= Cₖ*PC .+ R             # Innovation covariance

        Kₖ .= PC/Sₖ                  # Kalman gain
        T .= Iₓ .- Kₖ*Cₖ
        εₖ .= Kₖ*iₖ                  # State estimation error
        @. xₖ = x̃ₖ + εₖ        # State estimation
        Pₖ .= T*P̃ₖ*T' .+ Kₖ*R*Kₖ' # Covariance estimation

        # Smoothing pass
        xₛ .= xₖ
        Pₛ .= P̃ₖ
        Pᵢ .= P̃ₖ
        for _ ∈ 1:lag
            Pₛ .= Pₛ .- PC*Kₖ'
            Pᵢ .= Pᵢ*T'
            xₛ .= xₛ .+ Kₖ*iₖ
            PC .= Pᵢ*Cₖ'
            Kₖ .= PC/Sₖ
        end

        x̂ₖₛ[:, k] .= xₛ
        Pₖₛ[k] .= Pₛ

        # 4 - Process noise estimation
        Ωₖ .= Ωₖ .+ εₖ*εₖ'
        ρₖ += 1.
        @. Qₖ = Ωₖ/(nx + ρₖ + 1.)
    end

    return KalmanSmootherSolution(x̂ₖₛ, Pₖₛ)
end

# function flsmoother2(prob :: KalmanFilterProblem, lag = 10.)
#     # Problem unpacking
#     y = prob.y
#     (; C, R) = prob.ss
#     xₖ, Pₖ, Qₖ = prob.ic.x₀, prob.ic.P₀, Matrix{eltype(y)}(prob.ic.Q)

#     # Initialisation
#     nx = length(xₖ)  # Dimension of the state
#     ny, nf = size(y)  # Dimension of the measurement matrix

#     # Pre-allocations
#     x̂ₖₛ = Matrix{ComplexF64}(undef, nx, nf)
#     Pₖₛ = [Matrix{ComplexF64}(undef, nx, nx) for _ in 1:nf]

#     # Intermediate variables
#     xₛ = Vector{ComplexF64}(undef, nx)
#     Pₛ = Matrix{ComplexF64}(undef, nx, nx)
#     Pᵢ = Matrix{ComplexF64}(undef, nx, nx)
#     iₖ = Vector{ComplexF64}(undef, ny)
#     Cₖ = Matrix{ComplexF64}(undef, nx, nx)
#     Kₖ = Matrix{ComplexF64}(undef, nx, ny)
#     Sₖ = Matrix{ComplexF64}(undef, ny, ny)
#     PC = Matrix{ComplexF64}(undef, nx, ny)
#     T = Matrix{ComplexF64}(undef, nx, nx)
#     εₖ = Vector{ComplexF64}(undef, nx)
#     ρₖ = 0.

#     p = Progress(nf - 1, desc = "Fixed-lag smoother", showspeed = true, color = :black)
#     for k ∈ 1:nf
#         next!(p)

#         # Constant variables
#         Cₖ .= C[k]
#         iₖ .= y[:, k] .- Cₖ*xₖ      # Innovation
#         PC .= Pₖ*Cₖ'
#         Sₖ .= Cₖ*PC .+ R           # Innovation covariance

#         # One-step filtering
#         Kₖ .= PC/Sₖ
#         εₖ .= Kₖ*iₖ
#         @. xₖ = xₖ + εₖ

#         # Smoothing initialization
#         xₛ .= xₖ
#         Pₛ .= Pₖ
#         Pᵢ .= Pₖ
#         T .= I - Kₖ*Cₖ

#         for _ ∈ 1:lag
#             Pₛ .= Pₛ .- PC*Kₖ'
#             Pᵢ .= Pᵢ*T'
#             xₛ .= xₛ .+ Kₖ*iₖ
#             PC .= Pᵢ*Cₖ'
#             Kₖ .= PC/Sₖ
#         end

#         x̂ₖₛ[:, k] .= xₛ
#         Pₖₛ[k] .= Pₛ

#         # Prepare the next iteration
#         Qₖ .= Qₖ .+ εₖ*εₖ'
#         ρₖ += 1.
#         @. Qₖ = Qₖ/(nx + ρₖ + 1.)
#         Pₖ .= Pₖ*T' .+ Qₖ
#     end

#     return KalmanSmootherSolution(x̂ₖₛ, Pₖₛ)
# end


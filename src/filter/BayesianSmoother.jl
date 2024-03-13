"""
    BayesianSmootherSolution

Structure for storing the results of Bayesian smoothing

# Inputs
- `x`: smoothed state - Matrix{ComplexF64}
- `u`: smoothed input - Matrix{ComplexF64}
- `Pˣ`: smoothed covariance of the state - Array{Matrix{ComplexF64}}
- `Pᵘ`: smoothed covariance of the input - Array{Matrix{ComplexF64}}

```julia-repl
julia> bf = BayesianSmoother(u)
```
"""
@with_kw struct BayesianSmootherSolution
    x :: Matrix{ComplexF64}
    u :: Matrix{ComplexF64}
    Pˣ :: Vector{Matrix{ComplexF64}}
    Pᵘ :: Vector{Matrix{ComplexF64}}
    Pˣᵘ :: Vector{Matrix{ComplexF64}}
end

"""
    bsmoother(ss, bf)

Computes the Bayesian smoother of the state-space model `ss` using the Bayesian filter `bf`.

# Inputs
- `ss`: state-space model - StateSpace
- `bf`: Bayesian filter - BayesianFilter

# Output
- `bs`: Bayesian smoother - BayesianSmoother

# Example
```julia-repl
julia> bs = bsmoother(ss, bf)
```
"""
@views function bsmoother(ss :: StateSpace, bf :: BayesianFilterSolution)
    # Initialisation
    nu, nf = size(bf.u)
    nx = size(ss.B[1], 1)
    Q = ss.Q
    Iₓ = I(nx)

    # Pre-allocations
    ûₖₛ = Matrix{ComplexF64}(undef, nu, nf)
    x̂ₖₛ = Matrix{ComplexF64}(undef, nx, nf)
    xₐ = Vector{ComplexF64}(undef, nx + nu)
    x̃ₐ = Vector{ComplexF64}(undef, nx)
    Aₐ = Matrix{ComplexF64}(undef, nx, nx + nu)
    Pₐ = Matrix{ComplexF64}(undef, nx + nu, nx + nu)
    P̂ₐ = Matrix{ComplexF64}(undef, nx + nu, nx + nu)
    P̃ₓ = Matrix{ComplexF64}(undef, nx, nx)
    Pₖₛˣ = [Matrix{ComplexF64}(undef, nx, nx) for _ in 1:nf]
    Pₖₛᵘ = [Matrix{ComplexF64}(undef, nu, nu) for _ in 1:nf]
    Pₖₛˣᵘ = [Matrix{ComplexF64}(undef, nx, nu) for _ in 1:nf]

    # Initialisation
    x̂ₖₛ[:, end] = bf.x[:, end]
    Pₖₛˣ[end] = bf.Pˣ[end]

    ûₖₛ[:, end] = bf.u[:, end]
    Pₖₛᵘ[end] = bf.Pᵘ[end]
    Pₖₛˣᵘ[end] = bf.Pˣᵘ[end]

    x̂ₐ = [bf.x; bf.u]
    xₐ .= x̂ₐ[:, end]
    Pₐ .= [bf.Pˣ[end] bf.Pˣᵘ[end]; bf.Pˣᵘ[end]' bf.Pᵘ[end]]

    p = Progress(nf - 1, desc = "Bayesian smoother", showspeed = true, color = :black)
    @inbounds for k ∈ (nf - 1):-1:1
        next!(p)
        Bₖ = ss.B[k]
        Aₐ .= [Iₓ Bₖ]

        # 1 - Prédiction de l'état
        x̃ₐ .= Aₐ*x̂ₐ[:, k]
        P̂ₐ .= [bf.Pˣ[k] bf.Pˣᵘ[k]; bf.Pˣᵘ[k]' bf.Pᵘ[k]]
        P̃ₓ .= Aₐ*P̂ₐ*Aₐ' + Q

        # 2 - Lissage de l'état
        Gₖ = P̂ₐ*(Aₐ'/P̃ₓ)
        xₐ .= x̂ₐ[:, k] .+ Gₖ*(xₐ[1:nx] - x̃ₐ)
        Pₐ .= P̂ₐ .- Gₖ*(Pₐ[1:nx, 1:nx] - P̃ₓ)*Gₖ'

        x̂ₖₛ[:, k] .= xₐ[1:nx]
        Pₖₛˣ[k] .= Pₐ[1:nx, 1:nx]

        ûₖₛ[:, k] .= xₐ[(nx + 1):end]
        Pₖₛᵘ[k] .= Pₐ[(nx + 1):end, (nx + 1):end]
        Pₖₛˣᵘ[k] .= Pₐ[1:nx, (nx + 1):end]
    end

    return BayesianSmootherSolution(x̂ₖₛ, ûₖₛ, Pₖₛˣ, Pₖₛᵘ, Pₖₛˣᵘ)
end
"""
    StateSpace(H, freq, C)

Generates the state space model from the frequency response function `H` and the frequency range `freq`.

# Inputs
- `H`: frequency response function - Matrix{Complex{Float64}}
- `freq`: frequency range [Hz] - Vector{Float64}
- `C`: output matrix - Vector{Complex{Float64}}

# Output
- `B`: input matrix - Vector{Complex{Float64}}
- `D`: feedtrough matrix - Vector{Complex{Float64}}

# Example
```julia-repl
julia> B, D = ss_model(H, freq)
```
"""
@with_kw struct StateSpace
    B :: Vector{Matrix{ComplexF64}}
    D :: Vector{Matrix{ComplexF64}}
    Q :: UniformScaling{Float64}
    R :: Diagonal{Float64, Vector{Float64}}
    C :: Matrix{Float64}

    function StateSpace(H, Q, R, C = I(size(H[1], 1)))
        # Initialization
        nx, ny = size(H[1])
        nf = length(H)
        nc = size(C, 1)

        B = [eltype(H)(undef, nx, ny) for _ in 1:(nf - 2)]
        D = [eltype(H)(undef, nc, ny) for _ in 1:(nf - 2)]

        CH = Matrix{ComplexF64}(undef, nc, ny)

        # Matrice du système
        ee = 1
        @inbounds for k ∈ 2:(nf - 1)
            @. B[ee] = H[k + 1] - 2H[k] + H[k - 1]
            D[ee] .= mul!(CH, C, H[k - 1])
            ee += 1
        end

        new(B, D, Q, R, C)
    end
end

"""
    init_input(H₀, y₀, R₀)

Initializes the input vector `u` and the input covariance matrix `Pᵤ`.

# Inputs
- `H₀`: initial frequency response function - Vector{Complex{Float64}}
- `y₀`: initial measurement - Vector{Float64}
- `R₀`: initial measurement covariance matrix - Matrix{Float64}

# Output
- `u`: initial input vector - Vector{Complex{Float64}}
- `Pᵤ`: initial input covariance matrix - Matrix{Complex{Float64}}

# Example
```julia-repl
julia> u, Pᵤ = init_input(H₀, y₀, R₀)
```
"""
function init_input(H₀, y₀, R₀)
    # Initialization
    S = Diagonal(sqrt.(1 ./diag(R₀)))
    C = S*H₀
    x = S*y₀

    u, Λ = RVR_(C, x)
    Pᵤ = (C'*C + Λ)\I

    return u, Pᵤ
end

"""
    init_reduced_state(u, Pᵤ, H₀, Bₚ)

Initializes the reduced state vector `x̄`, the reduced state covariance matrix `P̄ₓ`, and the reduced state-input covariance matrix `P̄ₓᵤ`.

# Inputs
- `u`: initial input vector - Vector{Complex{Float64}}
- `Pᵤ`: initial input covariance matrix - Matrix{Complex{Float64}}
- `H₀`: initial frequency response function - Vector{Complex{Float64}}
- `Bₚ`: input matrix - Vector{Complex{Float64}}

# Output
- `x̄`: initial reduced state vector - Vector{Complex{Float64}}
- `P̄ₓ`: initial reduced state covariance matrix - Matrix{Complex{Float64}}
- `P̄ₓᵤ`: initial reduced state-input covariance matrix - Matrix{Complex{Float64}}

# Example
```julia-repl
julia> x̄, P̄ₓ, P̄ₓᵤ = init_reduced_state(u, Pᵤ, H₀, Bₚ)
```
"""
function init_reduced_state(u, Pᵤ, H₀, Bₚ)
    HB₀ = H₀ - Bₚ
    x̄ = HB₀*u
    P̄ₓ = HB₀*Pᵤ*HB₀'
    P̄ₓᵤ = HB₀*Pᵤ

    return x̄, P̄ₓ, P̄ₓᵤ
end

"""
    compute_state(bf, H, nf)

Computes the state vector `x`, the state covariance matrix `Pₓ` from the reduced state vector `x̄`, the reduced state covariance matrix `P̄ₓ`, and the reduced state-input covariance matrix `P̄ₓᵤ`.

# Inputs
- `bf`: Bayesian filter - BayesianFilter
- `H`: frequency response function - Vector{Matrix{Complex{Float64}}}
- `nf`: number of frequencies - Int

# Output
- `x`: state vector - Vector{Complex{Float64}}
- `Pₓ`: state covariance matrix - Matrix{Complex{Float64}}

# Example
```julia-repl
julia> x, Pₖˣ = compute_state(x̄, P̄ₓ, P̄ₓᵤ, H₀, Bₚ)
```
"""
function compute_state(bf, H)
    # Initialization
    nx = size(bf.x, 1)
    nf = length(H)

    # Pre-allocations
    xₖ = typeof(bf.x)(undef, nx, nf - 2)
    Pₖˣ = [typeof(bf.Pˣ[1])(undef, nx, nx) for _ in 1:(nf - 2)]

    # Compute state
    ee = 1
    for k ∈ 2:(nf - 1)
        Bₚ = H[k - 1]
        BP = Bₚ*bf.Pˣᵘ[ee]'

        xₖ[:, ee] .= bf.x[:, ee] + Bₚ*bf.u[:, ee]
        Pₖˣ[ee] .= bf.Pˣ[ee] + BP + BP' + Bₚ*bf.Pᵘ[ee]*Bₚ'
        ee += 1
    end

    return xₖ, Pₖˣ
end
"""
    RecursiveRegProblem

A struct to hold the data for a recursive regression problem.

# Inputs
- `A`: System matrix - Matrix{ComplexF64}
- `b`: measurement data - Vector{ComplexF64}
- `x₀`: initial solution - Vector{ComplexF64}
- `R`: covariance matrix of the measurement noise - Matrix{ComplexF64}

# Example
```julia-repl
julia> prob = RecursiveRegProblem(y, A, x₀, R)
```
"""
@with_kw struct RecursiveRegProblem
    A :: Vector{Matrix{ComplexF64}}
    b :: Matrix{ComplexF64}
    x₀ :: Vector{ComplexF64}
    R

    function RecursiveRegProblem(A, b, R)
        # Calculation of the initial input solution
        x₀ = init_input(A[1], b[:, 1], R)[1]

        return new(A, b, x₀, R)
    end
end

function solve(prob :: RecursiveRegProblem)
    ny, nf = size(prob.b)
    nx = size(prob.A[1], 2)

    # Initialisation
    xₖ = Matrix{ComplexF64}(undef, nx, nf)
    xₖ[:, 1] = prob.x₀
    R = prob.R

    # Pre-allocations
    εₖ = Vector{ComplexF64}(undef, nx)
    iₖ = Vector{ComplexF64}(undef, ny)
    # ε⁻ = xₖ[:, 1]

    # Pre-computations
    S = Diagonal(sqrt.(1 ./diag(R)))

    p = Progress(nf - 1, desc = "Recursive RVR...", showspeed = true, color = :black)
    @inbounds for k ∈ 2:nf
        next!(p)
        Aₖ = prob.A[k]

        x̃ₖ = xₖ[:, k-1]
        iₖ .= prob.b[:, k] .- Aₖ*x̃ₖ

        # εₖ .= estimation(S*Aₖ, S*iₖ, ε⁻)
        εₖ .= estimation(S*Aₖ, S*iₖ, xₖ[:, k - 1])
        xₖ[:, k] .= x̃ₖ .+ εₖ
        # ε⁻ .= εₖ
    end

    return xₖ
end

function estimation(A, b, x₀)
    # Initialization
    α = 1.
    β = 1e-18
    tol = 1e-8
    maxit = 200

    # Preallocation
    nᵤ = size(A, 2)
    x = Vector{eltype(b)}(undef, nᵤ)
    Λ = Diagonal{Float64}(undef, nᵤ)
    indm = diagind(Λ)
    τ = Vector{Float64}(undef, nᵤ)

    # Precomputation
    AtA = A'*A
    Atb = A'*b

    # Compute solution
    crit = 1.
    ee = 0
    while (crit > tol) && (ee ≤ maxit)
        @. τ = α/(β + abs2(x₀))
        Λ[indm] .= τ

        x.= (AtA + Λ)\Atb

        crit = norm(x - x₀, 1)/norm(x₀, 1)
        ee += 1
        x₀ .= x
    end

    return x
end
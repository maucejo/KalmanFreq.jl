"""
    agwn(x, snr_dB, rst = true)

Adds a Gaussian White Noise (AWGN) to a signal `x` with a given SNR.

# Inputs
- `x`: signal - Matrix{ComplexF64}
- `snr_dB`: signal to noise ratio [dB] - Float64
- `rst`: reset the random number generator - Bool

# Output
- `y`: noisy signal - Matrix{ComplexF64}

# Example
```julia-repl
julia> y = agwn(x, 25.)
```
"""
function agwn(x, snr_dB, rst = true)
    # Reset the RNG if required
    if rst
        rng = MersenneTwister(1000)
    end

    N, L = size(x)                          # Dimensions des données
    SNR = 10^(snr_dB/10.)                   # SNR en échelle linéaire
    En = sum(abs2, x, dims = 2)/L           # Énergie du signal
    V = En/SNR                              # Variance du bruit

    σ = sqrt.(V)                           # Écart-type
    n = σ.*randn(rng, ComplexF64, N, L)    # Bruit

    return x .+ n
end

"""
    estimated_SNR(x, var)

Estimates the SNR of a signal `x` with a given variance `var`.

# Inputs
- `x`: signal - Matrix{ComplexF64}
- `var`: variance - Float64

# Output
- `SNR`: signal to noise ratio [dB] - Float64

# Example
```julia-repl
julia> SNR = estimated_SNR(x, 1e-3)
```
"""
function estimated_SNR(x, var)

    L = size(x, 2)
    En = sum(abs2, x, dims = 2)/L

    SNR = En./var

    return 10log10.(SNR)
end

"""
    varest(x)

Estimates the noise variance of a signal `x` using Bayesian denoising.

# Inputs
- `x`: signal - Matrix{ComplexF64}

# Output
- `noisevar`: noise variance - Vector{Float64}

# Example
```julia-repl
julia> noisevar = varest(x)
```
"""
function varest(x)
    # Initialisation
    ndim = ndims(x)

    if ndim == 1
        noisevar = vareste_(x)
    elseif ndim == 2
        nx = size(x, 1)
        noisevar = Vector{Float64}(undef, nx)
        @inbounds @views for idx ∈ 1:nx
             noisevar[idx] = varest_(x[idx, :])
        end
    end

    return noisevar
end

"""
    varest_(x)

Estimates the noise variance of a signal `x` using Bayesian regularization.

Note: This function is not intended to be used directly

# Input
- `x`: signal - Vector{Float64}

# Output

- `noisevar`: noise variance - Float64
"""
function varest_(x)
    # Valeurs propres de la matrice de lissage d'ordre 1
    n = length(x)
    s = Vector{Float64}(undef, n)
    z = Vector{eltype(x)}(undef, n)

    @. s = 2(cos((0:n-1)π/n) - 1.)
    @. s[s == 0.] = 1e-8
    s² = s.^2

    # Calul de la DCT-2
    z .= dct(x)

    lb = -2
    ub = 20
    res = optimize(L -> func(L, z, s²), lb, ub)
    λ = 10^Optim.minimizer(res)

    fₖ = @. (1. + λ*s²)/s²
    gₖ = mean(@. abs2(z)/fₖ)

    return λ*gₖ
end

"""
    func(L, z, s²)

Function to be optimized in `noisevar1D_`.

Note: This function is not intended to be used directly

# Inputs

- `L`: parameter to be optimized - Float64
- `z`: signal - Vector{Float64}
- `s²`: eigenvalues of the smoothing matrix - Vector{Float64}

# Output
- `f`: function to be optimized - Float64
"""
function func(L, z, s²)
    n = length(z)
    fₖ = @. (1. + 10^L*s²)/s²
    gₖ = mean(@. abs2(z)/fₖ)

    return sum(log, fₖ) + (n - 2)*log(gₖ)
end


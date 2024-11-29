"""
    Beam(L, b, h, E, ρ)

Structure defining a beam with length `L`, width `b`, height `h`, Young's modulus `E` and mass density `ρ`.

# Inputs
- `L`: length [m] - Float64
- `b`: width [m] - Float64
- `h`: height [m] - Float64
- `E`: Young's modulus [Pa] - Float64
- `ρ`: mass density [kg/m³] - Float64

# Output
Beam structure with the following fields:
- `L`: length [m] - Float64
- `m`: linear mass [kg/m] - Float64
- `D`: Bending stiffness [N.m²] - Float64

# Example
```julia-repl
julia> b = Beam(1., 3e-2, 1e-2, 2.1e11, 7850.)
Beam(1.0, 2.355, 525)
```
"""
@with_kw struct Beam
    L :: Float64
    m :: Float64
    D :: Float64

    function Beam(L::Float64, b::Float64, h::Float64, E::Float64, ρ::Float64)
        S = b*h
        I = b*h^3/12.
        m = ρ*S
        D = E*I

        return new(L, m, D)
    end
end

"""
    eigval(b, fₘₐₓ)

Compute the eigenvalues of a simply supported beam `b` up to a maximum frequency `fₘₐₓ`.

# Inputs
- `b`: beam structure
- `fₘₐₓ`: maximum frequency [Hz] - Float64

# Outputs
- `ωₙ`: natural angular frequency [rad/s] - Vector{Float64}
- `kₙ`: natural wavenumber [rad/m] - Vector{Float64}

# Example
```julia-repl
julia> b = Beam(1., 3e-2, 1e-2, 2.1e11, 7850.)
julia> ωₙ, kₙ = eigval(b, 2000.)
```
"""
function eigval(b::Beam, fₘₐₓ)
    (; L, m, D) = b

    c = sqrt(D/m)
    ωₘₐₓ = 2π*fₘₐₓ

    ωₙ = Float64[]
    kₙ = Float64[]
    n = 0.
    ωᵢ = 0.
    while ωᵢ ≤ ωₘₐₓ
        n += 1.
        kᵢ = n*π/L
        ωᵢ = c*kᵢ^2
        push!(kₙ, kᵢ)
        push!(ωₙ, ωᵢ)
    end

    return ωₙ, kₙ
end

"""
    eigmode(b, kₙ, x)

Compute the eigenmodes (mode shapes) of a simply supportted beam `b` for given natural wavenumbers `kₙ` and given spatial coordinates `x`.

# Inputs
- `b`: beam structure
- `kₙ`: natural wavenumber [rad/m] - Vector{Float64}
- `x`: spatial coordinates [m] - Vector{Float64} or Range

# Output
- `ϕₙ`: eigenmodes - Matrix{Float64} (dimension: length(x) x length(kₙ))

# Example
```julia-repl
julia> b = Beam(1., 3e-2, 1e-2, 2.1e11, 7850.)
julia> ωₙ, kₙ = eigval(b, 2000.)
julia> x = LinRange(0., 1., 100)
julia> ϕₙ = eigmode(b, kₙ, x)
```
"""
function eigmode(b::Beam, kₙ, x)
    (; L, m) = b

    if !isa(x, Array)
        x = collect(x);
    end

    Mₙ = m*L/2.
    ϕₙ = sin.(x*kₙ')

    return @. ϕₙ/sqrt(Mₙ')
end

"""
    frf(ωₙ, ξₙ, ϕₘ, ϕₑ, freq)

Computes the frequency response function (FRF) of a simply supported beam `b` for given natural frequencies `ωₙ`, damping ratios `ξₙ`, eigenmodes `ϕₘ` and `ϕₑ`, and frequency range `freq`.

# Inputs
- `ωₙ`: natural angular frequency [rad/s] - Vector{Float64}
- `ξₙ`: damping ratio - Float64
- `ϕₘ`: eigenmodes at measurement points - Matrix{Float64}
- `ϕₑ`: eigenmodes at excitation point - Matrix{Float64}
- `freq`: frequency range [Hz] - Vector{Float64}

# Output
- `H`: frequency response function - Array{Matrix{Complex{Float64}}} (dimension: length(freq) x length(ϕₘ) x length(ϕₑ))

Example
```julia-repl
julia> H = frf(ωₙ, ξₙ, ϕₘ, ϕₑ, freq)
```
"""
function frf(ωₙ, ξₙ, ϕₘ, ϕₑ, freq)
    ωf = 2π*freq

    # Initialization
    Nₘ, Nmodes = size(ϕₘ)
    Nₑ = size(ϕₑ, 1)
    Nf = length(freq)
    M = Diagonal{ComplexF64}(undef, Nmodes)
    indm = diagind(M)

    c = Matrix{ComplexF64}(undef, Nₘ, Nₑ)
    d = Matrix{ComplexF64}(undef, Nₘ, Nmodes)
    FRF = [Matrix{ComplexF64}(undef, Nₘ, Nₑ) for _ in 1:Nf]
    @inbounds for (f, ω) in enumerate(ωf)
        @. M[indm] = -ω^2/(ωₙ^2 - ω^2 + 2im*ξₙ*ωₙ*ω)
        FRF[f] .= mul!(c, mul!(d, ϕₘ, M), ϕₑ')
    end

    return FRF
end

"""
    excitation(exc_type, nf)

Generates the excitation vector `F` for a given excitation type `exc_type` and number of frequencies `nf`.

# Inputs
- `exc_type`: excitation type - Symbol
- `nf`: number of frequencies - Int

# Output
- `F`: excitation - Vector{Float64}

# Example
```julia-repl
julia> F = excitation(:constant, 100)

julia> F = excitation(:random, 100)
```
"""
function excitation(exc_type, nf)
    if exc_type == :constant
        F = ones(nf)
    else
        F = 2 .+ 0.1randn(MersenneTwister(1000), nf)
    end

    return F
end

"""
    select_config(config, Npoint, Δx)

Select the measurement configuration.

# Inputs
- `config`: configuration number
- `L`: length of the beam
- `Npoint`: number of points
- `Δx`: shift from the left end of the beam

# Outputs
- `X`: positions of the measurement points
- `Nx`: location of the observation/excitation point

# Example
```julia-repl
julia> X, Nx = select_config(1, 1., 20)
```
"""
function select_config(config, L, Npoint = 20, Δx = 5e-2)
    Xmes = LinRange(Δx, L - Δx, Npoint)

    if config == 1			     # Maillage de mesure dense
        X = Xmes
        Nx = 13
    elseif config == 2		     # Maillage normal
        X = Xmes[1:2:end]
        Nx = 7
    else     				     # Maillage grossier
        X = Xmes[1:4:end]
        Nx = 4
    end

    return X, Nx
end
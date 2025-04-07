"""
    BeamMesh(xmin, L, Nelt)

Construct a mesh for a beam with `Nelt` elements, length `L` and starting at `xmin`.

# Inputs
- `xmin::Float64`: starting position of the beam
- `L::Float64`: length of the beam
- `Nelt::Int64`: number of elements

# Outputs
- `mesh::BeamMesh`: mesh of the beam

# Example
```julia-repl
julia> mesh = BeamMesh(0., 1., 10)
```
"""
@with_kw struct BeamMesh
    xmin::Float64
    L::Float64
    Nelt::Int64
    Nnodes::Int64
    Nodes::Matrix{Float64}
    Elt::Matrix{Int64}
    Ndof_per_node::Int64
    elem_size::Float64
    free_dofs :: Vector{Int64}

    function BeamMesh(xmin, L, Nelt, bc_type = :simply_supported)
        Nnodes = Nelt + 1
        Nodes = zeros(Nnodes,2)
        Elt = zeros(Nelt,3)
        Ndof_per_node = 2
        elem_size = L/Nelt

        for i = 1:Nnodes
            Nodes[i,1] = i
            Nodes[i,2] = xmin + (i-1)*elem_size
        end

        for i = 1:Nelt
            Elt[i,1] = i
            Elt[i,2] = i
            Elt[i,3] = i+1
        end

        if bc_type == :simply_supported
            free_dofs = [collect(2:2*Nnodes-2); 2Nnodes]
        elseif bc_type == :clamped
            free_dofs = collect(3:2Nnodes-2)
        else # free
            free_dofs = collect(1:2Nnodes)
        end

        new(xmin, L, Nelt, Nnodes, Nodes, Elt, Ndof_per_node, elem_size, free_dofs)
    end
end

"""
    assembly(b, bm)

Compute the global stiffness and mass matrices for a beam `b` with a mesh `bm`.

# Inputs
- `b`: beam structure
- `bm`: beam mesh

# Outputs
- `K`: global stiffness matrix
- `M`: global mass matrix

# Example
```julia-repl
julia> b = Beam(1., 3e-2, 1e-2, 2.1e11, 7850.)
julia> mesh = BeamMesh(0., 1., 10)
julia> K, M = beam_assembly(b, mesh)
```
"""
function assembly(b :: Beam, bm :: BeamMesh)
    # Compute elemental matrices
    kₑ, mₑ = beam_elem_elt(b, bm.elem_size)

    # Assemble global matrices
    Nnodes = bm.Nnodes
    Nelt = bm.Nelt
    Ndof_per_node = bm.Ndof_per_node
    Nddl = Nnodes*Ndof_per_node

    K = zeros(Nddl, Nddl)
    M = zeros(Nddl, Nddl)
    ind = zeros(eltype(Nnodes), Ndof_per_node^2)
    @inbounds @views for i = 1:Nelt
        ind .= (Ndof_per_node*bm.Elt[i,2:end]' .+ repeat((0:Ndof_per_node-1), 1, Ndof_per_node) .- Ndof_per_node .+ 1)[:];

        K[ind, ind] += kₑ
        M[ind, ind] += mₑ
    end

    return K, M
end

"""
    beam_elem_elt(b, h)

Compute the elemental stiffness and mass matrices for a beam with a element size `h`.

# Inputs
- `b`: beam structure
- `h`: element size

# Outputs
- `kₑ`: elemental stiffness matrix
- `mₑ`: elemental mass matrix

# Example
```julia-repl
julia> b = Beam(1., 3e-2, 1e-2, 2.1e11, 7850.)
julia> kₑ, mₑ = beam_elem_elt(b, 1e-2)
```
"""
function beam_elem_elt(b :: Beam, h)
    # Constants
    kc = b.D/h^3
    mc = b.m*h/420.

    # Elemental stiffness matrix
    kₑ = kc.*[12. 6h -12. 6h;
              6h 4h^2 -6h 2h^2;
              -12. -6h 12. -6h;
              6h 2h^2 -6h 4h^2]

    # Elemental mass matrix
    mₑ = mc.*[156. 22h 54. -13h;
              22h 4h^2 13h -3h^2;
              54. 13h 156. -22h;
              -13h -3h^2 -22h 4h^2]

    return kₑ, mₑ
end

"""
    dofs_selection(bm, X)

Select the dofs corresponding to the closest nodes to the positions `X`.

# Inputs
- `bm`: beam mesh
- `X`: positions

# Outputs
- `dofs`: dofs corresponding to the closest nodes
- `S`: selection matrix

# Example
```julia-repl
julia> mesh = BeamMesh(0., 1., 10)
julia> dofs, S = dofs_selection(mesh, [0.1, 0.2])
```
"""
function dofs_selection(bm :: BeamMesh, X)
    N = length(X)
    dofs = zeros(Int, N)
    S = zeros(N, length(bm.free_dofs))
    @inbounds for i = 1:N
        d = @. abs(bm.Nodes[:, 2] - X[i])
        dofs[i] = 2argmin(d) - 1

        pos = findall(bm.free_dofs .== dofs[i])
        if length(pos) != 0
            S[i, pos[1]] = 1
        end
    end

    return dofs, S
end

"""
    resp(K, M, C, F, freq)

Compute the response of a system with stiffness `K`, mass `M`, damping `C` and force `F` at the frequencies `freq`.

# Inputs
- `K`: stiffness matrix
- `M`: mass matrix
- `C`: damping matrix
- `F`: force vector
- `freq`: frequencies

# Output
- `X`: response

# Example
```julia-repl
julia> K = [1. 0.; 0. 1.]
julia> M = [1. 0.; 0. 1.]
julia> C = [0. 0.; 0. 0.]
julia> F = [1., 1.]
julia> X = resp(K, M, C, F, [1., 2.])
```
"""
function resp(K, M, C, F, freq)
    ωf = 2π*freq
    nf = length(freq)
    ndofs = size(K, 1)
    X = Matrix{ComplexF64}(undef, ndofs, nf)

    p = Progress(nf - 1, desc = "Response calculation...", showspeed = true)
    @inbounds @views for (i, ω) in enumerate(ωf)
        next!(p)
        X[:, i] .= -ω^2*((K + 1im*ω*C - ω^2*M)\F[:, i])
    end

    return X
end

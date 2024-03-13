"""
    RE(xref, xid)

Measure of the relative difference between a reference signal `xref` and the identified one `xid`.

# Inputs
- `xref`: Reference signal - Vector
- `xid`: Identified signal - Vector

# Output
- RE: Relative error - Float64

# Example
```julia
julia> re = RE(xref, xid)
```
"""
function RE(xref, xid)
    return 100norm(xref - xid, 1)/norm(xref, 1)
end


"""
    CorrCoeff(xref, xid)

Measure of the similarity between a reference signal `xref` and the identified one `xid`.

# Inputs
- `xref`: Reference signal - Vector
- `xid`: Identified signal - Vector

# Output
- CC: Relative error - Float64

# Example
```julia
julia> cc = CorrCoeff(xref, xid)
"""
function CorrCoeff(xref, xid)
    return 100abs(dot(xref, xid))/(norm(xref)*norm(xid))
end

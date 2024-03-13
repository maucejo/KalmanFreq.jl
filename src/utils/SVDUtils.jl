function unpack_gsvd(F)
    # Reordering of the singular values
    X = F.Q*F.R'
    sm = [diag(F.D1) diag(F.D2)]

    return F.U, sm, X, F.V
end
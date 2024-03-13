@setup_workload begin
    # Frequency range
    fmin = 100.
    fmax = 104.
    Δf = 1.
    freq = range(fmin, fmax, step = Δf)
    freqH = range(fmin - Δf, fmax + Δf, step = Δf)

    # Beam parameters
    L = 1.
    b = 3e-2
    h = 1e-2
    E = 2.1e11
    ρ = 7850.
    ξₙ = 1e-2

    @compile_workload begin
        # Analytical modes
        beam = Beam(L, b, h, E, ρ)
        ωₙ, kₙ = eigval(beam, 2freqH[end])

        # FE model
        Δx = 1e-3
        Nₚ = 4
        Nₑ = 2
        Xₘ = LinRange(Δx, L - Δx, Nₚ)
        Xₑ = [Xₘ[Nₑ]]

        Nelem = 4
        bm = BeamMesh(0., L, Nelem)

        K, M = assembly(beam, bm)
        ndofs = size(K, 1)

        Ks = K[bm.free_dofs, bm.free_dofs]
        Ms = M[bm.free_dofs, bm.free_dofs]
        λ, ϕ = eigen(Ks, Ms)
        ωfem = sqrt.(λ)

        Cₙ = Diagonal(2ξₙ*ωfem)
        Cs = Ms*ϕ*Cₙ*ϕ'*Ms

        id_dofs, S = dofs_selection(bm, Xₘ)
        exc_dof = id_dofs[Nₑ]
        pos_m = findall(ωfem/2π .≤ 2freqH[end])
        ωₙ = ωfem[pos_m]
        ϕₘ = S*ϕ[:, pos_m]          # Déformée propre aux points de mesure
        ϕᵢ = S*ϕ[:, pos_m]

        F = excitation(:constant, length(freq))
        F1 = excitation(:random, length(freq))
        Fref = zeros(ndofs, length(freq))
        Fref[exc_dof, :] .= F
        Fs = Fref[bm.free_dofs, :]

        # Noisy response
        snr = 25.
        yref = S*resp(Ks, Ms, Cs, Fs, freq[1:2])
        y = agwn(yref, snr)
        vary = varest(y)

        # FRF
        H = frf(ωₙ, ξₙ, ϕₘ, ϕₘ, freqH[1:4])
        Hd = spblkdiag(H[1:2])

        # Bayesian Filter
        vary[vary .< 1e-10] .= 1e-10
        R = Diagonal(vary) # Measurement noise covariance matrix
        Q = 1e-10I         # Process noise covariance matrix
        prob_bf = BayesianFilterProblem(H, y[:, 1:2], Q, R)
        sol_bf = solve(prob_bf)

        ## RVR
        prob_rvr = RVRProblem(H[2:3], y[:, 1:2], R)
        u_rvr = solve(prob_rvr)

        ## lq_reg
        prob_lq = LqRegProblem(H[2:3], y[:, 1:2], 0.5, R)
        u_lq = solve(prob_lq)

        ## lpq_reg
        prob_lpq = LpqRegProblem(H[2:3], y[:, 1:2], 2., 0.5, R)
        u_lpq = solve(prob_lpq)
    end
end
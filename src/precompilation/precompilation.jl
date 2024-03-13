@setup_workload begin
    ## Définition de la bande de fréquence d'intérêt
    fmin = 100. # Fréquence minimale [Hz]
    fmax = 135. # Fréquence maximale [Hz]
    Δf = 1.  # Résolution fréquentielle [Hz]
    freq = range(fmin, fmax, step = Δf) # Bande de fréquence d'intérêt
    freqH = range(fmin - Δf, fmax + Δf, step = Δf) # Bande de fréquence pour H

    ## Définition de la poutre
    L = 1. # Longueur de la poutre [m]
    b = 3e-2 # Largeur de la poutre [m]
    h = 1e-2 # Épaisseur de la poutre [m]
    E = 2.1e11 # Module de Young [Pa]
    ρ = 7850. # Masse volumique [kg/m³]
    ξₙ = 1e-2 # Facteur d'amortissement

    @compile_workload begin
        beam = Beam(L, b, h, E, ρ)  # Définition de la structure

        ## Calcul des pulsations propres
        ωₙ, kₙ = eigval(beam, 2freqH[end]) # Calcul des pulsations propres

        ## Définition du maillage de mesure
        Δx = 1e-3                       # Pas de discrétisation
        Nₚ = 4                          # Nombre de points de mesure
        Nₑ = 2                         # ID du point d'excitation
        Xₘ = LinRange(Δx, L - Δx, Nₚ)    # Maillage de mesure
        Xₑ = [Xₘ[Nₑ]]                   # Position d'excitation

        ## Définition du maillage de FEM
        Nelem = 4;                 # Nombre d'éléments
        bm = BeamMesh(0., L, Nelem)  # Maillage FEM

        ## Construction des matrices K et M
        K, M = assembly(beam, bm)
        ndofs = size(K, 1)

        ## Construction de la matrice d'amortissement
        Ks = K[bm.free_dofs, bm.free_dofs]
        Ms = M[bm.free_dofs, bm.free_dofs]
        λ, ϕ = eigen(Ks, Ms)
        ωfem = sqrt.(λ)

        ## Construction de la matrice d'amortissement
        Cₙ = Diagonal(2ξₙ*ωfem)   # Matrice d'amortissement modal
        Cs = Ms*ϕ*Cₙ*ϕ'*Ms

        ## Défintion des ddls de mesure et de reconstruction
        id_dofs, S = dofs_selection(bm, Xₘ)
        exc_dof = id_dofs[Nₑ]

        ## Calcul des déformées propres
        ϕₘ = eigmode(beam, kₙ, Xₘ)      # Déformée propre aux points de mesure
        ϕₑ = eigmode(beam, kₙ, Xₑ)     # Déformée propre au point d'excitation

        ## Définition de l'excitation
        F = excitation(:constant, length(freq)) # Vecteur d'excitation
        F1 = excitation(:random, length(freq))
        Fref = zeros(ndofs, length(freq))  # Vecteur d'excitation de référence
        Fref[exc_dof, :] .= F
        Fs = Fref[bm.free_dofs, :]

        ## Génération des données d'accélération bruitée
        snr = 25. # Rapport signal sur bruit
        yᵣ = resp(ωₙ, ξₙ, ϕₘ, ϕₑ, freq, F) # Signal de référence
        yref = resp(Ks, Ms, Cs, Fs, freq[1:2])
        y = agwn(yᵣ, snr) # Signal bruité
        vary = varest(y, :derrico) # Variance du bruit
        v1 = varest(y) # Variance du bruit

        ## Calcul de la fonction de transfert
        H = frf(ωₙ, ξₙ, ϕₘ, ϕₘ, freqH[1:4])
        H1 = frf_modal(ωₙ, ξₙ, freqH[1:4], ϕₘ)

        Hd = spblkdiag2(H[1:2])

        ## Construction du modèle d'état
        R = Diagonal(vary)
        Q = 1e-10I

        prob_bf = BayesianFilterProblem(H, y[:, 1:2], Q, R)
        sol_bf = solve(prob_bf)

        prob_bf2 = BayesianFilterProblem(H1, y[:, 1:2], Q, R, ϕₘ)

        sol_bs = bsmoother(prob_bf.ss, sol_bf)

        ## Calcul par RVR
        prob_rvr = RVRProblem(H[2:3], y[:, 1:2], R)
        u_rvr = solve(prob_rvr)

        ## Calcul lq_reg
        prob_lq = LqRegProblem(H[2:3], y[:, 1:2], 0.5, R)
        u_lq = solve(prob_lq)

        prob_lq2 = LqRegProblem(H[2:3], y[:, 1:2], 0.5, R, type = :add, method = :lc)
        u_lq2 = solve(prob_lq)

        ## Calcul lpq_reg
        prob_lpq = LpqRegProblem(H[2:3], y[:, 1:2], 2., 0.5, R)
        u_lpq = solve(prob_lpq)

        ## Kalman Filter
        prob_kf = KalmanFilterProblem(H[2:3], y[:, 1:2], Q(Nₚ), R)
        sol_kf = solve(prob_kf)

        sol_ks = ksmoother(sol_kf)

        ## Recursive Regularization
        prob_rr = RecursiveRegProblem(H[2:3], y[:, 1:2], R)
        u_rr = solve(prob_rr)
    end
end
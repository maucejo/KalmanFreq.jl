using DrWatson, Revise, MKL, GLMakie
@quickactivate "KalmanFrequency"

# using KalmanFrequency, LinearAlgebra

using Parameters, Random, LinearAlgebra, Optim, FFTW, Statistics, SpecialFunctions, DSP, DataInterpolations, StatsBase, ProgressMeter, BlockDiagonals, SparseArrays, LazyGrids
includet(srcdir("gendata", "VibData.jl"))
includet(srcdir("gendata", "FEM.jl"))
includet(srcdir("filter", "StateSpace.jl"))
includet(srcdir("filter", "Regularization.jl"))
includet(srcdir("filter", "BayesianFilter.jl"))
includet(srcdir("filter", "BayesianSmoother.jl"))
includet(srcdir("utils", "LinalgUtils.jl"))
includet(srcdir("utils", "NoiseUtils.jl"))
includet(srcdir("utils", "PlotUtils.jl"))

## Définition de la bande de fréquence d'intérêt
fmin = 100. # Fréquence minimale [Hz]
fmax = 1000. # Fréquence maximale [Hz]
Δf = 0.5  # Résolution fréquentielle [Hz]
freq = range(fmin, fmax, step = Δf) # Bande de fréquence d'intérêt
freqH = range(fmin - Δf, fmax + Δf, step = Δf) # Bande de fréquence pour H

## Définition de la poutre
L = 1. # Longueur de la poutre [m]
b = 3e-2 # Largeur de la poutre [m]
h = 1e-2 # Épaisseur de la poutre [m]
E = 2.1e11 # Module de Young [Pa]
ρ = 7850. # Masse volumique [kg/m³]
ξₙ = 1e-2 # Facteur d'amortissement
beam = Beam(L, b, h, E, ρ)  # Définition de la structure

## Définition du maillage de FEM
Nelem = 100;                 # Nombre d'éléments
bm = BeamMesh(0., L, Nelem)  # Maillage FEM

## Construction des matrices K et M
K, M = assembly(beam, bm)
ndofs = size(K, 1)

## Calcul des modes propres
Ks = K[bm.free_dofs, bm.free_dofs]
Ms = M[bm.free_dofs, bm.free_dofs]
λ, ϕ = eigen(Ks, Ms)
ωfem = sqrt.(λ)

## Construction de la matrice d'amortissement
Cₙ = Diagonal(2ξₙ*ωfem)   # Matrice d'amortissement modal
Cs = Ms*ϕ*Cₙ*ϕ'*Ms        # Matrice d'amortissement structurelle

## Définition du maillage de mesure
Δx = 5e-2                            # Pas de discrétisation
Npoint = 20                          # Nombre de points de mesure
Xmes = LinRange(Δx, L - Δx, Npoint)  # Maillage de mesure

config_mesure, config_id = (2, 1)	 # Configuration étudiée
config_mesure, config_id = (1, 1)	 # Configuration étudiée
Xₘ, Nₒ = select_config(config_mesure, L, Npoint, Δx)
Xᵢ, Nₑ = select_config(config_id, L, Npoint, Δx)
Nₘ = length(Xₘ)                      # Nombre de points de mesure
Nᵢ = length(Xᵢ)                      # Nombre de points d'excitation
Xₑ = [Xᵢ[Nₑ]]                        # Position d'excitation

type_mes = :coloc
# type_mes = :ncoloc
if type_mes == :ncoloc
    Xₘ = Xₘ .+ Δx/2
end

## Défintion des ddls de mesure et de reconstruction
meas_dofs, Sₘ = dofs_selection(bm, Xₘ)
id_dofs, Sᵢ = dofs_selection(bm, Xᵢ)
obs_dof = meas_dofs[Nₒ]
exc_dof = id_dofs[Nₑ]

## Calcul du vecteur excitation
exc_type = :constant # Type d'excitation
# exc_type = :random
F = excitation(exc_type, length(freq)) # Vecteur d'excitation
Fref = zeros(ndofs, length(freq))  # Vecteur d'excitation de référence
Fref[exc_dof, :] .= F
Fs = Fref[bm.free_dofs, :]     # Vecteur d'excitation
Fᵣ = Sᵢ*Fs                     # Vecteur d'excitation de référence

## Définition des déformées modales aux ddls mesurés et de reconstruction
pos_m = findall(ωfem/2π .≤ 2freqH[end])
ωₙ = ωfem[pos_m]
ϕₘ = Sₘ*ϕ[:, pos_m]          # Déformée propre aux points de mesure
ϕᵢ = Sᵢ*ϕ[:, pos_m]          # Déformée propre aux points de reconstruction

## Génération des données d'accélération bruitée
snr = 25. # Rapport signal sur bruit
yᵣ = Sₘ*resp(Ks, Ms, Cs, Fs, freq) # Signal de référence
ỹ = agwn(yᵣ, snr) # Signal bruité
vary = varest(ỹ, :derrico) # Variance du bruit

## Calcul de la fonction de transfert
H = frf(ωₙ, ξₙ, ϕₘ, ϕᵢ, freqH)

## Baysesian Filter
R = Diagonal(vary)
Q = 1e-10I

prob_bf = BayesianFilterProblem(H, ỹ, Q, R)
sol_bf = solve(prob_bf)
u_bf = sol_bf.u

## Bayesian Smoother
sol_bs = bsmoother(prob_bf.ss, sol_bf)
u_bs = sol_bs.u

## Bayesian Filter - reverse mode
Hrev = reverse(H)
ỹrev = reverse(ỹ, dims = 2)
prob_bfr = BayesianFilterProblem(Hrev, ỹrev, Q, R)
sol_bfr = solve(prob_bfr)
u_bfr = reverse(sol_bfr.u, dims  = 2)

## Calcul par RVR
prob_rvr = RVRProblem(H[2:end-1], ỹ, R)
u_rvr = solve(prob_rvr)

## Calcul lq_reg
q = 0.5
prob_lq = LqRegProblem(H[2:end-1], ỹ, q, R)
# prob_lq = LqRegProblem(H[2:end-1], ỹ, q, R, type = :add, method = :lc)
u_lq = solve(prob_lq)

## Calcul lpq_reg
p = 2.
prob_lpq = LpqRegProblem(H[2:end-1], ỹ, p, q, R)
u_lpq = solve(prob_lpq)

## Visualisation Bayesian Filter
fig_bfw = waterfall_plot(freq, Xᵢ,  real(u_bf), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_bf)), 1.12*maximum(real(u_bf))]);
display(GLMakie.Screen(), fig_bfw);

fig_bfs = plot(freq, F, real(u_bf[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_bfs);

## Visualisation Bayesian Smoother
fig_bsw = waterfall_plot(freq, Xᵢ,  real(u_bs), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_bs)), 1.12*maximum(real(u_bs))]);
display(GLMakie.Screen(), fig_bsw);

fig_bss = plot(freq, F, real(u_bs[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_bss);

## Visualisation Bayesian Filter - Reverse mode
fig_bfrw = waterfall_plot(freq, Xᵢ,  real(u_bfr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_bfr)), 1.12*maximum(real(u_bfr))]);
display(GLMakie.Screen(), fig_bfrw);

fig_bfrs = plot(freq, F, real(u_bfr[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_bfrs);

## Visualisation RVR
fig_rvrw = waterfall_plot(freq, Xᵢ,  real(u_rvr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_rvr)), 1.12*maximum(real(u_rvr))]);
display(GLMakie.Screen(), fig_rvrw);

fig_rvrs = plot(freq, F, real(u_rvr[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_rvrs);

## Visualisation Lq-regularization
fig_lqw = waterfall_plot(freq, Xᵢ, real(u_lq), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_lq)), 1.12*maximum(real(u_lq))]);
display(GLMakie.Screen(), fig_lqw);

fig_lqs = plot(freq, F, real(u_lq[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_lqs);

## Visualisation Lpq-regularization
fig_lpqw = waterfall_plot(freq, Xᵢ, real(u_lpq), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_lpq)), 1.12*maximum(real(u_lpq))]);
display(GLMakie.Screen(), fig_lpqw);

fig_lpqs = plot(freq, F, real(u_lpq[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_lpqs);

## Calcul de l'état
xₖ, Pₖˣ = compute_state(sol_bf, H)

## Visualisation de l'état
y1 = @. 20log10(abs(xₖ[Nₒ, :]))
y2 = @. 20log10(abs(ỹ[Nₒ, :]))
y3 = @. 20log10(abs(yᵣ[Nₒ, :]))
fig_state = plot(freq, y1, y2, y3; xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_state);
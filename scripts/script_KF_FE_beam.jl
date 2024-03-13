using DrWatson, Revise, MKL, GLMakie
@quickactivate "KalmanFrequency"

# using KalmanFrequency, LinearAlgebra

using Parameters, Random, LinearAlgebra, Optim, FFTW, Statistics, SpecialFunctions, DSP, DataInterpolations, StatsBase, ProgressMeter
includet(srcdir("gendata", "VibData.jl"))
include(srcdir("gendata", "FEM.jl"))
include(srcdir("filter", "StateSpace.jl"))
includet(srcdir("filter", "Regularization.jl"))
includet(srcdir("filter", "KalmanFilter.jl"))
include(srcdir("utils", "NoiseUtils.jl"))
includet(srcdir("utils", "PlotUtils.jl"))

## Définition de la bande de fréquence d'intérêt
fmin = 100. # Fréquence minimale [Hz]
fmax = 1000. # Fréquence maximale [Hz]
Δf = 0.5  # Résolution fréquentielle [Hz]
freq = range(fmin, fmax, step = Δf) # Bande de fréquence d'intérêt

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
# exc_type = :constant # Type d'excitation
exc_type = :random
F = excitation(exc_type, length(freq)) # Vecteur d'excitation
Fref = zeros(ndofs, length(freq))  # Vecteur d'excitation de référence
Fref[exc_dof, :] .= F
Fs = Fref[bm.free_dofs, :]     # Vecteur d'excitation
Fᵣ = Sᵢ*Fs                     # Vecteur d'excitation de référence

## Définition des déformées modales aux ddls mesurés et de reconstruction
pos_m = findall(ωfem/2π .≤ 2freq[end])
ωₙ = ωfem[pos_m]
ϕₘ = Sₘ*ϕ[:, pos_m]          # Déformée propre aux points de mesure
ϕᵢ = Sᵢ*ϕ[:, pos_m]          # Déformée propre aux points de reconstruction

## Génération des données d'accélération bruitée
snr = 25. # Rapport signal sur bruit
yᵣ = Sₘ*resp(Ks, Ms, Cs, Fs, freq) # Signal de référence
ỹ = agwn(yᵣ, snr) # Signal bruité
vary = varest(ỹ, :derrico) # Variance du bruit

## Calcul de la fonction de transfert
H = frf(ωₙ, ξₙ, ϕₘ, ϕᵢ, freq)

## Baysesian Filter
R = Diagonal(vary)
Q = 1e-10I(Nᵢ)

prob_kf = KalmanFilterProblem(H, ỹ, Q, R)
sol_kf = solve(prob_kf)
u_kf = sol_kf.x

## Bayesian Smoother
sol_ks = ksmoother(sol_kf)
u_ks = sol_ks.x

## Bayesian Filter - reverse mode
Hrev = reverse(H)
ỹrev = reverse(ỹ, dims = 2)
prob_kfr = KalmanFilterProblem(Hrev, ỹrev, Q, R)
sol_kfr = solve(prob_kfr)
u_kfr = reverse(sol_kfr.x, dims  = 2)

## Bayesian Smoother - reverse mode
sol_ksr = ksmoother(sol_kfr)
u_ksr = reverse(sol_ksr.x, dims  = 2)

## Fixed-lag Smoother
sol_fls = flsmoother(prob_kf, 10)
u_fls = sol_fls.x

## Visualisation Bayesian Filter
fig_kfw = waterfall_plot(freq, Xᵢ,  real(u_kf), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_kf)), 1.12*maximum(real(u_kf))]);
display(GLMakie.Screen(), fig_kfw);

fig_kfs = plot(freq, F, real(u_kf[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_kfs);

## Visualisation Bayesian Smoother
fig_ksw = waterfall_plot(freq, Xᵢ,  real(u_ks), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_ks)), 1.12*maximum(real(u_ks))]);
display(GLMakie.Screen(), fig_ksw);

fig_kss = plot(freq, real(u_ks[Nₑ, :]), F; xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_kss);

## Visualisation Bayesian Filter - Reverse mode
fig_kfrw = waterfall_plot(freq, Xᵢ,  real(u_kfr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_kfr)), 1.12*maximum(real(u_kfr))]);
display(GLMakie.Screen(), fig_kfrw);

fig_kfrs = plot(freq, real(u_kfr[Nₑ, :]), F; xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_kfrs);

## Visualisation Bayesian Smoother - Reverse mode
fig_ksrw = waterfall_plot(freq, Xᵢ,  real(u_ksr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_ksr)), 1.12*maximum(real(u_ksr))]);
display(GLMakie.Screen(), fig_ksrw);

fig_kssr = plot(freq, real(u_ksr[Nₑ, :]), F; xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_kssr);

## Visualisation Fixed-lag Smoother
fig_flsw = waterfall_plot(freq, Xᵢ,  real(u_fls), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_fls)), 1.12*maximum(real(u_fls))]);
display(GLMakie.Screen(), fig_flsw);

fig_flss = plot(freq, F, real(u_fls[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), fig_flss);
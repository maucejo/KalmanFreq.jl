using DrWatson, Revise, MKL, GLMakie
@quickactivate "KalmanFrequency"

# using KalmanFrequency, LinearAlgebra

using Parameters, Random, LinearAlgebra, Optim, FFTW, Statistics, SpecialFunctions, DSP, DataInterpolations, StatsBase, ProgressMeter
includet(srcdir("gendata", "VibData.jl"))
includet(srcdir("gendata", "FEM.jl"))
includet(srcdir("filter", "StateSpace.jl"))
includet(srcdir("filter", "Regularization.jl"))
includet(srcdir("filter", "RecursiveReg.jl"))
includet(srcdir("utils", "NoiseUtils.jl"))
includet(srcdir("utils", "PlotUtils.jl"))

## Définition de la bande de fréquence d'intérêt
fmin = 100. # Fréquence minimale [Hz]
fmax = 1000. # Fréquence maximale [Hz]
Δf = 1  # Résolution fréquentielle [Hz]
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
exc_type = :constant # Type d'excitation
# exc_type = :random
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

## Recursive Regularization
prob = RecursiveRegProblem(H, ỹ, R)
u_rr = solve(prob);

## Recursive Regularization - Reverse mode
Hrev = reverse(H)
ỹrev = reverse(ỹ, dims = 2)
prob_r = RecursiveRegProblem(Hrev, ỹrev, R)
u_rrr = reverse(solve(prob_r), dims = 2);

## Visualisation Bayesian Filter
figw = waterfall_plot(freq, Xᵢ,  real(u_rr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_rr)), 1.12*maximum(real(u_rr))]);
display(GLMakie.Screen(), figw);

figs = plot(freq, real(u_rr[Nₑ, :]), F; xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), figs);

## Visualisation Bayesian Filter - Reverse mode
figwr = waterfall_plot(freq, Xᵢ,  real(u_rrr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Amplitude (N)", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_rrr)), 1.12*maximum(real(u_rrr))]);
display(GLMakie.Screen(), figwr);

figsr = plot(freq, real(u_rrr[Nₑ, :]), F; xlab = "Frequency (Hz)", ylab = "Amplitude (N)", lw = 2.);
display(GLMakie.Screen(), figsr);
using DrWatson, MKL, GLMakie, LinearAlgebra
@quickactivate "KalmanFreq"

using KalmanFreq

## Definition of the frequency band of interest
fmin = 100.                                    # Min. frequency [Hz]
fmax = 1000.                                   # Max. frequency [Hz]
Δf = 0.5                                       # Frequency resolution [Hz]
freq = range(fmin, fmax, step = Δf)            # Frequency band of interest
freqH = range(fmin - Δf, fmax + Δf, step = Δf) # Frequency band for H

## Definition of the beam
L = 1.                      # Length [m]
b = 3e-2                    # Width [m]
h = 1e-2                    # Thickness [m]
E = 2.1e11                  # Young's modulus [Pa]
ρ = 7850.                   # Density [kg/m³]
ξₙ = 1e-2                   # Damping ratio
beam = Beam(L, b, h, E, ρ)  # Beam structure

## FE mesh definition
Nelem = 100;                 # Number of elements
bm = BeamMesh(0., L, Nelem)  # FE mesh

## Construction of K and M
K, M = assembly(beam, bm)
ndofs = size(K, 1)

## Eigenmodes calculation
Ks = K[bm.free_dofs, bm.free_dofs]
Ms = M[bm.free_dofs, bm.free_dofs]
λ, ϕ = eigen(Ks, Ms)
ωfem = sqrt.(λ)

## Construction of the damping matrix (modal viscous damping)
Cₙ = Diagonal(2ξₙ*ωfem)   # Modal damping matrix
Cs = Ms*ϕ*Cₙ*ϕ'*Ms        # Structural damping matrix

## Measurement mesh definition
Δx = 5e-2       # Space shift between points of measurement
Npoints = 20    # Number of measurement points

# Definition of the measurement and excitation configurations
# 1: Very fine mesh - Npoints = 20
# 2: Fine mesh - Npoints = 10
# 3: Coarse mesh - Npoints = 5
config_mesure, config_id = (1, 1)
Xₘ, Nₒ = select_config(config_mesure, L, Npoints, Δx)
Xᵢ, Nₑ = select_config(config_id, L, Npoints, Δx)
Nₘ = length(Xₘ)  # Number of measurement points
Nᵢ = length(Xᵢ)  # Number of identification points
Xₑ = [Xᵢ[Nₑ]]    # Excitation location

type_mes = :coloc # Collocated configuration
# type_mes = :ncoloc # Non-collocated configuratino
if type_mes == :ncoloc
    Xₘ = Xₘ .+ Δx/2
end

## Defintion of the measurement and identification dofs
meas_dofs, Sₘ = dofs_selection(bm, Xₘ)
id_dofs, Sᵢ = dofs_selection(bm, Xᵢ)
obs_dof = meas_dofs[Nₒ]
exc_dof = id_dofs[Nₑ]

## Definition of the excitation vector
# exc_type = :constant                    # Constant excitation
exc_type = :random                    # Random excitation
F = excitation(exc_type, length(freq))  # Force spectrum
Fref = zeros(ndofs, length(freq))       # Reference force vector
Fref[exc_dof, :] .= F
Fs = Fref[bm.free_dofs, :]              # Force vector for response calculation
Fᵣ = Sᵢ*Fs                              # Reference force vector on id. dofs

## Definition of the mode shapes at the measurement and identification points
pos_m = findall(ωfem/2π .≤ 2freqH[end])
ωₙ = ωfem[pos_m]
ϕₘ = Sₘ*ϕ[:, pos_m]          # Mode shape at the measurement points
ϕᵢ = Sᵢ*ϕ[:, pos_m]          # Mode shapes at the identification points

## Generation of noisy acceleration data
snr = 25.                          # Defined SNR
yᵣ = Sₘ*resp(Ks, Ms, Cs, Fs, freq) # Reference signal
ỹ = agwn(yᵣ, snr)                  # Noisy signal
vary = varest(ỹ)                   # Noise variance estimation

## Transfert function calculation
H = frf(ωₙ, ξₙ, ϕₘ, ϕᵢ, freqH)

## Baysesian Filter
R = Diagonal(vary) # Measurement noise covariance matrix
Q = 1e-10I         # Process noise covariance matrix

prob_bf = BayesianFilterProblem(H, ỹ, Q, R) # Problem definition
sol_bf = solve(prob_bf)                     # Problem resolution
u_bf = sol_bf.u

## Bayesian Filter - reverse mode
Hrev = reverse(H)
ỹrev = reverse(ỹ, dims = 2)
prob_bfr = BayesianFilterProblem(Hrev, ỹrev, Q, R)
sol_bfr = solve(prob_bfr)
u_bfr = reverse(sol_bfr.u, dims  = 2)

## RVR
prob_rvr = RVRProblem(H[2:end-1], ỹ, R)
u_rvr = solve(prob_rvr)

## lq_reg
q = 0.5
prob_lq = LqRegProblem(H[2:end-1], ỹ, q, R)
# prob_lq = LqRegProblem(H[2:end-1], ỹ, q, R, type = :add, method = :lc)
u_lq = solve(prob_lq)

## lpq_reg
p = 2.
prob_lpq = LpqRegProblem(H[2:end-1], ỹ, p, q, R)
u_lpq = solve(prob_lpq)

## Visualiztion Bayesian Filter
fig_bfw = waterfall_plot(freq, Xᵢ, real(u_bf), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [-5e-2 + minimum(real(u_bf)), 1.12*maximum(real(u_bf))]);
display(GLMakie.Screen(), fig_bfw);

fig_bfs = plot(freq, F, real(u_bf[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.);
display(GLMakie.Screen(), fig_bfs);

## Visualization Bayesian Filter - Reverse mode
fig_bfrw = waterfall_plot(freq, Xᵢ,  real(u_bfr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [-5e-2 + minimum(real(u_bfr)), 1.12*maximum(real(u_bfr))]);
display(GLMakie.Screen(), fig_bfrw);

fig_bfrs = plot(freq, F, real(u_bfr[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.);
display(GLMakie.Screen(), fig_bfrs);

## Visualization RVR
fig_rvrw = waterfall_plot(freq, Xᵢ, real(u_rvr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_rvr)), 1.12*maximum(real(u_rvr))]);
display(GLMakie.Screen(), fig_rvrw);

fig_rvrs = plot(freq, F, real(u_rvr[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.);
display(GLMakie.Screen(), fig_rvrs);

## Visualization lq-regularization
fig_lqw = waterfall_plot(freq, Xᵢ, real(u_lq), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_lq)), 1.12*maximum(real(u_lq))]);
display(GLMakie.Screen(), fig_lqw);

fig_lqs = plot(freq, F, real(u_lq[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.);
display(GLMakie.Screen(), fig_lqs);

## Visualization lpq-regularization
fig_lpqw = waterfall_plot(freq, Xᵢ, real(u_lpq), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [-5e-2 + minimum(real(u_lpq)), 1.12*maximum(real(u_lpq))]);
display(GLMakie.Screen(), fig_lpqw);

fig_lpqs = plot(freq, F, real(u_lpq[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.);
display(GLMakie.Screen(), fig_lpqs);
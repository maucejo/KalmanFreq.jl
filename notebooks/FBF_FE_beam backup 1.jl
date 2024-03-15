### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ ca4d9603-4fbf-44c3-9ed9-b17a3223b23f
begin
	using Pkg
	Pkg.activate("../.")

	using DrWatson, MKL, CairoMakie

	using LinearAlgebra
	include(srcdir("utils", "PlotUtils.jl"))

	using PlutoUI
	PlutoUI.TableOfContents(title = "Outline", indent = true)
end

# ╔═╡ 28e613fd-f1e8-4b42-87cb-3b1a309730e9
using KalmanFreq

# ╔═╡ fc7c2b60-acb8-11ee-2700-1b0d54844fb0
md"""
# Numerical experiment - Simply supported beam

The purpose of this notebook is to :

- Test the Bayesian filter developed in the frequency domain
- Compare the performance of this filter with :
  - the RVR (Relevant Vector Regression) method
  - $\ell_q$ regularization
  - mixed-norm $\ell_{p,q}$ regularization
"""

# ╔═╡ d74c87ca-7005-4113-b9f0-2cd1766e7dd5
md"""
## 1. Packages loading
"""

# ╔═╡ 6fc1976f-b3e9-4ee0-9228-726db9e0d252
md"""
## 2. Parameters definition

In this section, we define the main parameters of the study.

### 2.1. Definition of the beam
"""

# ╔═╡ 396a0f33-1725-474e-9798-7775410fef7a
begin
	L = 1.                      # Length [m]
	b = 3e-2                    # Width [m]
	h = 1e-2                    # Thickness [m]
	E = 2.1e11                  # Young's modulus [Pa]
	ρ = 7850.                   # Density [kg/m³]
	ξₙ = 1e-2                   # Damping ratio
	beam = Beam(L, b, h, E, ρ)  # Beam structure

	nothing
end

# ╔═╡ b5d5867a-44b1-457e-9ff1-ceaf3a64c244
md"""
### 2.2. Definition of the frequency band of interest
"""

# ╔═╡ 604e4090-6f6e-4190-9c18-c70be95b928b
begin
	fmin = 100.                                    # Min. frequency [Hz]
	fmax = 1000.                                   # Max. frequency [Hz]
	Δf = 0.5                                       # Frequency resolution [Hz]
	freq = range(fmin, fmax, step = Δf)            # Frequency band of interest
	freqH = range(fmin - Δf, fmax + Δf, step = Δf) # Frequency band for H
	nf = length(freq)

	nothing
end

# ╔═╡ 3e9c773e-e2b3-4acb-97b7-e5ae919ec027
md"""
### 2.3. Definition of the FE mesh of the beam
"""

# ╔═╡ a98cc35e-3f09-4b29-80ee-c778ec5a4fdb
begin
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

	nothing
end

# ╔═╡ 2f688b41-c824-4d91-8365-baea437ebf04
md"""
### 2.4. Defintion of the finite element model
"""

# ╔═╡ bb2a07a9-0d62-493e-a32b-fb1b947f9693
begin
	Nelem = 100;                 # Number of elements
	bm = BeamMesh(0., L, Nelem)  # FE mesh

	# Construction of K and M
	K, M = assembly(beam, bm)
	ndofs = size(K, 1)

	# Eigenmodes calculation
	Ks = K[bm.free_dofs, bm.free_dofs]
	Ms = M[bm.free_dofs, bm.free_dofs]
	λ, ϕ = eigen(Ks, Ms)
	ωfem = sqrt.(λ)

	# Construction of the damping matrix (modal viscous damping)
	Cₙ = Diagonal(2ξₙ*ωfem)   # Modal damping matrix
	Cs = Ms*ϕ*Cₙ*ϕ'*Ms        # Structural damping matrix

	# Defintion of the measurement and identification dofs
	meas_dofs, Sₘ = dofs_selection(bm, Xₘ)
	id_dofs, Sᵢ = dofs_selection(bm, Xᵢ)
	obs_dof = meas_dofs[Nₒ]
	exc_dof = id_dofs[Nₑ]

	# Definition of the mode shapes at the measurement and identification points
	pos_m = findall(ωfem/2π .≤ 2freqH[end])
	ωₙ = ωfem[pos_m]
	ϕₘ = Sₘ*ϕ[:, pos_m]  # Mode shape at the measurement points
	ϕᵢ = Sᵢ*ϕ[:, pos_m]  # Mode shapes at the identification points

	nothing
end

# ╔═╡ 454cbf59-9af7-4748-95fb-0bfda6cda2f0
md"""
## 3. Definition of the excitation
"""

# ╔═╡ 7f756619-bfa0-493c-987b-ae90a0a9473e
begin
	exc_type = :constant                    # Constant excitation
	# exc_type = :random                    # Random excitation
	F = excitation(exc_type, length(freq))  # Force spectrum
	Fref = zeros(ndofs, length(freq))       # Reference force vector
	Fref[exc_dof, :] .= F
	Fs = Fref[bm.free_dofs, :]              # Force vector for response calculation
	Fᵣ = Sᵢ*Fs                              # Reference force vector on id. dofs

	nothing
end

# ╔═╡ 32166d99-0ba6-4be4-95fa-4084c5937264
md"""
## 4. Data generation
"""

# ╔═╡ 050e1ddb-e0b9-4745-9f82-be457fef096a
md"""
### 4.1. Noisy measurements
"""

# ╔═╡ 0cb30594-06b0-4bf0-ab8f-a589c40955d8
# ╠═╡ show_logs = false
begin
	snr = 25.                          # Defined SNR
	yᵣ = Sₘ*resp(Ks, Ms, Cs, Fs, freq) # Reference signal
	ỹ = agwn(yᵣ, snr)                  # Noisy signal
	vary = varest(ỹ)                   # Noise variance estimation

	nothing
end

# ╔═╡ 31d47b06-f5da-43f4-8da4-6521656ab30b
md"""
### 4.2. Transfer functions matrix
"""

# ╔═╡ 0d7f270f-0594-4595-97e8-f4639143f7d1
H = frf(ωₙ, ξₙ, ϕₘ, ϕᵢ, freqH);

# ╔═╡ 5309db46-c412-47c1-82c1-fda561a7247f
md"""
## 5. Application

This section compares the different source identification methods described in the introduction to this notebook.
"""

# ╔═╡ d5f67ab6-54d9-4f3d-9781-46dcfeaabfb4
md"""
### 5.1. Bayesian filtering

Bayesian filtering is based on the definition of a state model of the form :

$\begin{cases}
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{B}_k \, \mathbf{u}_k + \mathbf{w}_k \
\mathbf{y}_k = \mathbf{C}\, \mathbf{x}_k + \mathbf{D}_k\, \mathbf{u}_k + \mathbf{v}_k
\end{cases}$
with $\mathbf{w}_k \sim \mathcal{N}(\mathbf{w}_k | \mathbf{0}, \mathbf{Q})$ and $\mathbf{v}_k \sim \mathcal{N}(\mathbf{v}_k | \mathbf{0}, \mathbf{R})$.

This state model serves as the basis for implementing a sequential Bayesian filter for estimating the state and control of a linear system. For more details, see the companion paper.

#### 5.1.1. Problem definition
"""

# ╔═╡ cde2d460-8490-49c6-9cbb-be8831f65fc6
begin
	R = Diagonal(vary)
	Q = 1e-10I
	prob_bf = BayesianFilterProblem(H, ỹ, Q, R)

	nothing
end

# ╔═╡ 13918ba2-6a2c-45c7-997f-130d47eebe76
md"""
#### 5.1.2. Computation of the solution
"""

# ╔═╡ 7d608bef-421c-4a31-a8d8-6467077b742c
# ╠═╡ show_logs = false
begin
	tbf = @elapsed (membf = @allocated (alloc_bf = @allocations sol_bf = solve(prob_bf)))
	u_bf = sol_bf.u

	# Calcul de l'état
	x_bf = compute_state(sol_bf, H)[1]

	# Performance indicators
	GRE_bf = round(RE(Fᵣ[:], u_bf[:]), digits = 2)
	RE_bf = round(RE(F, u_bf[Nₑ, :]), digits = 2)
	CC_bf = round(Corr(Fᵣ[:], abs.(u_bf[:])), digits = 2)
	GREx_bf = round(RE(yᵣ[:], x_bf[:]), digits = 2)
	REx_bf = round(RE(yᵣ[Nₒ, :], x_bf[Nₒ, :]), digits = 2)
	CCx_bf = round(Corr(abs.(yᵣ[:]), abs.(x_bf[:])), digits = 2)
	t_bf = round(tbf, digits = 3)
	mem_bf = Int64(round(membf/1e6, digits = 0))

	nothing
end

# ╔═╡ 35ec2ed7-3bd4-4e6d-84ef-4a08c21f48db
md"""
#### 5.1.3. Results visualization
"""

# ╔═╡ d81b520a-f0a8-4c1e-87a6-8bd3e1b3a83e
html"""
<center><b>Bayesian filtering - waterfall diagram</b></center>
"""

# ╔═╡ 27fd0de8-0279-4f33-950c-968996691f5b
begin
fig_bfw = waterfall_plot(freq, Xᵢ,  real(u_bf), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_bf)), 1.12*maximum(real(u_bf))])
end

# ╔═╡ ee797fad-da24-49a5-96bf-b37fdf0de5e8
html"""
<center><b>Bayesian filtering - Force spectrum</b></center>
"""

# ╔═╡ f89939c0-e2c1-4a3b-9a6c-afc221d96012
begin
fig_bfs = plot(freq, F, real(sol_bf.u[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.)
end

# ╔═╡ 0a746575-32d8-4899-ba1f-937d4e2b64a3
md"""
### 5.2 Bayesian filtering - Reverse mode

The idea here is to perform the calculation in reverse mode. This may lead to better results than in standard mode. This is linked to the calculation of the initial solution. Indeed, the solution may be of better quality at higher frequencies.

#### 5.2.1. Computation of the solution
"""

# ╔═╡ 66cc7d26-e46f-426c-a97c-9034d3463d0f
# ╠═╡ show_logs = false
begin
	Hrev = reverse(H)
	ỹrev = reverse(ỹ, dims = 2)
	prob_bfr = BayesianFilterProblem(Hrev, ỹrev, Q, R)
	tbfr = @elapsed (membfr = @allocated (alloc_bfr = @allocations sol_bfr = solve(prob_bfr)))
	u_bfr = reverse(sol_bfr.u, dims  = 2)

	# Computation of the state
	x_bfr = reverse(compute_state(sol_bfr, Hrev)[1], dims = 2)

	# Performance indicators
	GRE_bfr = round(RE(Fᵣ[:], u_bfr[:]), digits = 2)
	RE_bfr = round(RE(F, u_bfr[Nₑ, :]), digits = 2)
	CC_bfr = round(Corr(Fᵣ[:], abs.(u_bfr[:])), digits = 2)
	GREx_bfr = round(RE(yᵣ[:], x_bfr[:]), digits = 2)
	REx_bfr = round(RE(yᵣ[Nₒ, : ], x_bfr[Nₒ, :]), digits = 2)
	CCx_bfr = round(Corr(abs.(yᵣ[:]), abs.(x_bfr[:])), digits = 2)
	t_bfr = round(tbfr, digits = 3)
	mem_bfr = Int64(round(membfr/1e6, digits = 0))

	nothing
end

# ╔═╡ ad013490-682c-43f2-92db-127e2af28802
md"""
#### 5.2.2. Results visualization
"""

# ╔═╡ d85db39d-7521-4ea5-8ccd-609bd7b3c321
html"""
<center><b>Bayesian filtering (reverse mode) - Waterfall diagram</b></center>
"""

# ╔═╡ 846d5216-81f7-4aa3-a37e-60a52b5bf856
begin
fig_bfrw = waterfall_plot(freq, Xᵢ,  real(u_bfr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [-1e-1 + minimum(real(u_bfr)), 1.12*maximum(real(u_bfr))])
end

# ╔═╡ 8b50b044-fd34-42d8-a139-876be32f66c2
html"""
<center><b>Bayesian filtering (reverse mode) - Force spectrum</b></center>
"""

# ╔═╡ 229bde4d-bc6b-443c-a6fb-709a0f71ab7e
begin
fig_bfrs = plot(freq, F, real(u_bfr[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.)
end

# ╔═╡ f7f19ede-8490-4f5c-b4b3-6ba61937fc62
md"""
### 5.3. Relevant Vector Regression

The RVR method is a parsimonious regularization method whose aim is to solve the following problem:

$(\widehat{\mathbf{u}}_k, \widehat{\tau}_{ki}) = \underset{(\mathbf{u}_k, \tau_{ki})}{\text{argmax}}\; p(\mathbf{y}_k\, |\, \mathbf{u}_k, \tau_{ki})\, \prod_{i = 1}^{n_u}\, p(\mathbf{u}_{ki}\, |\, \tau_{ki})\, p(\tau_{ki}),$
where :
- $p(\mathbf{y}_k\, |\, \mathbf{u}_k, \tau_{ki}) = \mathcal{N}(\mathbf{y}_k\, |\, \mathbf{H}_k\, \mathbf{u}_k, \mathbf{R}),$
- $p(\mathbf{u}_{ki}\, |\, \tau_{ki}) = \mathcal{N}(\mathbf{u}_{ki}\, |\, 0, \tau_{ki}^{-1}),$
- p(\tau_{ki}) = \mathcal{G}(\tau_{ki}, |\, \alpha_i, \beta_i),$.

The previous problem is solved by an iterative procedure, similar to that presented in [1].

#### 5.3.1. Computation the solution
"""

# ╔═╡ 4550ba40-144c-4ae5-b781-2b5203b48489
# ╠═╡ show_logs = false
begin
	prob_rvr = RVRProblem(H[2:end-1], ỹ, R)
	t_rvr = @elapsed (mem_rvr = @allocated (alloc_rvr = @allocations u_rvr = solve(prob_rvr)))

	# Calcul de l'état
	x_rvr = Matrix{ComplexF64}(undef, Nₘ, nf)
	@inbounds for f ∈ 1:nf
		x_rvr[:, f] .= H[f+1]*u_rvr[:, f]
	end

	# Performance indicators
	GRE_rvr = round(RE(Fᵣ[:], u_rvr[:]), digits = 2)
	RE_rvr = round(RE(F, u_rvr[Nₑ, :]), digits = 2)
	CC_rvr = round(Corr(Fᵣ[:], abs.(u_rvr[:])), digits = 2)
	GREx_rvr = round(RE(yᵣ[:], x_rvr[:]), digits = 2)
	REx_rvr = round(RE(yᵣ[Nₒ, :], x_rvr[Nₒ, :]), digits = 2)
	CCx_rvr = round(Corr(abs.(yᵣ[:]), abs.(x_rvr[:])), digits = 2)
	t_rvr = round(t_rvr, digits = 3)
	mem_rvr = Int64(round(mem_rvr/1e6, digits = 0))

	nothing
end

# ╔═╡ 07ce9938-475f-4007-b895-7c921f0d1c70
md"""
#### 5.3.2. Results visualization
"""

# ╔═╡ 8c42c708-3400-4a8b-b060-23818b337cf9
html"""
<center><b>RVR - Waterfall diagram</b></center>
"""

# ╔═╡ 031b3992-29de-4155-915d-0c78ab85ca3d
begin
fig_rvrw = waterfall_plot(freq, Xᵢ,  real(u_rvr), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_rvr)), 1.12*maximum(real(u_rvr))])
end

# ╔═╡ d3daf6b0-6926-418c-a768-6a72ba2756b9
html"""
<center><b>RVR - Force spectrum</b></center>
"""

# ╔═╡ cc86a5e3-e772-408e-9af9-e6f25ff86433
begin
fig_rvrs = plot(freq, F, real(u_rvr[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.)
end

# ╔═╡ e2b4349c-c0d9-4e27-b2d4-3db2bc704eaa
md"""
### 5.4. Multiplicative $\ell_q$-regularization

In this method, the excitation field is the solution of the following minimization problem, known as multiplicative regularization:

$widehat{\mathbf{u}}_k = \underset{\mathbf{u}_k}{\text{argmin}}; \Vert \mathbf{y}_k - \mathbf{H}_k\, \mathbf{u}_k\Vert_\mathbf{R}^2 \cdot \Vert \mathbf{u}_k\Vert_q^q.$

The problem is solved iteratively using the algorithm described in [2].

#### 5.4.1. Computation of the solution
"""

# ╔═╡ 7a6dd458-4c14-4fd1-b774-10625c46accb
# ╠═╡ show_logs = false
begin
	q = 0.5
	prob_lq = LqRegProblem(H[2:end-1], ỹ, q, R)
	t_lq = @elapsed (mem_lq = @allocated (alloc_lq = @allocations u_lq = solve(prob_lq)))

	# Computation of the state
	x_lq = Matrix{ComplexF64}(undef, Nₘ, nf)
	@inbounds for f ∈ 1:nf
		x_lq[:, f] .= H[f+1]*u_lq[:, f]
	end

	# Performance indicators
	GRE_lq = round(RE(Fᵣ[:], u_lq[:]), digits = 2)
	RE_lq = round(RE(F, u_lq[Nₑ, :]), digits = 2)
	CC_lq = round(Corr(Fᵣ[:], abs.(u_lq[:])), digits = 2)
	GREx_lq = round(RE(yᵣ[:], x_lq[:]), digits = 2)
	REx_lq = round(RE(yᵣ[Nₒ, :], x_lq[Nₒ, :]), digits = 2)
	CCx_lq = round(Corr(abs.(yᵣ[:]), abs.(x_lq[:])), digits = 2)
	t_lq = round(t_lq, digits = 3)
	mem_lq = Int64(round(mem_lq/1e6, digits = 0))

	nothing
end

# ╔═╡ f2871e83-4bea-4b34-a551-057f4cf49125
md"""
#### 5.4.2. Results visualization
"""

# ╔═╡ 08f2cb7e-8983-47f8-9b57-5190ef6df525
html"""
<center><b>ℓ<sub><i>q</i></sub>-regularization - Waterfall diagram</b></center>
"""

# ╔═╡ 9494cb46-20f7-4549-91e3-bab8ed8331ac
begin
fig7 = waterfall_plot(freq, Xᵢ,  real(u_lq), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_lq)), 1.12*maximum(real(u_lq))])
end

# ╔═╡ 3dea323f-5030-4ffa-b9e4-52b47f861439
html"""
<center><b>ℓ<sub><i>q</i></sub>-regularization - Force spectrum</b></center>
"""

# ╔═╡ 4ba30d0d-96f6-421c-a844-6aa5356a1559
begin
fig8 = plot(freq, F, real(u_lq[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.)
end

# ╔═╡ 078ccdda-d1a8-4af1-b7f4-8b35e85949b7
md"""
### 5.5. Multiplicative $\ell_{p,q}$-regularization

In this approach, the space-frequency excitation field is a solution of the following minimization problem, known as multiplicative regularization in mixed norm:

\widehat{\mathbf{u}}_k = \underset{\mathbf{u}_k}{\text{argmin}}; \Vert \mathbf{y}_k - \mathbf{H}_k\, \mathbf{u}_k\Vert_\mathbf{R}^2 \cdot \Vert \mathbf{u}_k\Vert_{p,q}^q.$.

The problem is solved iteratively using the algorithm described in [4].

#### 5.5.1. Computation of the solution
"""

# ╔═╡ a8bf0017-062c-499d-8031-3aff402d9840
# ╠═╡ show_logs = false
begin
	p = 2.
	q₀ = 0.5
	prob_lpq = LpqRegProblem(H[2:end-1], ỹ, p, q₀, R)
	t_lpq = @elapsed (mem_lpq = @allocated (alloc_lpq = @allocations u_lpq = solve(prob_lpq)))

	# Calcul de l'état
	x_lpq = Matrix{ComplexF64}(undef, Nₘ, nf)
	@inbounds for f ∈ 1:nf
		x_lpq[:, f] .= H[f+1]*u_lpq[:, f]
	end

	# Performance indicators
	GRE_lpq = round(RE(Fᵣ[:], u_lpq[:]), digits = 2)
	RE_lpq = round(RE(F, u_lpq[Nₑ, :]), digits = 2)
	CC_lpq = round(Corr(Fᵣ[:], abs.(u_lpq[:])), digits = 2)
	GREx_lpq = round(RE(yᵣ[:], x_lpq[:]), digits = 2)
	REx_lpq = round(RE(yᵣ[Nₒ, :], x_lpq[Nₒ, :]), digits = 2)
	CCx_lpq = round(Corr(abs.(yᵣ[:]), abs.(x_lpq[:])), digits = 2)
	t_lpq = round(t_lpq, digits = 3)
	mem_lpq = Int64(round(mem_lpq/1e6, digits = 0))

	nothing
end

# ╔═╡ 6f03f7ba-79ac-4180-a6c0-44fd4c59033f
md"""
#### 5.5.2. Results visualization
"""

# ╔═╡ fc7e7ae9-cb87-4766-b0ab-934e08043571
html"""
<center><b>ℓ<sub><i>pq</i></sub>-regularization - Waterfall diagram</b></center>
"""

# ╔═╡ f275bea1-63f6-4093-8399-5fde069a38ba
begin
fig9 = waterfall_plot(freq, Xᵢ,  real(u_lpq), xlab = "Frequency (Hz)", ylab = "Location (m)", zlab = "Force (N) - Real part", xlim = [fmin, fmax], ylim = [0, L], zlim = [minimum(real(u_lpq)), 1.12*maximum(real(u_lpq))])
end

# ╔═╡ 6ca3000d-d7d3-436e-86b2-96206cbd6a1a
html"""
<center><b>ℓ<sub><i>pq</i></sub>-regularization - Force spectrum</b></center>
"""

# ╔═╡ 342a0bda-c3b3-4a7a-86c1-633f0f9234f7
begin
fig10 = plot(freq, F,real(u_lpq[Nₑ, :]); xlab = "Frequency (Hz)", ylab = "Force (N) - Real part", lw = 2.)
end

# ╔═╡ 2b2ae0de-371d-446e-bef4-a331466b4f47
md"""
### 5.6. State reconstruction - Results
"""

# ╔═╡ 5be53a0e-d32d-4a05-8894-305a2c41fd4e
html"""
<center><b>Bayesian filtering - Acceleration spectrum</b></center>
"""

# ╔═╡ 7ec1823d-3d32-4c44-8c4d-cd1836d4e3fd
begin
	y1_bf = @. 20log10(abs(x_bf[Nₒ, :]))
	y2 = @. 20log10(abs(ỹ[Nₒ, :]))
	y3 = @. 20log10(abs(yᵣ[Nₒ, :]))
	fig_state_bf = plot(freq, y1_bf, y2, y3, xlab = "Frequency (Hz)", ylab = "Amplitude (dB - ref: 1m/s²)", lw = 2.);
end

# ╔═╡ 54118661-e823-41d7-83c2-e8441ab58761
html"""
<center><b>Bayesian smoothing - Acceleration spectrum</b></center>
"""

# ╔═╡ 30d0089a-312e-44c6-8f0f-b0137d172e0c
begin
	y1_bs = @. 20log10(abs(x_bs[Nₒ, :]))
	fig_state_bs = plot(freq, y1_bs, y2, y3, xlab = "Frequency (Hz)", ylab = "Amplitude (dB - ref: 1m/s²)", lw = 2.);
end

# ╔═╡ 80281c61-e674-44c4-a7d3-84f9daf7f49d
html"""
<center><b>Bayesian filtering (reverse mode) - Acceleration spectrum</b></center>
"""

# ╔═╡ 6c8a8a9c-2944-40b0-96b7-b723556e24bd
begin
	y1_bfr = @. 20log10(abs(x_bfr[Nₒ, :]))
	fig_state_bfr = plot(freq, y1_bfr, y2, y3, xlab = "Frequency (Hz)", ylab = "Amplitude (dB - ref: 1m/s²)", lw = 2.);
end

# ╔═╡ 8ad4d17d-28e3-4318-8e98-7cb41d075934
html"""
<center><b>Bayesian smoothing (reverse mode) - Acceleration spectrum</b></center>
"""

# ╔═╡ c20adf6c-eb4c-4084-b81b-b23886c33c7e
begin
	y1_bsr = @. 20log10(abs(x_bsr[Nₒ, :]))
	fig_state_bsr = plot(freq, y1_bsr, y2, y3, xlab = "Frequency (Hz)", ylab = "Amplitude (dB - ref: 1m/s²)", lw = 2.);
end

# ╔═╡ e894ec9f-6720-45d5-81d7-3c6d00c2dbb1
html"""
<center><b>RVR - Acceleration spectrum</b></center>
"""

# ╔═╡ d3a73e52-a9bc-4b06-9059-d7fae4cb6453
begin
	y1_rvr = @. 20log10(abs(x_rvr[Nₒ, :]))
	fig_state_rvr = plot(freq, y1_rvr, y2, y3, xlab = "Frequency (Hz)", ylab = "Amplitude (dB - ref: 1m/s²)", lw = 2.);
end

# ╔═╡ bdc93a9c-1a48-4cf9-83f0-141fddb9605a
html"""
<center><b>ℓ<sub><i>q</i></sub>-regularization - Acceleration spectrum</b></center>
"""

# ╔═╡ 2615f0d4-2eb5-49a4-9be0-ba5130415ee7
begin
	y1_lq = @. 20log10(abs(x_lq[Nₒ, :]))
	fig_state_lq = plot(freq, y1_lq, y2, y3, xlab = "Frequency (Hz)", ylab = "Amplitude (dB - ref: 1m/s²)", lw = 2.);
end

# ╔═╡ cb9ac92e-c4bf-42df-b5bf-67d94849d442
html"""
<center><b>ℓ<sub><i>pq</i></sub>-regularization - Acceleration spectrum</b></center>
"""

# ╔═╡ 09a41c97-d677-4380-9e09-892df00585a6
begin
	y1_lpq = @. 20log10(abs(x_lpq[Nₒ, :]))
	fig_state_lpq = plot(freq, y1_lpq, y2, y3, xlab = "Frequency (Hz)", ylab = "Amplitude (dB - ref: 1m/s²)", lw = 2.);
end

# ╔═╡ e2eba367-43fd-4563-b008-ace16f19a377
md"""
## 7. Analyse des résultats
"""

# ╔═╡ 4ceb1b40-4279-4bf4-a3df-4c2c01d562c6
Markdown.parse("""
**Reminder of the problem :**
* Frequency range of interest : [`$fmin` Hz, `$fmax` Hz]
* Excitation type : `$exc_type`
* Configuration : `$type_mes`
* SNR : `$snr` dB


**Results - Excitation**

|            Method            |   GRE (%)  |    RE (%)   |   CC (%)  |  Time (s) | Mem. (MiB) | Alloc.|
|:----------------------------:|:----------:|:-----------:|:---------:|:----------:|:-----------:|:-----:|
|       Bayesian Filter        | `$GRE_bf`  |  `$RE_bf`   | `$CC_bf`  | `$t_bf` | `$mem_bf ` | `$alloc_bf` |
| Bayesian Filter (rev. mode)  | `$GRE_bfr` |  `$RE_bfr`  | `$CC_bfr` | `$t_bfr` | `$mem_bfr ` | `$alloc_bfr` |
|            RVR               | `$GRE_rvr` |  `$RE_rvr`  | `$CC_rvr` | `$t_rvr` | `$mem_rvr` | `$alloc_rvr` |
| ``\\ell_q``-regularization   | `$GRE_lq`  |  `$RE_lq`   | `$CC_lq`  | `$t_lq` | `$mem_lq` | `$alloc_lq` |
|``\\ell_{p,q}``-regularization| `$GRE_lpq` |  `$RE_lpq`  | `$CC_lpq` | `$t_lpq` | `$mem_lpq` | `$alloc_lpq` |

**Results - State**

|            Method            |    GRE (%)  |     RE (%)   |    CC (%)  |
|:----------------------------:|:-----------:|:------------:|:----------:|
|       Bayesian Filter        | `$GREx_bf`  |  `$REx_bf`   | `$CCx_bf`  |
| Bayesian Filter (rev. mode)  | `$GREx_bfr` |  `$REx_bfr`  | `$CCx_bfr` |
|            RVR               | `$GREx_rvr` |  `$REx_rvr`  | `$CCx_rvr` |
| ``\\ell_q``-regularization   | `$GREx_lq`  |  `$REx_lq`   | `$CCx_lq`  |
|``\\ell_{p,q}``-regularization| `$GREx_lpq` |  `$REx_lpq`  | `$CCx_lpq` |
""")

# ╔═╡ e75cfe20-f80c-464f-8f45-629d11f3c23c
md"""
## 8. Bibliography

[1] M. E. Tipping. Sparse Bayesian Learning and the Relevance Vector Machine. _Journal of Machine Learning Research_. 1, pp. 211-244. 2001.

[2] M. Aucejo, O. De Smet. Multi-parameter multiplicative regularization : An application to force reconstruction problems. _Journal of Sound and Vibration_. 469, pp. 115135. 2020.

[3] M. Aucejo, O. De Smet. A novel algorithm for solving multiplicative mixed-norm regularization problems. _Mechanical Systems and Signal Processing_. 144, pp. 106887. 2020.
"""

# ╔═╡ Cell order:
# ╟─fc7c2b60-acb8-11ee-2700-1b0d54844fb0
# ╟─d74c87ca-7005-4113-b9f0-2cd1766e7dd5
# ╠═ca4d9603-4fbf-44c3-9ed9-b17a3223b23f
# ╠═28e613fd-f1e8-4b42-87cb-3b1a309730e9
# ╟─6fc1976f-b3e9-4ee0-9228-726db9e0d252
# ╠═396a0f33-1725-474e-9798-7775410fef7a
# ╟─b5d5867a-44b1-457e-9ff1-ceaf3a64c244
# ╠═604e4090-6f6e-4190-9c18-c70be95b928b
# ╟─3e9c773e-e2b3-4acb-97b7-e5ae919ec027
# ╠═a98cc35e-3f09-4b29-80ee-c778ec5a4fdb
# ╟─2f688b41-c824-4d91-8365-baea437ebf04
# ╠═bb2a07a9-0d62-493e-a32b-fb1b947f9693
# ╟─454cbf59-9af7-4748-95fb-0bfda6cda2f0
# ╠═7f756619-bfa0-493c-987b-ae90a0a9473e
# ╟─32166d99-0ba6-4be4-95fa-4084c5937264
# ╟─050e1ddb-e0b9-4745-9f82-be457fef096a
# ╠═0cb30594-06b0-4bf0-ab8f-a589c40955d8
# ╟─31d47b06-f5da-43f4-8da4-6521656ab30b
# ╠═0d7f270f-0594-4595-97e8-f4639143f7d1
# ╟─5309db46-c412-47c1-82c1-fda561a7247f
# ╟─d5f67ab6-54d9-4f3d-9781-46dcfeaabfb4
# ╠═cde2d460-8490-49c6-9cbb-be8831f65fc6
# ╟─13918ba2-6a2c-45c7-997f-130d47eebe76
# ╠═7d608bef-421c-4a31-a8d8-6467077b742c
# ╟─d885223a-5870-44bc-9736-6b55a934c161
# ╠═e25c024a-5bc3-4dee-806e-e8798ca5806f
# ╟─35ec2ed7-3bd4-4e6d-84ef-4a08c21f48db
# ╟─d81b520a-f0a8-4c1e-87a6-8bd3e1b3a83e
# ╟─27fd0de8-0279-4f33-950c-968996691f5b
# ╟─ee797fad-da24-49a5-96bf-b37fdf0de5e8
# ╟─f89939c0-e2c1-4a3b-9a6c-afc221d96012
# ╟─80116917-992c-409b-81cc-22e6caa976c0
# ╟─943d9f1e-ee2e-481d-86de-26968b62f35f
# ╟─e68080db-fae5-4b72-b2e7-e676ee562692
# ╟─a34bd622-3d79-48ec-ac09-4d134a0925d9
# ╟─0a746575-32d8-4899-ba1f-937d4e2b64a3
# ╠═66cc7d26-e46f-426c-a97c-9034d3463d0f
# ╟─bc442e82-7a22-410e-8a2a-52e4c34b9287
# ╠═0196e1ca-78db-4938-ba51-007d3d134b71
# ╟─ad013490-682c-43f2-92db-127e2af28802
# ╟─d85db39d-7521-4ea5-8ccd-609bd7b3c321
# ╟─846d5216-81f7-4aa3-a37e-60a52b5bf856
# ╟─8b50b044-fd34-42d8-a139-876be32f66c2
# ╟─229bde4d-bc6b-443c-a6fb-709a0f71ab7e
# ╟─72219542-a7c1-42d2-acd2-eb8620cdb0da
# ╟─585bae02-0407-422a-bb74-405113eabb78
# ╟─f2543729-9198-4f7d-b128-5b1bde6759f2
# ╟─fa4e9622-6507-40b6-88d2-2c2e07e9ef6f
# ╟─f7f19ede-8490-4f5c-b4b3-6ba61937fc62
# ╠═4550ba40-144c-4ae5-b781-2b5203b48489
# ╟─07ce9938-475f-4007-b895-7c921f0d1c70
# ╟─8c42c708-3400-4a8b-b060-23818b337cf9
# ╟─031b3992-29de-4155-915d-0c78ab85ca3d
# ╟─d3daf6b0-6926-418c-a768-6a72ba2756b9
# ╟─cc86a5e3-e772-408e-9af9-e6f25ff86433
# ╟─e2b4349c-c0d9-4e27-b2d4-3db2bc704eaa
# ╠═7a6dd458-4c14-4fd1-b774-10625c46accb
# ╟─f2871e83-4bea-4b34-a551-057f4cf49125
# ╟─08f2cb7e-8983-47f8-9b57-5190ef6df525
# ╟─9494cb46-20f7-4549-91e3-bab8ed8331ac
# ╟─3dea323f-5030-4ffa-b9e4-52b47f861439
# ╟─4ba30d0d-96f6-421c-a844-6aa5356a1559
# ╟─078ccdda-d1a8-4af1-b7f4-8b35e85949b7
# ╠═a8bf0017-062c-499d-8031-3aff402d9840
# ╟─6f03f7ba-79ac-4180-a6c0-44fd4c59033f
# ╟─fc7e7ae9-cb87-4766-b0ab-934e08043571
# ╟─f275bea1-63f6-4093-8399-5fde069a38ba
# ╟─6ca3000d-d7d3-436e-86b2-96206cbd6a1a
# ╟─342a0bda-c3b3-4a7a-86c1-633f0f9234f7
# ╟─2b2ae0de-371d-446e-bef4-a331466b4f47
# ╟─5be53a0e-d32d-4a05-8894-305a2c41fd4e
# ╟─7ec1823d-3d32-4c44-8c4d-cd1836d4e3fd
# ╟─54118661-e823-41d7-83c2-e8441ab58761
# ╟─30d0089a-312e-44c6-8f0f-b0137d172e0c
# ╟─80281c61-e674-44c4-a7d3-84f9daf7f49d
# ╟─6c8a8a9c-2944-40b0-96b7-b723556e24bd
# ╟─8ad4d17d-28e3-4318-8e98-7cb41d075934
# ╟─c20adf6c-eb4c-4084-b81b-b23886c33c7e
# ╟─e894ec9f-6720-45d5-81d7-3c6d00c2dbb1
# ╟─d3a73e52-a9bc-4b06-9059-d7fae4cb6453
# ╟─bdc93a9c-1a48-4cf9-83f0-141fddb9605a
# ╟─2615f0d4-2eb5-49a4-9be0-ba5130415ee7
# ╟─cb9ac92e-c4bf-42df-b5bf-67d94849d442
# ╟─09a41c97-d677-4380-9e09-892df00585a6
# ╟─e2eba367-43fd-4563-b008-ace16f19a377
# ╟─4ceb1b40-4279-4bf4-a3df-4c2c01d562c6
# ╟─e75cfe20-f80c-464f-8f45-629d11f3c23c

module KalmanFrequency

using PrecompileTools

using MKL

using Parameters, Random, LinearAlgebra, Optim, FFTW, Statistics, SpecialFunctions, DataInterpolations, StatsBase, ProgressMeter, BlockDiagonals, SparseArrays, LazyGrids

import DSP: conv

include("gendata/VibData.jl")
export Beam, eigval, eigmode, excitation, resp, frf, frf_modal

include("gendata/FEM.jl")
export BeamMesh, assembly, dofs_selection, select_config

include("utils/NoiseUtils.jl")
export agwn, estimated_SNR, varest

include("filter/StateSpace.jl")
export BayesianFilterProblem, compute_state

include("filter/Regularization.jl")
include("utils/LinalgUtils.jl")
export RVRProblem, LqRegProblem, LpqRegProblem, RE, Corr

include("filter/BayesianFilter.jl")
include("filter/BayesianSmoother.jl")
export solve, bsmoother

include("filter/KalmanFilter.jl")
export KalmanFilterProblem, ksmoother

include("filter/RecursiveReg.jl")
export RecursiveRegProblem

include("precompilation/precompilation.jl")

end # KalmanFrequency
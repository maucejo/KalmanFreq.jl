module KalmanFreq

using PrecompileTools

using MKL

using Parameters, Random, LinearAlgebra, Optim, Statistics, FFTW, SpecialFunctions, StatsBase, ProgressMeter, BlockDiagonals, SparseArrays, LazyGrids

include("gendata/VibData.jl")
export Beam, eigval, eigmode, excitation, resp, frf, frf_modal

include("gendata/FEM.jl")
export BeamMesh, assembly, dofs_selection, select_config

include("utils/NoiseUtils.jl")
export agwn, estimated_SNR, varest

include("estimation/StateSpace.jl")
include("estimation/BayesianFilter.jl")
export BayesianFilterProblem, solve, compute_state

include("estimation/Regularization.jl")
export RVRProblem, LqRegProblem, LpqRegProblem

include("estimation/PerfIndicators.jl")
export RE, CorrCoeff

include("utils/LinalgUtils.jl")

include("precompilation/precompilation.jl")

end # KalmanFrequency
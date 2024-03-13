# KalmanFreq.jl

[![Generic badge](https://img.shields.io/badge/Licence-MIT-<COLOR>.svg)]()  [![Generic badge](https://img.shields.io/badge/Version-0.1.0-blue.svg)]()

KalmanFrequency.jl is a [Julia](https://julialang.org/) project, powered by  [DrWatson.jl](https://juliadynamics.github.io/DrWatson.jl/stable/), which is the companion code base of the paper:

M. Aucejo and O. De Smet. A frequency-domain sequential Bayesian filter for sparse and broadband force estimation problems. *Mechanical Systems and Signal Processing*. vol. xxx, xxxxx, 2024.

## Content

This project contains 3 folders :

* `notebooks` - [Pluto](https://plutojl.org/) notebook used to compute the result of the numerical experiment (section 4)

* `scripts` - Script version of the notebook used to compute the results of the numerical experiment.

* `src`
   * `KalmanFreq.jl`: file defining the module *KalmanFreq*
   * `gendata` - Functions used to generate the vibration data
   * `estimation` - Estimation methods (Bayesian Filter, RVR, multiplicative $\ell_q$ and $\ell_{p,q}$-regularizations), State space model and performance indicators
   * `utils` - Utility functions (Noise estimation, Plots, LinearAlgebra)
   * `precompilation` - Workload used to precompile the module


## How to reproduce this project

This code base is using the [Julia Language](https://julialang.org/) and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to make a reproducible scientific project

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "KalmanFreq"
```
which auto-activate the project and enable local path handling from DrWatson.

## License

MIT licensed

Copyright (C) 2022 Mathieu AUCEJO (maucejo)


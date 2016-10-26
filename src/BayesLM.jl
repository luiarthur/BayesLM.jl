__precompile__()
module BayesLM

import Base.show
import Base.summary

using Distributions

export lm, summary, dic, show

include("MCMC.jl")
include("common.jl")
include("lm.jl")

end # module BayesLM

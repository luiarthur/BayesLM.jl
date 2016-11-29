__precompile__()
module BayesLM

import Base.show
import Base.summary

using Distributions

export lm, summary, dic, show, cpo, Hyper, glm, bclogit

include("MCMC.jl")
include("common.jl")
include("lm.jl")
include("anova.jl")
include("glm.jl")
include("manova.jl")
include("survreg.jl")
include("bclogit.jl")

end # module BayesLM

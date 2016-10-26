__precompile__()
module LinearModels

import Base.show
using Distributions

export lm, post_summary, dic, show

include("MCMC.jl")
include("common.jl")
include("lm.jl")

end # module LinearModels

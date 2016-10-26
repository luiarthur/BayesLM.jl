__precompile__()
module LinearModels

import MCMC
import Base.show
using Distributions

export lm, post_summary, dic, show

include("common.jl")
include("lm.jl")

end # module LinearModels

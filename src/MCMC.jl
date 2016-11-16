# MCMC.jl
"""
My MCMC stuff

Currently the only method is gibbs. Eventually, I would
like to add a Metropolis Hastings method.
"""
module MCMC

import Distributions
export gibbs, metropolis
"""
Generalized Gibbs sampler.

# Arguments
* `init`: Initial values for gibbs sampler
* `update`: function which takes a state::State as the only input and returns the update
* `B`: number of MCMC samples desired
* `burn`: burn-in
* `printFreq`: shows progress bar every printFreq%. (default: 0 => don't print progress)

# Example:

```julia
using Distributions

# Generate data
const n = 1000
const (mu,sig2) = (5,2)
const y = rand(Normal(mu,sqrt(sig2)),n)

# precomputes
const ybar = mean(y)
const (sig2_a, sig2_b) = (2.0, 1.0)
rig(shape::Float64,rate::Float64) = 1 / rand(Gamma(shape,1/rate))

# Define State
immutable State
  mu::Float64
  sig2::Float64
end

# define an update function for State
function update(state::State)
  const newMu = rand(Normal(ybar,sqrt(state.sig2/n)))
  const newSig2 = rig(sig2_a+n/2, sig2_b + sum((y-newMu).^2)/2)
  State(newMu,newSig2)
end

# run gibbs
@time out = gibbs(State(0,1),update,10000,1000);

## post processing
const postMu = map(o -> o.mu, out)
const postSig2 = map(o -> o.sig2, out)

```
"""

function gibbs{T}(init::T, update, B::Int, burn::Int; printFreq::Int=0)
  const out = Array{T,1}( (B+burn) )
  out[1] = init
  for i in 2:(B+burn)
    out[i] = update(out[i-1])
    if printFreq > 0 && i % printFreq == 0
      print("\rProgress: ",i,"/",B+burn)
    end
  end
  out[ (burn+1):end ]
end

"""
Univariate Metropolis step (with Normal proposal)

    metropolis(curr, loglike_plus_logprior, candSig::Real; 
               inbounds = x-> -Inf<x<Inf)

# Arguments
* `curr`: current value of parameter
* `loglike_plus_logprior`: log-likelihoodd plus log-prior as function of the parameter to be updated
* `candSig`: The sd of the normal proposal
* `inbounds`: A function which checks if the current parameter is in the support
"""
function metropolis(curr, loglike_plus_logprior, candSig::Real; 
                    inbounds = x-> -Inf<x<Inf)
  # need to do autotune

  const cand = rand(Distributions.Normal(curr,candSig))

  if inbounds(cand)
    p = loglike_plus_logprior(cand) - loglike_plus_logprior(curr)
    p > log(rand()) ?  cand : curr
  else
    curr
  end

end

end # module

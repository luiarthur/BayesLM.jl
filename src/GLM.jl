module GLM

import MCMC
using Distributions

export lm

rig{A <: Real, B <: Real}(a::A,b::B) = 1 / rand( Gamma(a,1/b) )

function lm(y::Vector, X::Matrix; B::Int=10000)
  const (N,K) = size(X)
  const XXi = inv(X'X)
  const a = (N-K) / 2

  # ???

end

function lm(y:: Vector, X::Matrix, V::Matrix)
  # ???
end

end # GLM

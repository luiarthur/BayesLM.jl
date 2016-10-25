module GLM

import MCMC
using Distributions

export lm, post_summary

rig{A <: Real, B <: Real}(a::A,b::B) = 1 / rand( Gamma(a,1/b) )

immutable State_lm
  b::Vector
  sig2::Float64
end


function post_summary(out::Array{State_lm,1}; alpha=.05)
  post_beta= hcat(map(o -> o.b, out)...)'
  mean_beta = mean(post_beta,1)
  std_beta = std(post_beta,1)
  const (N,P) = size(post_beta)
  zeroInCI = Array{String,1}(P)
  for p in 1:P
    q = quantile(post_beta[:,p],[alpha/2,1-alpha/2])
    zeroInCI[p] = q[1] <= 0 <= q[2] ? "" : "*"
  end

  println("Coefficients:\tmean\tstd\t≠0")
  for k in 1:P
    @printf "\t\t%.4f\t%.4f\t%s\n" mean_beta[k] std_beta[k] zeroInCI[k]
  end

  post_sig = map(o -> sqrt(o.sig2), out)
  @printf "σ: %.4f ± %.4f\n"  mean(post_sig)  std(post_sig)

  # print out DIC
end

function lm(y::Vector, X::Matrix; B::Int=10000, burn::Int=10, addIntercept::Bool=true)
  if (addIntercept) 
    X = [ones(size(X,1)) X]
  end
  const (N,K) = size(X)
  const XXi = inv(X'X)
  const a = (N-K) / 2
  const BETA_HAT = XXi * X'y # FIXME: use QR to do this instead?
  const beta_init = fill(0.0,K)
  const s2_init = 1.0


  function update(state::State_lm)
    const s2_new = rig(a, sum((y-X*state.b).^2)/2)
    const b_new = rand(MultivariateNormal(BETA_HAT,s2_new*XXi))
    State_lm(b_new,s2_new)
  end

  MCMC.gibbs(State_lm(beta_init,s2_init), update, B, burn)
end

#= For general covariance matrix
function lm(y:: Vector, X::Matrix, V::Matrix)
end
=#

end # GLM

#=
include("GLM.jl")
using GLM, RCall
n = 100
X = randn(n,1)
b = [0,3]
y = [ones(n) X] * b + randn(n)
model = lm(y,X);
post_summary(model,alpha=0)

@rput y X;
R"summary(lm(y~X))"
=#

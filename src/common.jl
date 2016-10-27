const quants = [0., .025, .25, .5, .75, .975, 1.]

rig{A <: Real, B <: Real}(a::A,b::B) = 1 / rand( Gamma(a,1/b) )

function dic{T}(post_samples::Vector{T},loglikelihood)
  const D = -2 * loglikelihood.(post_samples)
  return mean(D) + var(D) / 2
end

function cpo{T}(post_params::T, f, X::Matrix{Float64}) # f = pdf
  const N = size(X,1)
  const J = length(post_params)
  return 1 ./ mean([ 1 / f(post_params[b], X[i,:]) for i in 1:N, b in 1:B ], 2)
end

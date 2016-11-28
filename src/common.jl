"""
symmetrize matrix
"""
sym(M::Matrix{Float64}) = (M' + M) / 2

const quants = [0., .025, .25, .5, .75, .975, 1.]

rig{A <: Real, B <: Real}(a::A,b::B) = 1 / rand( Gamma(a,1/b) )

function dic{T}(post_samples::Vector{T},loglikelihood)
  const D = -2 * loglikelihood.(post_samples)
  return mean(D) + var(D) / 2
end

"""
cpo = conditional posterior ordinate. 
      which is the leave-one-out density evaluated at the
      actual observed y's. used for *comparing* models. Not
      for prediction.
"""
function cpo{T}(post_params::Vector{T}, den, 
                y::Vector{Float64}, 
                X::Matrix{Float64})

  const N = size(X,1)
  const J = length(post_params)
  const CPO = Matrix{Float64}(N,J)

  for i in 1:N
    for j in 1:J
      CPO[i,j] = 1 / den(post_params[j], y[i], X[i,:])
    end
  end
  
  return 1 ./ mean(CPO,2)
end

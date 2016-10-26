const quants = [0., .025, .25, .5, .75, .975, 1.]

rig{A <: Real, B <: Real}(a::A,b::B) = 1 / rand( Gamma(a,1/b) )

function dic(post_samples,loglikelihood)
  const D = -2 * loglikelihood.(post_samples)
  return mean(D) + var(D) / 2
end

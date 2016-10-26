VM = Union{Vector,Matrix}

immutable State_lm
  b::Vector
  sig2::Float64
end

immutable LM
  post_samps::Array{State_lm,1}
  DIC::Float64
end

immutable Summary_lm
  mean_beta::Vector{Float64}
  std_beta::Vector{Float64}
  quantile_beta::Matrix{Float64}
  mean_sig::Float64
  std_sig::Float64
  quantile_sig::Vector{Float64}
  DIC::Float64
  B::Int
end

function show(io::IO, SL::Summary_lm)
  const P = length(SL.mean_beta)
  zeroInCI = [SL.quantile_beta[2,p] <= 0 <= SL.quantile_beta[6,p] ? "" : "*" for p in 1:P]

  @printf "%5s%10s%10s%3s\n" "" "mean" "std" "≠0"
  for k in 1:P
    @printf "%5s%10.4f%10.4f%3s\n" string("β",k-1) SL.mean_beta[k] SL.std_beta[k] zeroInCI[k]
  end
  @printf "%5s%10.4f%10.4f\n" "σ" SL.mean_sig SL.std_sig
  @printf "%5s%10.2f\n" "DIC" SL.DIC
end

function dic{T <: VM}(post_samples::Array{State_lm,1}, y::Vector, X::T; 
                      addIntercept=true)
  if addIntercept
    X = [ones(size(X,1)) X]
  end
  const (N,P) = size(X)

  function loglike(param)
    -N/2 * log(2pi*param.sig2) - sum((y - X*param.b).^2) / (2param.sig2)
  end

  dic(post_samples,loglike)
end

function summary(out::LM; alpha=.05)
  const post_beta= hcat(map(o -> o.b, out.post_samps)...)'
  const (B,P) = size(post_beta)
  const mean_beta = vec(mean(post_beta,1))
  const std_beta = vec(std(post_beta,1))
  const quantile_beta = hcat([ quantile(post_beta[:,p], quants) for p in 1:P]...)
  const post_sig = map(o -> sqrt(o.sig2), out.post_samps)
  const mean_sig = mean(post_sig)
  const std_sig = std(post_sig)
  const quantile_sig = quantile(post_sig, quants)

  sum_lm = Summary_lm(mean_beta,std_beta,quantile_beta,
                      mean_sig,std_sig,quantile_sig,
                      out.DIC,B)
  
  return sum_lm
end

function lm{T <: VM}(y::Vector, X::T; B::Int=10000, burn::Int=10, 
                     addIntercept::Bool=true)
  if addIntercept
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

  const post_samps = MCMC.gibbs(State_lm(beta_init,s2_init), update, B, burn)
  const DIC = dic(post_samps,y,X,addIntercept=false)

  LM(post_samps,DIC)
end

#= For general covariance matrix
function lm(y:: Vector, X::Matrix, V::Matrix)
end
=#


#=
using BayesLM
n = 100
X = randn(n)
b = [0,3]
y = [ones(n) X] * b + randn(n)
@time model = lm(y,X);
post_summary(model,alpha=.05)
=#

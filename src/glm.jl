# glm.jl

typealias Hyper Dict{Symbol,Float64}

immutable Param_GLM
  β::Vector{Float64}
  θ::Hyper
end

function θ_vec(θ::Hyper, keys::Vector{Symbol})
  return [θ[k] for k in keys]
end

function toθ(v::Vector{Float64}, keys::Vector{Symbol})
  assert(length(v) == length(keys))
  d = Hyper()
  for i in 1:length(keys)
    d[keys[i]] = v[i]
  end
  return d
end


"""
invlink(xi'β) should be the mean of distribution F
logf(y::Vector{Float64}, mean::Vector{Float64}, θ::Hyper) should return the logden
"""

function glm(y::Vector{Float64}, X::Matrix{Float64}, # include your own intercept!
             Σ_β::Matrix{Float64}, Σ_θ:: Matrix{Float64}, θ_bounds::Matrix{Float64},
             θ_names::Vector{Symbol}, invlink, logf, θ_logprior;
             B::Int=10000, burn::Int=1000)

  assert(size(θ_bounds,2) == 2)
  const θ_lower = θ_bounds[:,1]
  const θ_upper = θ_bounds[:,2]
  β_logprior(β::Vector{Float64}) = 0
  logprior(β::Vector{Float64}, θ::Hyper) = β_logprior(β) + θ_logprior(θ)
  loglike(β::Vector{Float64}, θ::Hyper) = sum(logf(y, invlink.(X*β), θ))
  
  ll_plus_lp(β::Vector{Float64}, θ::Hyper) = loglike(β, θ) + logprior(β, θ)

  function update(p::Param_GLM)
    # Update Coefficients (β)
    β_cand = rand(MvNormal(p.β,Σ_β))
    if ll_plus_lp(β_cand, p.θ) - ll_plus_lp(p.β, p.θ) > log(rand())
      β_new = β_cand
    else
      β_new = copy(p.β)
    end

    # Update Hyper Params (θ)
    θ_curr = θ_vec(p.θ,θ_names)
    θ_cand = rand(MvNormal(θ_curr,Σ_θ))
    θ_cand_s = toθ(θ_cand,θ_names)

    if all(θ_lower .< θ_cand .< θ_upper) && ll_plus_lp(β_new, θ_cand_s) - ll_plus_lp(β_new, p.θ) > log(rand())
      θ_new = θ_cand_s
    else
      θ_new = copy(p.θ)
    end

    return Param_GLM(β_new,θ_new)
  end #update

  const β₀ = zeros(Float64,size(Σ_β,1))
  const θ₀ = toθ(ones(Float64,size(Σ_θ,1)), θ_names)
  const init = Param_GLM(β₀,θ₀)
             
  return MCMC.gibbs(init, update, B, burn)
end #glm

# glm.jl

typealias Hyper Dict{Symbol,Float64}

immutable Param_GLM
  β::Vector{Float64}
  θ::Hyper
end

function θ_vec(θ::Hyper)
  #[getfield(θ,fn) for fn in fieldnames(θ)]
  
end

### FIXME
function toθ(v::Vector{Float64}, D::Datatype)
  D()
end


"""
invlink(xi'β) should be the mean of distribution F
logf{T <: Hyper}(y::Vector{Float64}, mean::Vector{Float64}, θ::T) should return the logden
"""

function glm(y::Vector{Float64}, X::Matrix{Float64}, # include your own intercept!
             Σ_β::Matrix{Float64}, Σ_θ:: Matrix{Float64}, θ_bounds::Matrix{Float64},
             invlink, logf, θ_logprior) 

  assert(size(θ_bounds,2) == 2)
  const θ_lower = θ_bounds[:,1]
  const θ_upper = θ_bounds[:,2]
  β_logprior(β::Vector{Float64}) = 0
  logprior{T <: Hyper}(β::Vector{Float64}, θ::T) = β_logprior(β) + θ_logprior(θ)
  loglike{T <: Hyper}(β::Vector{Float64} ,θ::T) = sum(logf(y, invlink.(X*β), θ))
  
  ll_plus_lp(p::Param_GLM) = loglike(p.β, p.θ) + logprior(p.β, p.θ)

  function update(p::Param_GLM)
    # Update Coefficients (β)
    β_cand = rand(MvNormal(p.β,Σ_β))
    if ll_plus_lp(β_cand, p.θ) - ll_plus_lp(p.β, p.θ) > log(rand())
      β_new = β_cand
    else
      β_new = p.β
    end

    # Update Hyper Params (θ)
    θ_curr = θ_vec(p.θ)
    θ_cand = rand(MvNormal(θ_curr),Σ_θ)
    θ_cand_s = toθ(θ_cand)

    if all(θ_lower .< θ_cand .< θ_upper) && ll_plus_lp(β_new, θ_cand_s) - ll_plus_lp(β_new, p.θ) > log(rand())
      θ_new = θ_cand_s
    else
      θ_new = p.θ
    end

    return Param_GLM(β_new,θ_new)
  end #update

  const init = Param_GLM(zeros(Float64,size(Σ_β,1)), toθ(zeros(Float64,size(Σ_θ,1))))
  return gibbs(init, update, B, burn)
end #glm



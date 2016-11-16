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

immutable GLM
  post_params::Vector{Param_GLM}
  y::Vector{Float64}
  X::Matrix{Float64}
end


"""
invlink(xi'β) should be the mean of distribution F
loglike(y::Vector{Float64}, Xb::Vector{Float64}, θ::Hyper) should return the loglikelihood
"""

lp_θ_default(θ::Hyper) = 0.0
function glm(y::Vector{Float64}, X::Matrix{Float64}, # include your own intercept!
             Σ_β::Matrix{Float64}, loglike; 
             Σ_θ::Matrix{Float64}=eye(1)*1., 
             θ_bounds::Matrix{Float64}=[0. Inf],
             θ_names::Vector{Symbol}=[:empty], 
             θ_logprior=lp_θ_default,
             B::Int=10000, burn::Int=1000, printFreq::Int=0)

  assert(size(θ_bounds,2) == 2)

  const empty = θ_names[1] == :empty
  const θ_lower = θ_bounds[:,1]
  const θ_upper = θ_bounds[:,2]

  β_logprior(β::Vector{Float64}) = 0
  lp(β::Vector{Float64}, θ::Hyper) = β_logprior(β) + θ_logprior(θ)
  ll(β::Vector{Float64}, θ::Hyper) = loglike(y, X*β, θ)
  
  ll_plus_lp(β::Vector{Float64}, θ::Hyper) = ll(β, θ) + lp(β, θ)

  function update(p::Param_GLM)
    # Update Coefficients (β)
    β_cand = rand(MvNormal(p.β,Σ_β))
    if ll_plus_lp(β_cand, p.θ) - ll_plus_lp(p.β, p.θ) > log(rand())
      β_new = β_cand
    else
      β_new = copy(p.β)
    end

    # Update Hyper Params (θ)
    if !empty
      θ_curr = θ_vec(p.θ,θ_names)
      θ_cand = rand(MvNormal(θ_curr,Σ_θ))
      θ_cand_s = toθ(θ_cand,θ_names)

      if all(θ_lower .< θ_cand .< θ_upper) && ll_plus_lp(β_new, θ_cand_s) - ll_plus_lp(β_new, p.θ) > log(rand())
        θ_new = θ_cand_s
      else
        θ_new = copy(p.θ)
      end
    else
      θ_new = p.θ
    end

    return Param_GLM(β_new,θ_new)
  end #update

  const β₀ = zeros(Float64,size(Σ_β,1))
  const θ₀ = toθ(ones(Float64,size(Σ_θ,1)), θ_names)
  const init = Param_GLM(β₀,θ₀)
             
  const post = MCMC.gibbs(init, update, B, burn, printFreq=printFreq)
  return GLM(post,y,X)
end #glm

function stats(v::Vector{Float64}; α::Float64=.05)
  (mean(v), std(v), quantile(v,[α/2, 1-α/2]))
end

function stats(m::Matrix{Float64}; α::Float64=.05)
  const P = size(m,2)
  return (vec(mean(m,1)), vec(std(m,1)), 
          hcat([quantile(m[:,p], [α/2, 1-α/2]) for p in 1:P]...)')
end

immutable Coef
  mean::Vector{Float64}
  std::Vector{Float64}
  q::Matrix{Float64}
  ne0::Vector{Bool}
  acc::Float64
end

function summary(model::GLM; α::Float64=.05)
  const β = hcat(map(o -> o.β, model.post_params)...)'
  const β_stats = stats(β)
  const ne0 = !(β_stats[3][:,1] .<= 0 .<= β_stats[3][:,2])
  const β_acc = size(unique(β,1),1) / size(β,1)

  const coef = Coef(β_stats[1], β_stats[2], β_stats[3], ne0, β_acc)
  const hyper = map(m -> m.θ, model.post_params)
  return  Summary_glm(coef,hyper)
end

function show(io::IO, coef::Coef)
  const P = length(coef.mean)
  @printf "%5s%10s%10s%10s%10s\n" "" "mean" "std" "CI_lower" "CI_upper"
  for k in 1:P
    @printf "%5s%10.4f%10.4f%10.4f%10.4f%3s\n" string("β",k-1) coef.mean[k] coef.std[k] coef.q[k,1] coef.q[k,2] (coef.ne0[k] ? "*" : "")
  end
end

immutable Summary_glm
  coef::Coef
  hyper::Vector{Hyper}
end

function show(io::IO, hyper::Vector{Hyper})
  for k in keys(hyper[1])
    s = stats(map(h -> h[k], hyper))
    @printf "%5s%10.4f%10.4f%10.4f%10.4f\n" string(k) s[1] s[2] s[3][1] s[3][2]
  end
end

function show(io::IO, sglm::Summary_glm)
  const empty = collect(keys(sglm.hyper[1]))[1] == :empty

  show(io, sglm.coef)
  if !empty
    show(io, sglm.hyper)
  end

  k = collect(keys(sglm.hyper[1]))[1]
  some_hyper = map(h -> h[k], sglm.hyper)
  acc_hyper = length(unique(some_hyper)) / length(some_hyper)

  println()

  @printf "%5s%10.4f\n" "accβ" sglm.coef.acc
  if !empty
    @printf "%5s%10.4f\n" "accθ" acc_hyper
  end
end

#function dic(g::GLM)
#end

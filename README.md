# BayesLM [![Build Status](https://travis-ci.org/luiarthur/BayesLM.jl.svg?branch=master)](https://travis-ci.org/luiarthur/BayesLM.jl)

My Bayesian implementations of typical linear models functions from R, written in julia.

## To Do

- [x] lm
- [x] glm
      - Sampling Distributions
      - [ ] binomial	(link = logit)
      - [ ] Gaussian	(link = identity)
      - [ ] Gamma	(link = inverse)
      - [ ] inverse gaussian	(link = 1/mu^2)
      - [ ] poisson	(link = log)
      - [ ] quasi	(link = identity, variance = constant)
      - [ ] quasi-binomial	(link = logit)
      - [ ] quasi-poisson	(link = log)
- [ ] anova (for model comparing and comparing means)
- [ ] manova
- [ ] survreg
- [ ] summary function for printing out the above model objects
      - [ ] Posterior mean
      - [ ] Posterior st
      - [ ] whether the 90%, 95%, 99% CI for the coefficients contain 0
      - [ ] Posterior 95% CI?
      - [ ] 5 quantiles of residuals
      - [ ] DIC
      - [ ] posterior mean and sd for R^2

## Usage Example


### Linear Regression (LM)
```julia
using BayesLM, Distributions
srand(1);
n = 100
X = randn(n)
b = [0,3]
sig = .5
y = [ones(n) X] * b + randn(n) * sig

@time model = lm(y,X,B=10000);

summary(model)
#= output:
           mean       std ≠0
   β0   -0.0258    0.0523
   β1    3.0582    0.0518  *
    σ    0.5194    0.0378
  DIC    153.56
=#
```

### Logistic Regression (GLM)
```julia
using BayesLM, Distributions
srand(1);
N = 1000
b = [.2, .4, .6, .8]
J = length(b)
X = [ones(N) randn(N,length(b)-1)]

invlogit(xb::Float64) = 1 / (1 + exp(-xb))
y = map(xb -> rand(Bernoulli(invlogit(xb)))*1.0, X*b)

function loglike(y::Vector{Float64}, Xb::Vector{Float64}, θ::Hyper)
  const mu = invlogit.(Xb)
  sum([ logpdf(Bernoulli(mu[i]),y[i]) for i in 1:N ])
end

@time model = glm(y, X, eye(J)*1E-2, loglike, B=2000,burn=10000);

summary(model)
#=
           mean       std  CI_lower  CI_upper
   β0    0.3663    0.0726    0.2309    0.5118  *
   β1    0.3805    0.0754    0.2444    0.5469  *
   β2    0.6759    0.0761    0.5286    0.8301  *
   β3    0.8435    0.0842    0.6857    1.0071  *

 accβ    0.2630
=#
```

### Linear Regression with Normal Likelihood (GLM)
```julia
using BayesLM, Distributions
srand(1);
N = 100
b = [2,3,4,5.]
X = [ones(N) randn(N,length(b)-1)]
σ² = 2
y = X*b + randn(N)*sqrt(σ²)

function loglike(y::Vector{Float64}, mu::Vector{Float64}, θ::Hyper)
  -N/2 * log(2*pi*θ[:σ²]) - sum((y-mu).^2) / (2*θ[:σ²])
end

# Need to define prior for other parameters
# Type alias Hyper = Dict{Symbol, Float}
lp_θ(θ::Hyper) = (-2-1) * log(θ[:σ²]) - 1 / θ[:σ²]

@time model = glm(y, X, eye(Float64,4)*.03, loglike, 
                  Σ_θ=eye(Float64,1)*.1, 
                  θ_bounds=[0. Inf], 
                  θ_names=[:σ²], 
                  θ_logprior=lp_θ);

summary(model)
#= output
           mean       std  CI_lower  CI_upper
   β0    1.8624    0.1416    1.5896    2.1397  *
   β1    2.7697    0.1432    2.4770    3.0375  *
   β2    3.9507    0.1483    3.6604    4.2415  *
   β3    4.9771    0.1345    4.7059    5.2323  *
   σ²    2.1488    0.3136    1.6270    2.8552

 accβ    0.2893
 accθ    0.6826
=#
```

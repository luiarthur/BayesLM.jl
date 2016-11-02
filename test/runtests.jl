println("Loading Packges for BayesLM test...")
using BayesLM
using Base.Test
println("Starting Tests for BayesLM test...")

@testset "lm" begin
  srand(1);
  n = 100
  X = randn(n)
  b = [0,3]
  sig = .5
  y = [ones(n) X] * b + randn(n) * sig

  @time model = lm(y,X,B=10000);

  #= private tests
  using RCall
  @rput y X n b sig;
  R"summary(mod <- lm(y~X))"
  R"plot(X,y)"
  =#

  post_sum = summary(model,alpha=.05)
  println(post_sum)
  @test all(abs(b - post_sum.mean_beta) .< .2)
  @test abs(sig - post_sum.mean_sig) < .1
  @time CPO = cpo(model,y,[ones(n) X])

  #= visual tests
  @rput CPO
  R"plot(y,dnorm(y,cbind(1,X)%*%mod$coef,sig))"
  R"points(y,CPO,col='blue',cex=2)"

  # If I had CPO's from 2 different models, use this visual
  R"CPO2 <- dnorm(y,cbind(1,X)%*%b,sig) # this is not really a CPO, but will be used for demo";
  R"plot(y,CPO-CPO2,bty='n',fg='grey',
         main='CPO_M1 - CPO_M2',ylab='',xlab='Observed y')"
  R"abline(h=0,col='red')"

  # Note that since most of the points lie below 0,
  # M1 may be poorer than M2.
  =#
end

@testset "glm" begin
  srand(1);
  N = 100
  b = [2,3,4,5.]
  X = [ones(N) randn(N,length(b)-1)]
  σ² = 2
  y = X*b + randn(N)*sqrt(σ²)

  function logf(y::Vector{Float64}, mu::Vector{Float64}, θ::Hyper)
    -N/2 * log(2*pi*θ[:σ²]) - sum((y-mu).^2) / (2*θ[:σ²])
  end

  lp_θ(θ::Hyper) = (-2-1) * log(θ[:σ²]) - 1 / θ[:σ²]

  @time model = glm(y, X, eye(Float64,4)*.03, eye(Float64,1)*.1, [0. Inf],
                    [:σ²], identity, logf, lp_θ);

  print(summary(model))
end

@testset "glm logistic" begin
  using Distributions
  srand(1);
  N = 100
  b = [.2,.3,.4,.5]
  X = [ones(N) randn(N,length(b)-1)]

  invlogit(xb::Float64) = 1 / (1 + exp(-xb))
  y = map(xb -> rand(Bernoulli(invlogit(xb)))*1.0, X*b)

  function logf(y::Vector{Float64}, Xb::Vector{Float64}, θ::Hyper)
    const mu = invlogit.(Xb)
    sum([ logpdf(Bernoulli(mu[i]),y[i]) for i in 1:N ])
  end

  lp_θ(θ::Hyper) = 0.0

  @time model = glm(y, X, eye(Float64,4)*1E-3, eye(Float64,1), [0. Inf],
                    [:empty], invlogit, logf, lp_θ);

  print(summary(model))
end

println("Loading Packges for Linear Models test...")
using BayesLM
using Base.Test
println("Starting Tests for Linear Models test...")

@testset "lm" begin
  srand(1)
  n = 100
  X = randn(n)
  b = [0,3]
  y = [ones(n) X] * b + randn(n)

  @time model = lm(y,X);

  #= private tests
  using RCall
  @rput y X n;
  R"summary(lm(y~X))"
  =#

  post_sum = summary(model,alpha=.05)
  println(post_sum)
  @test all(abs(b - post_sum.mean_beta) .< .2)
  @test abs(1 - post_sum.mean_sig) < .1
end


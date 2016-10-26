println("Loading Packges for Linear Models test...")
using LinearModels
using Base.Test
using RCall
println("Starting Tests for Linear Models test...")

@testset "lm" begin
  srand(1)
  n = 100
  X = randn(n)
  b = [0,3]
  y = [ones(n) X] * b + randn(n)

  @time model = lm(y,X);
  println()

  @rput y X n;
  R"mod <- lm(y~X)"
  R"co_R <- coef(mod)"
  R"sig2_R <- sd(mod$residuals) *sqrt( (n-1)/(n-2) )"
  #println(R"summary(mod)")
  @rget co_R sig2_R;
  post_sum = post_summary(model,alpha=.05)
  @test all(abs(co_R - post_sum.mean_beta) .< .01)
  @test abs(sig2_R - post_sum.mean_sig) < .05
end


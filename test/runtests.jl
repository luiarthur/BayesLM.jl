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
  @rput y X n;
  R"summary(lm(y~X))"
  =#

  post_sum = summary(model,alpha=.05)
  println(post_sum)
  @test all(abs(b - post_sum.mean_beta) .< .2)
  @test abs(sig - post_sum.mean_sig) < .1
  @time CPO = cpo(model,[ones(n) X])

  println("RMSE(y,CPO): ", StatsBase.rmsd(y,CPO))

  #=
  for i in 1:n
    @printf "%8.4f%8.4f\n" y[i] CPO[i]
  end
  =#

  #= visual tests
  @rput CPO
  R"plot(X,y,col='grey',pch=20,cex=2)"
  R"points(X,CPO,col='dodgerblue',cex=2,pch=20)"
  R"abline(0,3)"
  R"plot(y,CPO,xlim=range(CPO),ylim=range(CPO))"
  R"abline(0,1)"
  =#
end

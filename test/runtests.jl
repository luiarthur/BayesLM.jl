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

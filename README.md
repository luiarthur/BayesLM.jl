# BayesLM

My Bayesian implementations of typical linear models functions from R, written in julia.

## To Do

- [x] lm
- [ ] glm
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


"""
Baseline-category logit for Multinomial regression. Check out my notes [here](http://nbviewer.jupyter.org/github/luiarthur/GLM_AMS274/blob/master/notes/notes11.ipynb)
"""
function bclogit(Y::Matrix{Int}, X::Matrix{Float64}, cs::Vector{Float64};
             B::Int=2000, burn::Int=10000, printFreq::Int=0)

  const (N,K) = size(X)

  assert( N == size(Y,1) )

  const J = size(Y,2)
  const stepSize_β = [cs[j] * sym(inv(X'X)) for j in 1:J-1]

  function ll(β::Matrix{Float64})
    EXB = exp(X * β)
    pi_denom = 1 + sum(EXB,2) # N by 1
    pi_mat = [ EXB ones(N) ] ./ pi_denom # N by J
    sum( Distributions.logpdf(Distributions.Multinomial(sum(Y[i,:]),pi_mat[i,:]), Y[i,:]) for i in 1:N )
  end

  function update(β::Matrix{Float64})
    newβ = copy(β)

    for j in 1:size(β,2)
      const cand = rand(Distributions.MvNormal(β[:,j], stepSize_β[j]))
      newβ[:,j] = cand
      
      if ll(newβ) - ll(β) < log(rand())
        newβ[:,j] = β[:,j]
      end
   end

   return newβ
  end

  const init = zeros(K,J-1)

  gibbs(init, update, B, burn, printFreq=printFreq)

end

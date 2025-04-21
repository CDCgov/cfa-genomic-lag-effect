# Genomic lag effect

At a high level, our model is
```math
\begin{align*}
&R(t) \sim \mathrm{Pr}(R(t) \mid \boldsymbol{\theta}), r \sim \mathrm{Pr}(r \mid \boldsymbol{\theta}), I_0 \sim \mathrm{Pr}(I_0 \mid \boldsymbol{\theta})\\
&I(t) := g_{\mathrm{renewal}}(R(t), I_0, r, \boldsymbol{\theta}) \\
&P(t) := g_{\mathrm{duration}}(I(t), \boldsymbol{\theta}) \\
&\mathbf{c} \sim \mathrm{Pr}(\mathbf{c} \mid \mathbf{s}, I(t), P(t), \boldsymbol{\theta})
\end{align*}
```
where:

- We sweep other parameters into $\boldsymbol{\theta}$ for readability
- $R(t)$ is the time-varying reproduction number
- $I(t)$ is the incidence, which is a deterministic transformation governed by
  - A deterministic renewal model function $g_{\mathrm{renewal}}$
  - An initial incidence $I_0$
  - An initial exponential growth rate $r$
- $P(t)$ is the prevalence, a deterministic transformation governed by
  - $g_{\mathrm{duration}}$, which stipulates that an individual is infected for a fixed generation interval
- $\mathbf{c}$ are [coalescent times](https://en.wikipedia.org/wiki/Viral_phylodynamics#Coalescent_theory_and_phylodynamics) (which we take to be observed)
- $\mathbf{s}$ are the (known) times at which samples are taken from the population of infecteds

Details about the coalescent likelihood used here, as well as the renewal model, are available in the [model document](model.md).

The pipeline used for a simulation study of the effect of lag on $R_t$ estimation is provided in the [pipeline document](pipeline.md).

There is also an [API reference](api.md) for the python code which implements this model.

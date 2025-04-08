# Model details

For a high-level overview, see [here](index.md).

## Notation

### Time's arrow
Renewal models are typically encountered as forward-time (past to present) models, while coalescent models are backwards-time (present to past) models.
We reserve $t$ for forward time, and $\tau$ for backwards time.
Given the total elapsed time $T$, we can convert from forward time to backward time with $\tau = T - t$.


### Piecewise constant functions
We assume all piecewise-constant functions exist on a regular grid (generally daily), and write either $g(t) := g_{\lfloor t \rfloor}$ or $g(t) := g_{\lceil t \rceil}$ depending on whether the function is left- or right-inclusive.
Note that a left-inclusive function viewed forward in time is a right-inclusive function viewer backward in time, and vice-versa.

## $R(t)$

We model the effective reproduction number $R(t)$ as a piecewise constant function with weekly change points
```math
R(t) = R_{\lfloor t \rfloor}
```
where $\mathbf{R}$ is modelled as a log-scale Gaussian Markov random field
```math
\begin{align*}
\log(R_0) &\sim \mathrm{Normal}(0, \sigma^2) \\
\log(R_t) &\sim \mathrm{Normal}(\log(R_{t - 1}), \sigma^2)
\end{align*}
```
Note that $R_0$ is _not_ the basic reproduction number, it is simply the first value of the effective reproduction number in the piecewise constant function.

## $I(t)$

Renewal models track, forwards in time, incident infections as a function of the history of incidence and reproduction number.
In particular, renewal models work in discrete time steps, with[@cori2013new]
```math
\mathbb{E}[I_t] = R_t \sum_{s=1}^{t} I_{t - s} w_s
```
where $\mathbf{w}$ is a discrete-time infectivity profile and $\sum_s w_s = 1$.
We make the large-population approximation $I_t \approx \mathbb{E}[I_t]$, yielding the deterministic model
```math
I_t = R_t \sum_{s=1}^{t} I_{t - s} w_s
```

In general, $\mathbf{w}$ is estimated separately from applying the renewal model from other sources of data.
As such, deterministic renewal models are a convenient method for linking $R_t$ with incidence without need for many additional (free) model parameters.

## $P(t)$ and the force of infection

To link prevalence with a renewal model's discrete-time incidence, we must make assumptions about the rate of accumulation of incident infections $f_{\mathrm{I}}(t)$ and the duration for which infections last.

We assume that infections last a constant (integer) duration $\gamma$.
Thus, in an interval, new incident infections enter the population at rate $f_\mathrm{I}(t)$ while previous incident infections exit the population at rate $f_\mathrm{I}(t - \gamma)$
```math
\frac{\mathrm{d} P(t)}{\mathrm{d}t} = f_\mathrm{I}(t) - f_\mathrm{I}(t - \gamma)
```

We further assume that the rate of accumulation of incident infections in a day is constant, which makes $P(t)$ piecewise linear and enables piecewise integration.
```math
P(t) = \left( \sum_{s = 0}^{\gamma - 1} I_{\lfloor t \rfloor - s} \right) + \left( t - {\lfloor t \rfloor} \right) \left( f_\mathrm{I}(t) - f_\mathrm{I}(t - \gamma) \right)
```
In particular, $f_{\mathrm{I}}(t) = I_{\lceil t \rceil}$, so
```math
P(t) = \left( \sum_{s = 0}^{\gamma - 1} I_{\lfloor t \rfloor - s} \right) + \left( t - {\lfloor t \rfloor} \right) \left( I_{\lceil t \rceil} - I_{\lceil t \rceil - \gamma} \right)
```

## Linking coalescent models to prevalence and incidence

### Coalescent models in epidemiology
The coalescent[@kingman1982coalescent] models, backwards in time (from present to past), infections in a sample from the population of infecteds.
Each coalescent event represents the common ancestor of two infections, each of which is either a sampled infection or an ancestor of two or more sampled infections.

A coalescent model expresses the probability density of the coalescent events as $\mathrm{Pr}(\mathbf{c} \mid \mathbf{s}, A')$, where $\mathbf{c}$ is the vector of coalesccent times, $\mathbf{s}$ is the vector of sampling times, and $A'$ is the function specifying the rate of change of the number of extant individuals in the sample history.
The function $A'(\tau)$ is the bridge between the coalescent times and the process of infections.
The sampling times $\mathbf{s}$ are conditioned upon, which means both that no explicit information is drawn from them and that the coalescent is "unaware" of vagaries of the sampling process such as the lag from sample collection to the sequence being available.

Coalescent models are Markov models, such that conditional on the state of the process at time $\tau$, events at times greater than $\tau$ are independent of events at times less than $\tau$.
The probability density can thus be factorized conveniently into intervals in which $A(\tau)$ is constant.
There are thus two possibilities for an interval $(\tau, \tau + \Delta\tau]$:
1. A coalescent event occurs and $A(\tau + \Delta \tau) = A(\tau) - 1$. The probability of this is
```math
\text{Pr}(A(\tau + \Delta \tau) = A(\tau) - 1) = {A(\tau) \choose 2} A'(\tau + \Delta \tau) \exp\left[- \int_{s = \tau}^{\tau + \Delta \tau} A'(s) \mathrm{d}s \right]
```
2. A sampling event occurs and $A(\tau + \Delta \tau) = A(\tau) + 1$. The probability of this is
```math
\text{Pr}(A(\tau + \Delta \tau) = A(\tau) - 1) = \exp\left[- \int_{s = \tau}^{\tau + \Delta \tau} A'(s) \mathrm{d}s \right]
```

We can write these jointly by introducing an indicator function $\mathbb{I_C}(\tau)$ which is 1 when there is a coalescent event at time $\tau$ and 0 otherwise.
This yields the interval-based probability density
```math
\mathrm{Pr}((\tau, \tau + \Delta\tau]) = \exp \left[ \left( \mathbb{I_C}(\tau + \Delta \tau) {A(\tau) \choose 2} A'(\tau + \Delta \tau) \right) - \int_{s = \tau}^{\tau + \Delta \tau} A'(s) \mathrm{d}s \right]
```

Frost and Volz (2010)[@frost2010viral] study $A'(\tau)$ for a variety of compartmental models under the large-$P(t)$ approximation (Equation 2.4)
```math
A'(\tau) = \frac{\mathrm{d}A(\tau)}{\mathrm{d}t} = -{A(\tau) \choose 2}\frac{2 f_{\mathrm{I}}(\tau)}{P(\tau)^2}
```
### Renewal coalescent models

We can use our previous assumptions about the incidence-derived piecewise-constant $f_{\mathrm{I}}(\tau)$ and piecewise-linear $P(\tau)$ in concert with Frost and Volz's approximation to obtain
```math
A'(\tau) = \frac{\mathrm{d}A(\tau)}{\mathrm{d}t} = -{A(\tau) \choose 2}\frac{2 I_{\lfloor \tau \rfloor}}{P(\tau)^2}
```
The integral has an analytical solution, but simulating coalescent times from this model is highly nontrivial.

Coalescent models with piecewise-constant $A'$, such as the popular coalescent skygrid model[@gill2013improving], are analytically straightforward and convenient to sample from.
To approximate $A'(\tau)$ as piecewise constant, we employ the daily average value of $P(\tau)^2$.
That is, we define
```math
\tilde{P}(\tau) = \tilde{P}_{\lfloor \tau \rfloor} = \int_{x = \tau}^{\tau + 1} P(x)^2 \mathrm{d}x
```
and make the approximation
```math
A'(\tau) \approx \tilde{A}'(\tau) = -{A(\tau) \choose 2}\frac{2 I_{\lfloor \tau \rfloor}}{\tilde{P}_{\lfloor \tau \rfloor}}
```
Note that as we are using this approximation to both simulate and analyze the simulated data, in a sense the approximation error cancels out.

This approximated $\tilde{A}'(\tau)$ is a piecewise constant function which changes every day and at each sampling and coalescent event.
If we collect all these times in the vector $\mathbf{g}$, we can expand the previous interval-based form of the probability density.
(Note that this vector is sorted in increasing backwards time, such that $g_0 = 0$ is the present day.)
This gives us the log-likelihood
```math
\log \mathrm{Pr}(\mathbf{c} \mid \mathbf{s}, \tilde{A}') = \sum_{k = 1}^{|\mathbf{g}|} \left[ \left( g_k - g_{k - 1} \right) \left( {A_k \choose 2} \frac{2 I_{\lfloor g_k \rfloor}}{\tilde{P}_{\lfloor g_k \rfloor}} \right) + \mathbb{I_C}(g_k) \log \left( {A_k \choose 2} \frac{2 I_{\lfloor g_k \rfloor}}{\tilde{P}_{\lfloor g_k \rfloor}} \right) \right]
```

It is perhaps more convenient to contemplate the average prevalence in forwards time, where
```math
\begin{align*}
\tilde{P}(t) &= \tilde{P}_{\lceil t \rceil} \\
&= \int_{x = t}^{t + 1} P(x)^2 \mathrm{d}s \\
&= \int_{x = t}^{t+ 1} \left[ P(t) + \left( x - t \right) \left( I_{t + 1} - I_{t + 1 - \gamma} \right) \right]^2 \mathrm{d}x \\
&= \frac{\left[ P(t) + \left( x - t \right) \left( I_{t + 1} - I_{t + 1 - \gamma} \right) \right]^3}{3 \left( I_{t + 1} - I_{t + 1 - \gamma} \right)}\Bigg|_{t}^{t+1} \\
&= \frac{\left[ P(t) + \left( I_{t + 1} - I_{t + 1 - \gamma} \right) \right]^3 - P(t)^3}{3 \left( I_{t + 1} - I_{t + 1 - \gamma} \right)}
\end{align*}
```

### Assumptions

We should also grapple briefly with what we are assuming about disease progression.
Coalescent models of the type used here are models of a panmictic, monotypic population.
Panmictic means there is no spatial structure; if we applied this model to the entire US, there would be one US-side pool of infected individuals, and an infection in California is just as likely to share an immediate common ancestor with another from California as one in New York.

\bibliography

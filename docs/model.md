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

## Renewal models

In brief, renewal models use the effective reproduction number $R(t)$ and the generation time distribution to propagate incidence.

### $R(t)$

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

### $I(t)$

In continuous time, renewal models link the current instantaneous incidence rate to $R(t)$ and the history of incidence as [@green2022inferring]
```math
\mathbb{E}[I(t)] = R(t) \int_{s = 0}^\infty I(t - s) \omega(s) \mathrm{d}s
```
where $w(s)$ is the generation time distribution, linking time since infection to infectivity ($\int_s \omega(s) \mathrm{d}s = 1$).

Renewal models are particularly convenient in discrete time, both because observations of emissions from the process are typically binned (such as daily hospitalization counts) and because it enables replacing integrals with summations.
In discrete time, [@cori2013new]
```math
\mathbb{E}[I_t] = R_t \sum_{s=1}^{t} I_{t - s} w_s
```
where $\mathbf{w}$ is the discretized PMF form of $\omega$.

It is also often convenient to make the large-population approximation $I_t \approx \mathbb{E}[I_t]$, yielding the deterministic model
```math
I_t = R_t \sum_{s=1}^{t} I_{t - s} w_s
```

Typically, $\mathbf{w}$ is taken to be known when applying renewal models, having been estimated separately.
As such, deterministic renewal models are a convenient method for linking $R_t$ with incidence without need for many additional (free) model parameters.
Some assumptions are necessary to initialize the process, we assume that at and prior to time 0, the process is in a steady state of constant incidence ($I_0 = I_{-1} = I_{-2} = \dots$), which is a parameter that must be estimated.

### $P(t)$

Renewal models propagate incident infections without touching prevalence.
However, let us assume that the generation time distribution has finite support over a range $[0, \gamma]$.
(That is, $\gamma$ is the largest $s$ such that $\omega(s) > 0$ in continuous time or such that $w_s > 0$ in discrete time.)
Then, $\gamma$ is the duration of infectiousness, and the size of the infectious population at time $t$ is
```math
P(t) = \int_{s = 0}^{\gamma} I(t - s) \mathrm{d}t
```
in continuous time or
```math
P_t = \sum_{s = 0}^{\gamma} I(t - s)
```
in discrete time.

## Epidemiological coalescent models

### Coalescent background

The coalescent[@kingman1982coalescent] models, backwards in time (from present to past), infections in a sample from the population of infecteds.
A coalescent event is the merger of two lineages (each of which has at least one descendant in the sample, or is itself in the sample) into one lineage.

Somewhat more formally, the coalescent is a Markov model which provides a probability distribution on the number of ancestral lineages in a sample at time $A(\tau)$, where time is measured from present, $\tau = 0$, to past, $\tau > 0$.
As a Markov process, it naturally decomposes into a series of independent intervals.
If the vector $\mathbf{c}$ has the coalescent events sorted in increasing order, as sampling events are deterministic we can write the coalescent likelihood as
```math
\mathrm{Pr}(c_i \mid c_{i - 1}) = \frac{A(c_{i - 1}) \choose 2}{\nu(c_i)} \exp \left[ - \int_{\tau = c_{i - 1}}^{c_i} \frac{A(\tau) \choose 2}{\nu(\tau)} \mathrm{d} \tau \right]
```
where $\nu(\tau)$ is a function that controls the (inverse of the) branching rate.

Note that as the process is Markov, we could break this integration at any arbitrary time $c_{i - 1} \leq \delta \leq c_i$ to obtain
```math
\mathrm{Pr}(c_i \mid c_{i - 1}) = \frac{A(c_{i - 1}) \choose 2}{\nu(c_i)} \exp \left[ - \left( \int_{\tau = c_{i - 1}}^{\delta} \frac{A(\tau) \choose 2}{\nu(\tau)} \mathrm{d} \tau \right) - \left( \int_{\tau = \delta}^{c_i} \frac{A(\tau) \choose 2}{\nu(\tau)} \mathrm{d} \tau \right) \right]
```
As such, we can write the coalescent log-likelihood as a sum over any arbitrary set of times $\tau_0, \tau_1, \dots, \tau_G$.
Assuming that all coalescent times are among these times (and thus $G \geq |\mathbf{c}|$), we get
```math
\log \mathrm{Pr}(\mathbf{c} \mid \mathbf{s}, \nu) = \sum_{g = 1}^{G} \left( \mathbb{I_C}(\tau_i)\frac{A(\tau_{i - 1}) \choose 2}{\nu(\tau_i)} + \exp \left[-{A(\tau_{i - 1}) \choose 2} \int_{\tau = \tau_{i - 1}}^{\tau_i} \frac{1}{\nu(\tau)} \mathrm{d} \tau \right] \right)
```
where we are implicitly assuming right-inclusive intervals (hence the number of ancestral lineages in the interval is $A(\tau_{i - 1})$) and that $\tau_0 \leq \min(\min(\mathbf{c}), \min(\mathbf{s}))$.
WLOG we can set $\tau_0 = 0$ because the combinatoric terms are such that the coalescent rate in intervals with $A(\tau) < 2$ is 0.

While there are tractable analytical solutions to this likelihood for some forms of $\nu$, general solutions require numerical integration.
As such, the most widely used coalescent models, such as the skygrid[@gill2013improving], approximate more complex functional forms with piecewise-constant functions.
In the case where $\nu(\tau)$ is constant within the intervals, the likelihood simplifies to
```math
\log \mathrm{Pr}(\mathbf{c} \mid \mathbf{s}, \nu) = \sum_{g = 1}^{G} \left( \mathbb{I_C}(\tau_i)\frac{A(\tau_{i - 1}) \choose 2}{\nu(\tau_i)} + \exp \left[-{A(\tau_{i - 1}) \choose 2} \frac{\tau_i - \tau_{i - 1}}{\nu(\tau_{i - 1})} \right] \right)
```
In addition to the analytically straightforward form of the likelihood, simulating coalescent times piecewise constant models is much more straightforward and efficient than for other functional forms.[@palacios2013gaussian]

### Phylodynamic coalescent models: big picture

Applying the coalescent model to epidemiology is part of the field of [(viral) phylodynamics](https://en.wikipedia.org/wiki/Viral_phylodynamics).
One assumes that lineages represent infections, and as such coalescent events are the backwards-time view of new infections arising.
It is worth noting at this point that the disease history assumed by this model is one in which individuals are _infectious_ for the entirety of the time they are _infected_.
At least one individual involved in a transmission event must be infectious, and the result is one newly infected individual.
As the coalescent assumes exchangeability of all individuals in the population at hand[@wakely2009], we have thus equated the two.

As a model of the history of a sample, the coalescent conditions on the sample taken, rather than drawing information from the sampling times.
This avoids the need to model the sampling process, both in terms of how samples are initially collected and the lag from sample collection to the sequence being available.

A useful framework for linking the coalescent to dynamics of infection is presented by Volz _et al_. (2009)[@volz2009phylodynamics] and Frost and Volz (2010)[@frost2010viral].
The rate of coalescent events among the entire population of infecteds is given by the overall rate of infection, and the rate that is observed in the sample is modulated by the probability that any such coalescent is between lineages with sampled descendants.

### Coalescent models in phylodynamics: math

Let the absolute (not per-capita) rate of new infections arising at time $\tau$ before present ne $f_{\mathrm{I}}(\tau)$.
Then the rate of coalescent events among the entire population of infecteds is $-f_{\mathrm{I}}(\tau)$, as coalescents decrease the number of ancestral lineages backwards in time.
A coalescent event happens between one pair of the $P(\tau)$ infecteds, while we can only see it if both of those are in the $A(\tau)$ ancestral lineages, so the "observed" coalescent rate is
```math
-f_{\mathrm{I}}(\tau) \frac{A(\tau) \choose 2}{P(\tau) \choose 2}
```

The expected rate of change of the number of active lineages is
```math
\frac{\mathrm{d} A}{\mathrm{d} \tau} = - \frac{A(\tau) \choose 2}{\nu(\tau)}
```

By equating these two quantities, and assuming $P(t)$ is sufficiently large that ${P(t) \choose 2} \approx P(t)^2 / 2$ we arrive at
```math
\frac{\mathrm{d} A}{\mathrm{d} \tau} = -\frac{A(\tau) \choose 2}{\nu(\tau)} \approx -2f_{\mathrm{I}}(\tau) \frac{A(\tau) \choose 2}{P(\tau)^2}
```
which we can rearrange to obtain
```math
\nu(\tau) = \frac{P(\tau)^2}{2 f_{\mathrm{I}}(\tau)}
```

### Renewal coalescent models

Renewal models can be used in the Volz _et al._ (2009) framework.
The rate of accumulation of incident infections is $f_{\mathrm{I}}$, and we have seen above how to obtain prevalence from a renewal model.
The renewal model should be started far enough in the past to provide a well-defined prevalence and incidence rate for all coalescent times.

The big challenge in combining a renewal and coalescent model is that the functional forms induced by a renewal model for $\nu(t)$ are arbitrary and preclude analytical forms of the likelihood.
As such, we approximate $\nu(\tau)$ as piecewise constant on a daily scale.
This further enables the use of discrete-time renewal model dynamics.
In particular, we make the convenient approximation
```math
\nu(\tau) \approx \frac{P_{\lceil \tau \rceil}^2}{2 I_{\lceil \tau + 1 \rceil}}
```
that is, we use the discrete-time incidence accumulated during an interval as the rate of new infections and the discrete-time prevalence at the (oldest) end of the interval as the prevalence.

### Violations of the coalescent and their implications

The coalescent is a model for the behavior of exchangeable individuals in a neutrally-evolving panmictic population.
Violations of these assumptions, and a variety of other unmodeled complications, often have the effect of making the coalescent process behave as if the population were smaller.
Thus, often the _effective population size_ is discussed rather than the absolute population size.

When applying renewal coalescent models to populations of infections, we might expect a similar pattern.
The renewal model has overlapping generations and age-dependent reproduction.
The former is not a model violation, but overlapping generations serve to reduce the effective population size by a factor related to the variance in the offspring distribution [@hill1979note].
The latter is a model violation (making individuals non-exchangeable) and, in combination with the former and fluctuations in the environment or population size (both of which we expect in reality), serve to further depress the effective population size[@engen2005effective].
As real pathogens are often evolving in response to selective pressures induced by the human immune system, the effects of which vary by the nature of the selection at hand and pathogen genome architecture with respect to the part of the genome used in phylogenetic inference, but can serve to depress the effective population size[@charlesworth2009effective].

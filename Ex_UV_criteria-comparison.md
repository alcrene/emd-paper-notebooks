---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python (emd-paper)
  language: python
  name: emd-paper
---

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

---
bibliography:
    ../Fitted_model_comparison.bib
math:
    '\mspace': '\hspace{#1}'
    '\Bemd' : 'B_{#1}^{\mathrm{EMD}}'
    '\Bconf': 'B^{\mathrm{conf}}_{#1}'
    '\Bspec': '\mathcal{B}'
    '\EE'   : '\mathbb{E}'
    '\VV'   : '\mathbb{V}'
    '\eE'   : '\mathcal{E}'
    '\logL' : '\mathcal{l}'
    '\nN'   : '\mathcal{N}'
    '\Unif' : '\operatorname{Unif}'
    '\Poisson': '\operatorname{Poisson}'
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Comparison with other criteria

{{ prolog }}

%{{ startpreamble }}
%{{ endpreamble }}

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
from Ex_UV import *
import re
import dataclasses
import matplotlib as mpl
#from myst_nb import glue
from viz import glue
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
hv.extension("matplotlib", "bokeh")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
## logsumexp using natural log ##
# log = np.log
# exp = np.exp
# logsumexp = scipy.special.logsumexp
# base_e_to_x = 1
# base_x_to_10 = base_e_to_10
## logsumexp using pow 2 ##
from functools import reduce
log = np.log2                  # If we use log transforms for numerical stability, log2 makes
exp = lambda x: 2**x           # more sense: faster and more stable
def logsumexp(x, axis=0): assert axis==0; return np.logaddexp2.reduce(x)  # Faster
#def logsumexp(x, axis=0): assert axis==0; return reduce(np.logaddexp2, x)  # Lower memory
base_e_to_x = np.log2(np.exp(1))
base_x_to_10 = np.log10(2)

array = np.array
replace = dataclasses.replace
```

## Notebook parameters

+++ {"editable": true, "slideshow": {"slide_type": ""}}

For the comparison with other criteria, we use variations of the dataset shown in {numref}`fig_UV_setup`:

- We add an intermediate dataset size `L_med`.
- We increase the upper wavelength range so data go into the microwave range.
- We use three different noise levels (with the middle noise closest to the data in {numref}`fig_UV_setup`.
- We use two bias conditions: unbiased, or bias equal to {numref}`fig_UV_setup` / 8.
  (We need to reduce the bias because at large wavelengths the radiance is smaller.)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
L_list = [L_small, L_med, L_large]
Lᑊ = 2**12
s_list = 2.**array([12, 16, 20]) * Bunits**-1  # data_noise_s is about 2**16.5

table_λwindow1 = (20*μm, 1000*μm)
table_λwindow2 = (data_λ_min, data_λ_max)
table_T    = data_T
table_B0   = data_B0/8

dataset_list = [Dataset("Criterion comparison", L, λmin, λmax, s, T=table_T, B0=B0)
                for s in s_list
                for L in L_list
                for (λmin, λmax) in [table_λwindow1, table_λwindow2]
                for B0 in (table_B0, 0*Bunits)]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

To provide a reference, we compute for each dataset the ratio between the variance (controlled by $s$) and the maximum value of the radiance. This gives a sense of how the noise compares with the scale of the data.

The variance is actually $λ$-dependent; to give a single value, we average the expression for the variance given in the paper over $λ$:

$$\text{noise sd} = \sqrt{ \EE_λ \Bigl[ \VV\bigl[\Bspec(λ;T)\bigr]\Bigr]} = \sqrt{\EE_λ \left[ \frac{\Bspec_{\mathrm{P}}(λ;T)}{s}  \right]} $$

We compare this against $\max_λ\Bigl(\Bspec_{\mathrm{P}}(λ;T)\Bigr) = \Bspec_{\mathrm{P}}(λ_{min};T) + \Bspec_0$:

$$\text{noise fraction} := \frac{\text{noise sd}}{\max_λ\Bigl(\Bspec_{\mathrm{P}}(λ;T)\Bigr)} $$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
noise_fraction = {}
for 𝒟 in dataset_list:
    λ, B = 𝒟.get_data()
    Ba = 𝒟.data_model[1](𝒟.L)[1]
    noise_sd = np.sqrt((Ba/𝒟.s).mean())
    noise_fraction[𝒟.B0, 𝒟.s] = noise_sd / (Ba+𝒟.B0).max()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Criteria definitions

+++ {"editable": true, "slideshow": {"slide_type": ""}}

For comparability, we write all criteria as ratios of probabilities, so a criterion $B^C_{AB}$ is understood as

> Model $A$ is $B^C_{AB}$ times more probable than model $B$.

where $C$ can be “model evidence”, “relative likelihood”, etc. Thus, if $P(A)$ is the “probability of model $A$ and $P(B)$ the “probability of model $B$, then a criterion corresponds to
$$B^C_{AB} = \frac{P(A)}{P(B)} \qquad \leftrightarrow \qquad \log B^C_{AB} = \log P(A) - \log P(B) \,.$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
@dataclasses.dataclass
class Criterion:
    lognum  : float
    logdenom: float
    @property
    def ratio(self): return exp(self.lognum - self.logdenom)
    @property
    def logratio(self): return (self.lognum - self.logdenom)*base_x_to_10
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Criteria are often presented this way because the notion of “probability of a model” is ill-defined, and the quantities $P(A)$ and $P(B)$ often diverge or go to zero – making the ratio the only quantity with some degree of stability. Of course, ratios of the form $\tfrac{0}{0}$ or $\tfrac{\inf}{\inf}$ are known to lead to all kinds of pathologies, which is what we want to illustrate with this section.

The $\Bemd{}$ attempts to resolve this by defining a proper probability which does not require a ratio to obtain a finite number:
$$\Bemd{} = P(R_A < R_B) \,.$$
We consider this a better assessment quantity, less prone to over/underconfidence than a ratio with diverging denominator. However, since other criteria have no equivalent form, for the purpose of comparability, we will convert $\Bemd{}$ into a ratio-like quantity $\underline{B}^{\mathrm{EMD}}$ (we use $\approx^*$ to denote “conceptual equivalence”, rather a strict mathematical equality):
$$\begin{align}
\Bemd{} &\approx^* \frac{P(A)}{P(A) + P(B)} \\
\therefore \; \underline{B}^{\mathrm{EMD}} &\approx^* \frac{P(A)}{P(B)} \approx^* \frac{\Bemd{}}{1 - \Bemd{}} \\
\log \underline{B}^{\mathrm{EMD}} &\approx^* \log {\Bemd{}} - \log \bigl(1 - \Bemd{}\bigr) \,.
\end{align}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{note}
:class: margin
$\Bemd{}$ does not require specifying priors on the parameters or reserving test data.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@memory.cache
def Bemd(𝒟):
    #c = 2**-1 if 𝒟.λmax == 30*μm else 2**0
    c = c_chosen
    _mixed_ppf, _synth_ppf = get_ppfs(𝒟)
    _Bemd = emd.Bemd(_mixed_ppf["Planck"], _mixed_ppf["Rayleigh-Jeans"],
                     _synth_ppf["Planck"], _synth_ppf["Rayleigh-Jeans"],
                     c=c, res=8, M=128,
                     progbarA=None, progbarB=None)
    return Criterion(log(_Bemd), log(1-_Bemd))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

| Symbol | Variable | Meaning |
|--------|----------|---------|
|$L$     | `L`  | Number of data samples used to fit the model. |
|$L'$    | `Lᑊ`      | Number of *new* data samples used only to test the model |
|$\mathcal{D}$ | `𝒟` | Dataset |
|$\hat{σ}$, $\hat{T}$ | `σˆ`, `Tˆ` | Maximum likelihood parameters |

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$\logL$: Log likelihood function
~ Recall that the likelihood is a function of model parameters, so we can write
  $$\logL_a(σ, T)  := \sum_{i=1}^L -Q_{a}(\Bspec_i \mid λ_i, σ, T) \,,$$
  since we defined the loss $Q$ to be the negative log probability.

  To assign a likelihood to a model, some criteria use $\max_{σ, T} \logL_a(σ, T)$; i.e. they evaluate the at the fitted parameters. In the following we denote this $\logL_a(\hat{σ}_a, \hat{T}_a)$.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def lₐ(a, σ, T, 𝒟):
    return -Q(a, σ=array(σ).reshape(-1,1), T=array(T).reshape(-1,1)
             )(𝒟.get_data()).sum().squeeze() * base_e_to_x
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
def lₐ(a, σ, T, 𝒟):  # This version can deal with larger datasets, by splitting the data into manageable chunks
    λ, ℬ = 𝒟.get_data()
    Δ = 2**12
    return sum(
        -Q(a, σ=array(σ).reshape(-1,1), T=array(T).reshape(-1,1)
          )((λ[i*Δ:(i+1)*Δ], ℬ[i*Δ:(i+1)*Δ])).sum().squeeze() * base_e_to_x
        for i in range(int(np.ceil(len(λ)/Δ)))
    )
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{note}
:class: margin

When we check the fitted temperatures, they are all in the range 3200-4100 K. So these are reasonable datasets
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
σˆ = {}; Tˆ = {}
def f(log2σ_T, a, 𝒟): σ, T = 2**log2σ_T; return -lₐ(a, σ, T, 𝒟)
class res:
    success = False
for 𝒟 in dataset_list:
    for a in ["Rayleigh-Jeans", "Planck"]:
        _𝒟 = replace(𝒟, L=L_med)  # Always use a reasonable number of samples to fit σ and T => Too large can fail, and too small adds a different source variability than what we want to compare
        s0 = np.sqrt(_𝒟.get_data()[1].mean()/𝒟.s).m  # Initial s guess with moment matching
        res = optimize.minimize(f, np.log2([s0, 4000]), (a, _𝒟), tol=1e-3); #assert res.success
        if not res.success: print(np.log2(D.s.m))
        σˆ[(a,𝒟)] = 2**res.x[0]; Tˆ[(a,𝒟)] = 2**res.x[1]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$R$: Expected risk
~ In our paper we treat the expected risk as a random variable, and the EMD distribution is a way to induce a distribution on $R$. In this section however we mean the expected risk in the usual sense, where it is a scalar obtained by averaging the loss on test samples:
  $$R_a := \frac{1}{L'} \sum_{j=1}^{L'} Q_{a}(\Bspec_j \mid λ_j, \hat{σ}_{a}, \hat{T}_a) \approx -\frac{1}{L} \logL_a(\hat{σ}_{a}, \hat{T}_a) \,.$$
  Note that the log likelihood is evaluated on *training* samples, while the expected risk is evaluated on *test* samples. That said, because the models are very simple and we can make $L$ and $L'$ as large as we want, the difference between the two sides of this equation should be negligeable.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@cache
def R(a,𝒟):
    return Q(a, σ=σˆ[a,𝒟], T=Tˆ[a,𝒟])(replace(𝒟, purpose="test", L=Lᑊ).get_data()).mean() * base_e_to_x
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$\eE$: Model evidence
~ The Bayesian model evidence is obtained by integerating the posterior. For this we need to set priors on the parameters $T$ and $σ$; a common practice is to use “uninformative priors”, to “let the data speak”.{cite:p}`gelmanHolesBayesianStatistics2021` (In other words, we choose broad priors which let the posterior concentrate near the likelihood.) For this example we use the following priors:
  $$\begin{aligned}
    π\Bigl(\log \frac{σ}{[σ]}\Big) &\sim \Unif(\log 2^9, \log 2^{14})   &&\quad& π_2\Bigl(\log \frac{σ}{[σ]}\Big) &\sim \Unif(\log 2^9, \log 2^{10})\\
    π\Bigl(\log \frac{T}{[T]}\Bigr) &\sim \Unif(\log 1000, \log 5000) &&\quad& π_2\Bigl(\log \frac{T}{[T]}\Bigr) &\sim \Unif(\log 3900, \log 4100)\\
    π(λ) &\sim \Unif(20 \mathrm{μm}, 30 \mathrm{μm})
  \end{aligned}$$
  (The “prior” over $λ$ reflects that we sample the wavelengths at regular intervals betwen $20 \mathrm{μm}$ and $30 \mathrm{μm}$. Thanks to the choice of a uniform distribution it drops out from the calculations.) For simplicity and to emulate modelling errors, we don’t define a prior over the bias $\Bspec_0$: we take all candidate models to be *unbiased*.
  :::{admonition} Sensitivity to the choice of prior
  :class: note dropdown
  The use of uninformative priors is well motivated in the context of *inference*, since with enough data the posterior becomes insensitive to the choice of prior. This is **not** the case however for the computation of the evidence, where sensitivity to the prior is strong for all dataset sizes $L$. Thus we could get numerically different results with different choices of the prior. However at least in this example, simply tightening the priors (i.e. using $π_2$ above) results in numerical differences on the order of 1%.
  :::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
# To use joblib.Memory, we need objects which can be hashed & pickled consistently
# Stats distributions don’t do this, but we can store their arguments instead
@dataclasses.dataclass(frozen=True)
class Prior:
    distname: str
    args: tuple
    rng: int|str|tuple[int|str]
    @property
    def rv(self):
        rv = getattr(stats, self.distname)(*self.args)
        rv.random_state = utils.get_rng(self.rng)
        return rv
    @property
    def rvs(self): return self.rv.rvs
    @property
    def pdf(self): return self.rv.pdf
    @property
    def logpdf(self): return self.rv.logpdf
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
π1logσ = Prior("uniform", (log(2**9), log(2**14)), "prior σ")
π2logσ = Prior("uniform", (log(2**9), log(2**10)), "prior σ")
π1logT = Prior("uniform", (log(1000), log(5000)), "prior T")
π2logT = Prior("uniform", (log(3900), log(4100)), "prior T")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

  With these choices, the *evidence* for model $a$ can be written
  $$\eE_a = \iint \! dσ dT \, π(σ) π(T) \; p\bigl(\{λ_i, \Bspec_i\}_{i=1}^L \mid a, λ, σ, T\bigr)  \,,$$
  with $p(\{λ_i, \Bspec_i\}_{i=1}^L \mid a, σ, T)$ given by the Gaussian observation model and $π(λ_i) = \frac{1}{10}$:
  $$\begin{aligned}
  p(\{λ_i, \Bspec_i\}_{i=1}^L \mid a, σ, T) &= \prod_{i=1}^L \frac{1}{\sqrt{2π}σ} \exp\left(-\frac{\bigl(\Bspec_i-\hat{\Bspec}_a(λ;T)\bigr)^2}{2σ^2} \right) π(λ_i) \,, \\
  &= \exp\left[-\frac{L}{2} \log(2πσ^2) - \sum_{i=1}^L \frac{\bigl(\Bspec_i-\hat{\Bspec}_a(λ;T)\bigr)^2}{2σ^2} + L\log \frac{1}{10} \right]\,, \\
  &= \exp\left[ \logL_a\Bigl(σ, T\Bigr) + L \log \frac{1}{10} \right] \,.
  \end{aligned}$$
  The evidence in this example is therefore computed with a double integral, which can be done directly with `scipy.dblquad` if $L$ is small. When $L$ is large however, the probability is too small and we run into numerical underflow; in that case we can resort to evaluating the integral by generating a large number $M$ of prior samples:
  $$\begin{aligned}
  \eE_a &\approx \sum_j p(\{λ_i, \Bspec_i\}_{i=1}^L \mid a,  σ_j, T_j) \qquad\text{where}\quad (σ_j, T_j) \sim  π(σ) π(T) \,, \\
  &\approx \left(\frac{1}{10}\right)^L \sum_j \exp\left[ \ll_a\Bigl(σ_j, T_j\Bigr) \right] \\
  \log \eE_a &\approx -L\log10 + \log\left[ \sum_j \exp \biggl(\ll_a\Bigl(σ_j, T_j\Bigr) \biggr)\right] \,.
  \end{aligned}$$

We can then replace the probabilities with their logs and use `logsumexp` to get
$$\log \eE_a &\approx -L\log10 + \mathtt{scipy.special.logsumexp}\left(\ll_a^{(1)}, \ll_a^{(2)}, \dotsc \right)\mathtt{.mean()}\,,$$
where $\ll_a^{(j)} = \ll_a(σ_j, T_j)$, $(σ_j, T_j) \sim  π(σ) π(T)$ and $j = 1, 2, \dotsc, L_{\eE}$.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@memory.cache
def logℰ(a, 𝒟, πlogσ, πlogT):
    Lℰ = 2**14   # Number of Monte Carlo samples for integral. Use giant number so result is effectively exact
    rng = utils.get_rng("uv", "evidence")
    return logsumexp([lₐ(a, σj, Tj, 𝒟)  # NB: We omit the `-L log 10`, which evenually drops out
                      for σj, Tj in zip(exp(πlogσ.rvs(Lℰ, random_state=rng)), exp(πlogT.rvs(Lℰ, random_state=rng)))
                     ])
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$\mathrm{elpd}$: Expected log pointwise predictive density, WAIC, LOO
~ The expected log pointwise predictive density ($\mathrm{elpd}$) on unseen data is often considered a gold standard when it comes to comparing models. Directly estimating the elpd requires putting aside a large amount of data for testing, which is rarely feasible; hence approximations have been developed, like the widely applicable information criterion (WAIC) and leave-one-out (LOO) cross-validation.({cite:p}`vehtariPracticalBayesianModel2017`) In this example however we can generate data at will, so there is no need for approximations: we can compute the $\mathrm{elpd}$ directly.

  In the following we use $λ_i$, $\Bspec_i$ to denote the original data points used to fit the model, and $λ_j'$, $\Bspec_j'$ (with $j \in \{1,\dotsc,L'\}$) to denote the *new* data points on which we evaluate the $\mathrm{elpd}$. The posterior over parameters is denoted $p^*(T, σ \mid \{λ_i, \Bspec_i\}_{i=1}^L\bigr)$.
  $$\begin{aligned}
    \mathrm{elpd}_a
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log p\bigl(λ_j, \Bspec_j' \mid \{λ_i, \Bspec_i, a\}\bigr) \\
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log \Bigl[
        \iint \!dTdσ \; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, p^*\bigl(T, σ \mid \{λ_i, \Bspec_i\}, a\bigr)
        \Bigr] \\
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log \left[
        \iint \!dTdσ\; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, \frac{p\bigl(\{λ_i, \Bspec_i\} \mid T, σ, a \bigr)\, π(T)π(σ)}{\int \!dTdσ\; p\bigl(\{λ_i, \Bspec_i\} \mid T, σ, a \bigr)\, π(T)π(σ)}
        \right] \\
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log \left[
        \iint \!dTdσ\; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, \frac{e^{\ll_a(σ, T)} \, π(λ)π(T)π(σ)}{\eE_a}
        \right] \qquad \left(π(λ) \equiv \frac{1}{10}\right) \\
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log \left[
        \iint \!dTdσ\; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, e^{\ll_a(σ, T)} \, π(T)π(σ)
        \right] - \log \eE_a - \log 10
  \end{aligned}$$
  The integrand involves only probabilities of single samples, and so can be computed directly with `scipy.dblquad`.

  :::{margin}
  As it turns out, even though direct integration is feasible, it is still about 100x slower than averaging over samples and using `logaddexp`.
  :::
  Since $π(T)$ and $π(σ)$ are uniform in log space, we can further simplify the integral by a change of variables:

  $$\begin{aligned}
  \mathrm{elpd}_a
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log \Biggl[
        \iint \!dTdσ\; \exp\Bigl( \underbrace{\log p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr)}_{\eqqcolon\, -Q_a(λ_j, \Bspec_j; σ, T)} + \ll_a(σ, T) \Bigr) \, π(T)π(σ)
        \Biggr] - \log \eE_a - \log 10 \\
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log \Biggl[
       \underbrace{π(\log T)}_{=(\log 5)^{-1}} \; \underbrace{π(\log σ)}_{=(\log 16)^{-1}} \; \int_{\log 1000}^{\log 5000}\mspace{-48mu} d(\log T) \int_{\log 2}^{\log 32}\mspace{-32mu} d(\log σ)\; \exp\Bigl( \ll_a(e^{\log σ}, e^{\log T}) - Q_a(λ_j, \Bspec_j; e^{\log σ}, e^{\log T})\Bigr)
        \Biggr] - \log \eE_a - \log 10 \\
    &\approx \frac{1}{L'}\sum_{j=1}^{L'} \log \Biggl[
       \int_{\log 1000}^{\log 5000}\mspace{-48mu} d(\log T) \int_{\log 2}^{\log 32}\mspace{-32mu} d(\log σ)\; \underbrace{\exp\Bigl( \ll_a(e^{\log σ}, e^{\log T}) - Q_a(λ_j, \Bspec_j; e^{\log σ}, e^{\log T})\Bigr)}_{\eqqcolon h(\log σ, \log T)}
        \Biggr] - \log \eE_a - \log 10 - \log\log 5 - \log \log 16
\end{aligned}$$

~ The $\mathrm{elpd}$ is closely related to the expected risk $R$; in fact, if we define $Q$ to be the log *posterior* instead of the log *likelihood*, it becomes equivalent.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@memory.cache
def elpd(a, 𝒟, πlogσ, πlogT):
    Lℰ = 2**14   # Number of Monte Carlo samples for integral. Use giant number so result is effectively exact
    rng = utils.get_rng("uv", "elpd")
    def h(σ, T, λ_ℬ, a=a, 𝒟=𝒟): return lₐ(a, σ, T, 𝒟) - Q(a, σ, T)(λ_ℬ)*base_e_to_x
    σarr = exp(πlogσ.rvs(Lℰ, random_state=rng))
    Tarr = exp(πlogT.rvs(Lℰ, random_state=rng))
    λ_test, ℬ_test = replace(𝒟, L=Lᑊ, purpose="test").get_data()
    return logsumexp(h(σarr, Tarr, (λ_test[:,None], ℬ_test[:,None])), axis=0
                     ).mean() - logℰ(a, 𝒟, πlogσ, πlogT)  # NB: We omit constant terms
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{dropdown} Slower version using `dblquad`
```python
def elpd(a, L):
    dblquad = scipy.integrate.dblquad
    def h(logσ, logT, λ_ℬ, a=a, L=L):
        return exp( lₐ(a, L, σ:=exp(logσ), T:=exp(logT)) - Q(a, σ, T)(λ_ℬ) )
    λ_test, ℬ_test = replace(𝒟, L=Lᑊ, purpose="test").get_data()
    return sum( log( dblquad(h, log(1000), log(5000), log(2), log(32), [(λj, ℬj)]) )
                for λj, ℬj in tqdm(zip(λ_test, ℬ_test), total=Lᑊ)) / Lᑊ \
           - logℰ(a, L)  # We omit the constant terms, which cancel out in the ratio
```
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$B^l$: Relative likelihood / AIC
~ Computing the ratio of likelihoods is equivalent to the difference of *log* likelihoods:
  $$\log B^l = \logL_{\mathrm{P}}(\hat{σ}_{\mathrm{P}}, \hat{T}_{\mathrm{P}}) - \logL_{\mathrm{RL}}(\hat{σ}_{\mathrm{P}}, \hat{T}_{\mathrm{P}}) \,.$$
~ The Akaike information criterion (AIC) for a model with $k$ parameters reads
  $$AIC_a = 2k - 2\logL_a(\hat{σ}_a, \hat{T}_a) \,.$$
  Since the Rayleigh-Jeans and Planck models both have the same number of parameters, this is equivalent to the likelihood ratio (up to a factor of 2).

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@cache
def Bl(𝒟): return Criterion(
    lₐ("Planck",        σˆ[("Planck",𝒟)],         Tˆ[("Planck",𝒟)],         𝒟),
    lₐ("Rayleigh-Jeans", σˆ[("Rayleigh-Jeans",𝒟)], Tˆ[("Rayleigh-Jeans",𝒟)], 𝒟)
)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$B^{\mathrm{Bayes}}$: Bayes factor
~ The Bayes factor is the ratio of the evidence for each model:
  $$\begin{aligned}
  B^{\mathrm{Bayes}} &= \frac{\eE_{\mathrm{P}}}{\eE_{\mathrm{RL}}} \,, \\
  \log B^{\mathrm{Bayes}} &= \log \eE_{\mathrm{P}} - \log \eE_{\mathrm{RL}}
  \end{aligned}$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def BBayes(𝒟, πlogσ, πlogT): return Criterion(logℰ("Planck", 𝒟, πlogσ, πlogT),
                                              logℰ("Rayleigh-Jeans", 𝒟, πlogσ, πlogT))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$B^{\mathrm{elpd}}$: $\mathrm{elpd}$ criterion
~ The $\mathrm{elpd}$ is usually reported as-is, but since it does scale like a log probability, we can make it comparable to other criteria by defining

  $$\begin{aligned}
  \log B^{\mathrm{elpd}}
  &\coloneqq \mathrm{elpd}_{\mathrm{P}} - \mathrm{elpd}_{\mathrm{RL}} \\
  &= - \log \eE_{\mathrm{P}} + \log \eE_{\mathrm{RL}} \\
  &\quad \begin{aligned}
      \;+\, \frac{1}{L'}\sum_{j=1}^{L'} \Bigg\{
      &\log \left[
      \iint \!dTdσ\; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, \ll_a(σ, T) \, π(T)π(σ)
      \right] \\
      &- \log \left[
       \iint \!dTdσ\; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, \ll_a(σ, T) \, π(T)π(σ)
      \right] \Biggr\}
    \end{aligned}
  \end{aligned}$$

  In practice, a large positive value for $\log B^{\mathrm{elpd}}_{AB}$ would be interpreted as strong evidence for model $A$.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def Belpd(𝒟, πlogσ, πlogT): return Criterion(elpd("Planck", 𝒟, πlogσ, πlogT),
                                             elpd("Rayleigh-Jeans", 𝒟, πlogσ, πlogT))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$B^R$: Ratio of expected risk
~ As with the $\mathrm{elpd}$, the expected risk is more commonly given directly. However since we chose $Q$ to be the negative log likelihood, it is reasonable to present it as a ratio to make it comparable with other criteria:
  :::{margin}
  Signs are flipped because our criteria are interpreted as ratios of probabilities (i.e. negative loss).
  :::
  $$\log B^R = -R_{\mathrm{P}} + R_{\mathrm{RJ}}$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def BR(𝒟): return Criterion(-R("Planck", 𝒟), -R("Rayleigh-Jeans", 𝒟))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Comparison table

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
df = pd.DataFrame({(f"{𝒟.λmin.m}–{𝒟.λmax.m}", 𝒟.L, 𝒟.B0.m, 𝒟.s.m, noise_fraction[𝒟.B0, 𝒟.s].m):
                   {r"$\underline{B}^{\mathrm{EMD}}_{\mathrm{P,RJ}}$": Bemd(𝒟).logratio,
                    r"$B^R_{\mathrm{P,RJ}}$": BR(𝒟).logratio,
                    r"$B^l_{\mathrm{P,RJ}}$": Bl(𝒟).logratio,
                    r"$B^{\mathrm{Bayes}}_{\mathrm{P,RJ};π}$": BBayes(𝒟, π1logσ, π1logT).logratio,
                    #r"$B^{\mathrm{Bayes}}_{\mathrm{P,RJ};π_2}$": BBayes(𝒟, π2logσ, π2logT).logratio,
                    r"$B^{\mathrm{elpd}}_{\mathrm{P,RJ};π}$": Belpd(𝒟, π1logσ, π1logT).logratio,
                    #r"$B^{\mathrm{elpd}}_{\mathrm{P,RJ};π_2}$": Belpd(𝒟, π2logσ, π2logT).logratio,
                    }
                   for 𝒟 in tqdm(dataset_list)})
df.columns.names=["λ", "L", "B0", "s", "rel. σ"]
df.index.name = "Criterion"

df = df.stack(["λ", "L"])
# Put biased data before unbiased, to match figure
df = df.sort_index(axis="columns", ascending=[False, True])
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Undo the sorting along λ, so the longer wavelengths case comes first (same order as the figure)
_index = []
for i in range(0, len(df.index), 6):
    _index.extend(df.index[i+3:i+6])
    _index.extend(df.index[i:i+3])
df = df.loc[_index]
# Make a MultiIndex so the heading shows at the top instead of to the left
#df.columns = pd.MultiIndex.from_product([["noise $(s^{-1})$"], df.columns.values])
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
# Colorscheme for shading the table. Values will be transformed with log10;
# scheme goes from 0.01 (-2) to 100 (+2);
# See https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html#matplotlib.colors.LinearSegmentedColormap
# and https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html
shade_planck = config.figures.colors["pale"]["cyan"]
shade_rj = config.figures.colors["pale"]["red"]

cdict = {"red": [], "green": [], "blue": [], "alpha": []}
r,g,b = mpl.colors.to_rgb(shade_rj)  # Strong evidence for Rayleigh-Jeans
cdict["red"].append((0, r, r)); cdict["green"].append((0, g, g)); cdict["blue"].append((0, b, b))
cdict["alpha"].append((0, 1.0, 1.0))
r,g,b = 1,1,1  # Center
cdict["red"].append((0.5, r, r)); cdict["green"].append((0.5, g, g)); cdict["blue"].append((0.5, b, b))
cdict["alpha"].append((0.5, 0.0, 0.0))
r,g,b = mpl.colors.to_rgb(shade_planck)  # Strong evidence for Planck
cdict["red"].append((1, r, r)); cdict["green"].append((1, g, g)); cdict["blue"].append((1, b, b))
cdict["alpha"].append((1, 1.0, 1.0))

table_cmap = mpl.colors.LinearSegmentedColormap("rb_diverging", cdict)
del r,g,b
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# Format values B near 0 with 2 decimals
# B ⩾ 10000 : scientific notation (1 sig digit)
# B < 0.01  : scientific notation (1 sig digit)
def format_B_value(B: float, format):
    if B == 0:            return "0"
    elif abs(B) >= 10000: return viz.format_scientific(B, sig_digits=1, format=format)
    elif abs(B) < 0.01:   return viz.format_scientific(B, sig_digits=1, format=format)
    else:                 return f"{B:.2f}"
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The colour scheme was picked to saturate at ratios of 30:1, which is around where [common practice](https://www.statlect.com/fundamentals-of-statistics/Jeffreys-scale) would interpret the evidence of a Bayes factor to be “strong” or “very strong”.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
df = df \
    .rename_axis(["Criterion", r"$λ (\mathrm{μm})$", "$L$"], axis="index") \
    .rename_axis(["", "$s$", "rel. $σ$"], axis="columns")
# Common options
df_styled_html = df.copy().style \
    .background_gradient(table_cmap, axis=0, vmin=-1.5, vmax=+1.5)  # log₁₀ 30 ≈ 1.47
df_styled_latex = df.copy().style \
    .background_gradient(table_cmap, axis=0, vmin=-1.5, vmax=+1.5)  # log₁₀ 30 ≈ 1.47
# HTML options
df_styled_html = df_styled_html \
    .format_index(lambda L: viz.format_pow2(L), axis="index", level=2, escape="latex") \
    .format_index(lambda s: viz.format_pow2(s), axis="columns", level=1, escape="html") \
    .format_index(lambda nf: f"{nf*100:2.0f}%", axis="columns", level=2) \
    .format_index(lambda B0: "$\mathcal{B}_0 > 0$" if B0>0 else "$\mathcal{B}_0 = 0$", axis="columns", level=0, escape=None) \
    .format(partial(format_B_value, format="unicode")) \
    .set_table_styles([{"selector": "td", "props": "padding: 0.5em;"},
                       {"selector": "th", "props": "text-align: center; border-bottom: 1px black"}])
df_styled_html
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
label = "tbl_uv_criteria-comparison"
short_caption = "Comparison of different model selection criteria."
# '@' is a hack to prevent our own preprocessors from substituting \cref with {numref}
caption = r"""
\textbf{Comparison of different model selection criteria} for variations of the dataset shown in
\@cref{fig_UV_setup}. Criteria compare the Planck model ($\MP$) against the 
Rayleigh-Jeans model ($\MRJ$) and are evaluated for different dataset sizes
($L$), different levels of noise ($s$) and different wavelength windows ($λ$); $L$, $s$ and $λ$ values were chosen to span the
transition from weak/ambiguous evidence for either $\MP$ or $\MRJ$,
to reasonably strong evidence for $\MP$.
To give a better sense of scale, the noise level is reported in two ways: $s$ is the actual value used to
generate the data,
while "rel.\ $σ$" reports the resulting standard deviation as a fraction of the maximum radiance within
the data window.
The 15 to 30 $\mathrm{\mu m}$ window for $λ$ corresponds to the data that shown in \@cref{fig_UV_setup},
while the {{λmicrowave_low}} to {{λmicrowave_high}} $\mathrm{\mu m}$ window stretches further into the microwave range, where the two models are nearly indistinguishable.
As in \@cref{fig_uv-example_r-distributions}, we perform calculations under both positive
and null bias conditions (resp.\ ($\mathcal{B}_0 > 0$ and $\mathcal{B}_0 = 0$);
the former emulates a situation where neither model can fit the data perfectly.
To allow for comparisons, all criteria are presented as $\log_{10}$ ratios of probabilities,
even though for non-Bayesian criteria this is not the usual form.
Positive (negative) values indicate evidence in favour of the Planck (Rayleigh-Jeans) model:
for example, a value of -1 is interpreted as $\MP$ being 10 times less likely than $\MRJ$,
while a value of +2 would suggest that $\MP$ is 100 times \emph{more} likely than $\MRJ$.
Expressions for all criteria are given in \@cref{app_expressions-other-criteria}.
For the $\underline{B}^{\mathrm{EMD}}_{\mathrm{P,RJ}}$ criteria, we used $c={{c_chosen}}$.
Note that especially for small $L$, some values are sensitive to the random seed used to generate the data.
""".replace("@", "") \
.replace("{{λmicrowave_low}}", str(table_λwindow1[0].m)) \
.replace("{{λmicrowave_high}}", str(table_λwindow1[1].m)) \
.replace("{{c_chosen}}", str(c_chosen))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
def tex2md(text):
    text = re.sub(r"\\emph{([^}]*)}", r"*\1*", text)
    text = re.sub(r"\\textbf{([^}]*)}", r"**\1**", text)
    text = re.sub(r"\\cref{([^}]*)}", r"{numref}`\1`", text)
    return text
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

Jupyter book has poor support for long table captions.
The solution used below is to put the caption in a sidebar.
Biggest disadvantage is that this doesn’t get the standard table link and formatting.
(Possibly with lots of extra work we could emulate that ?)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
with open(config.paths.figures/"table_criterion_comparison.html", 'w') as f:
    html_code = viz.jb_fixes.display_dataframe_with_math(df_styled_html)
    f.write("`````{only} html\n")
    f.write("````{margin}\n")         # Open margin
    # Workaround to get a numbered table: Put an empty list-table in the margin
    f.write(f"({label})=\n")          # Anchor for cross-references
    f.write(f":::{{list-table}} &nbsp;\n" "* - &nbsp;\n:::\n") # Empty table
    f.write("````\n")                 # Close margin
    # Workaround to associate caption & table: put them in a card
    f.write(":::{card} \n")           # Open card
    f.write(tex2md(caption).strip() + "\n")  # Add caption
    f.write("```{raw} html\n" + html_code._repr_html_() + "```\n")  # Table content
    f.write(":::\n")                  # Close card
    f.write("`````\n")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# LaTeX options
(df_styled_latex
    .format_index(lambda B0: "$\mathcal{B}_0 > 0$" if B0>0 else "$\mathcal{B}_0 = 0$",
                  escape=None, axis="columns", level=0)
    .format_index(lambda s: viz.format_pow2(s, format='$latex$'),
                  escape="latex", axis="columns", level=1)
    .format_index(lambda nf: f"{nf*100:2.0f}\%",
                  escape="latex", axis="columns", level=2) \
    .format_index(lambda L: viz.format_pow2(L, format='$latex$'),
                  escape="latex", axis="index", level=2) \
    .format_index(escape=None, axis="index")
    .format(partial(format_B_value, format='siunitx'))
);
# # Escape the # in the index name (escape='latex' only applies to index values, not headers)
# names = list(df_styled_latex.index.names)
# names[2] = r'\# samples ($L$)'
# df_styled_latex.index.names = names
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
with open(config.paths.figures/"table_criterion_comparison.latex", 'w') as f:
    latex_code = df_styled_latex.to_latex(
            label=label, caption=(caption, short_caption),
            environment="table*",
            column_format="ccrSSSSSS", multirow_align="c", multicol_align="c",
            hrules=True, clines="skip-last;data",
            sparse_index=True, sparse_columns=True, siunitx=True,
            convert_css=True,
        )
    # There is no column equivalent to clines, so we add the \cmidrules ourselves
    i = latex_code.index("\\\\") + 3 # Position after the first header (i.e. the bias header)
    latex_code = latex_code[:i] + "\\cmidrule(rl){4-6} \\cmidrule(rl){7-9}\n" + latex_code[i:]
    # Remove the last two clines: they are superfluous with bottomrule
    for m in list(re.finditer(r"\\cline{.*?}", latex_code))[-1::][::-1]:
        latex_code = latex_code[:m.start()] + latex_code[m.end():]
    # Replace clines with cmidrules (probably a bug: if we already use booktabs, cmidrule is much better)
    latex_code = latex_code.replace("cline", "cmidrule")
    f.write("```{raw} latex\n")
    f.write(latex_code)
    f.write("```")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

## Exported notebook variables

These can be inserted into other pages.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
glue("c_chosen", f"${viz.format_pow2(c_chosen, 'latex')}$", display=True)

glue("πσdesc", r"$σ$ follows a log uniform distribution between "
               f"2^{{{int(π1logσ.args[0])}}} and 2^{{{int(π1logσ.args[1])}}}")
glue("πTdesc", r"$T$ follows a log uniform between "
               f"{exp(π1logT.args[0]).round().astype(int)} "
               f"and {exp(π1logT.args[1]).round().astype(int)}")

#glue("sunits") defined in Ex_UV
glue("table_s1", viz.format_pow2(s_list[0].m, format='latex'), raw_myst=True, raw_latex=True)
glue("table_s2", viz.format_pow2(s_list[1].m, format='latex'), raw_myst=True, raw_latex=True)
glue("table_s3", viz.format_pow2(s_list[2].m, format='latex'), raw_myst=True, raw_latex=True)
glue("table_L_list", f"${viz.format_pow2(L_list[0], format='latex')}$, "
                     f"${viz.format_pow2(L_list[1], format='latex')}$, and "
                     f"${viz.format_pow2(L_list[2], format='latex')}$",
     raw_myst=True, raw_latex=True)
glue("table_λrange1", f"{table_λwindow1[0].m}–{table_λwindow1[1].m} ${table_λwindow1[0].units:~L}$",
     raw_myst=True, raw_latex=True)
glue("table_λrange2", f"{table_λwindow2[0].m}–{table_λwindow2[1].m} ${table_λwindow2[0].units:~L}$",
     raw_myst=True, raw_latex=True)
# glue("table_λmin1", **viz.formatted_quantity(table_λwindow1[0], precision=0))
# glue("table_λmax1", **viz.formatted_quantity(table_λwindow1[1], precision=0))
# glue("table_λmin2", **viz.formatted_quantity(table_λwindow2[0], precision=0))
# glue("table_λmax2", **viz.formatted_quantity(table_λwindow2[1], precision=0))
glue("table_T", **viz.formatted_quantity(table_T))
glue("table_B0_mag", data_B0.m)

glue("color_RJ", color_labels.RJ, display=True)
glue("color_Planck", color_labels.Planck, display=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
emd.utils.GitSHA()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---

```

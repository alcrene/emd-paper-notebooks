# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md:myst
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# ---
# bibliography:
#     ../Fitted_model_comparison.bib
# math:
#     '\mspace': '\hspace{#1}'
#     '\Bemd' : 'B_{#1}^{\mathrm{EMD}}'
#     '\Bconf': 'B^{\mathrm{conf}}_{#1}'
#     '\Bspec': '\mathcal{B}'
#     '\EE'   : '\mathbb{E}'
#     '\VV'   : '\mathbb{V}'
#     '\eE'   : '\mathcal{E}'
#     '\logL' : '\mathcal{l}'
#     '\nN'   : '\mathcal{N}'
#     '\Unif' : '\operatorname{Unif}'
#     '\Poisson': '\operatorname{Poisson}'
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Implementations of common model comparison criteria
#
# {{ prolog }}
#
# %{{ startpreamble }}
# %{{ endpreamble }}

# %% [markdown] editable=true slideshow={"slide_type": ""}
# This file collects definitions for other popular model comparison criteria.
#
# Devising a criterion involves casting the model selection problem in a way that can be replaced by a model score; the higher/lower the score, the more favoured a model is. A criterion is then an estimator for that score.
#
# It’s important to remember that not all criteria start from the same score, and therefore they are not all completely exchangeable. In particular, $\Bemd{}$ is the only criterion which tries to estimate the uncertainty of its score.

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
import re
import dataclasses
import logging
import multiprocessing
from functools import cached_property, cache, partial
from typing import Literal
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# %% editable=true slideshow={"slide_type": ""}
from joblib import Memory
import psutil

memory = Memory(".joblib-cache", verbose=0)
ncores = min(psutil.cpu_count(logical=False), 64)
# ncores = max(2, min(psutil.cpu_count(logical=False)-4, 100))

# %% editable=true slideshow={"slide_type": ""}
import numpy as np
from scipy import stats
import dynesty
import xarray
import arviz

import emdcmp as emd

import matplotlib as mpl

import utils

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# Many of the functions below use logarithms in intermediate calculations to improve numerics.
# The particular basis is not really important, since it cancels out in the end; in the code below, we denote the basis of the calculation logarithm as $x$. It may be $e$ (more nature for analytics) or 2 (more natural/efficient for numerics).
#
# Uncomment the appropriate block of code below to select between either base $e$ or base 2 for the calculation logarithm. This defines `log`, `exp`, `logsumexp`, `base_e_to_2` and `base_2_to_10` to use the appropriate “x” basis. The latter two are defined as
#
# $$\begin{aligned}
# \mathtt{base\_e\_to\_x} &= \log_x e \\
# \mathtt{base\_x\_to\_10} &= \log_{10} x
# \end{aligned}$$
#
# and can be used to convert between the log basis used for calculations and the one used and reporting, using the log transformation rule:
#
# $$\log_a y = {\log_b y} \bigm/ {\log_b a} \,.$$

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
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
base_e_to_2 = np.log2(np.exp(1))
base_2_to_10 = np.log10(2)

array = np.array
replace = dataclasses.replace


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Criteria definitions

# %% [markdown] editable=true slideshow={"slide_type": ""}
# For comparability, we write all criteria as ratios of probabilities, so a criterion $B^C_{AB}$ is understood as
#
# > Model $A$ is $B^C_{AB}$ times more probable than model $B$.
#
# where $C$ can be “model evidence”, “relative likelihood”, etc. Thus, if $P(A)$ is the “probability of model $A$ and $P(B)$ the “probability of model $B$, then a criterion corresponds to
# $$B^C_{AB} = \frac{P(A)}{P(B)} \qquad \leftrightarrow \qquad \log B^C_{AB} = \log P(A) - \log P(B) \,.$$ (eq_conceptual-probability-ratio)

# %% [markdown]
# :::{margin}
# `Criterion` class takes two arguments – `lognum` and `logdenom` –  which correspond respectively to $\log P(A)$ and $\log P(B)$ in {eq}`eq_conceptual-probability-ratio`.
# __Values must be provided in base $e$.__ (I.e. we assume natural logarithms.)
# A `Criterion` object has two attributes, `ratio` and `logratio`, the latter of which is __in base 10__.
# :::

# %% editable=true slideshow={"slide_type": ""}
@dataclasses.dataclass
class Criterion:
    lognum  : float
    logdenom: float
    @property
    def ratio(self): return exp(self.lognum - self.logdenom)
    @property
    def logratio(self): return (self.lognum - self.logdenom)*base_2_to_10


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Criteria are often presented this way because the notion of “probability of a model” is ill-defined, and the quantities $P(A)$ and $P(B)$ often diverge or go to zero – making the ratio the only quantity with some degree of stability. Of course, ratios of the form $\tfrac{0}{0}$ or $\tfrac{\inf}{\inf}$ are known to lead to all kinds of pathologies, which is one way to understand why many criteria work better in theory than in practice.
#
# The $\Bemd{}$ attempts to resolve this by defining a proper probability which does not require a ratio to obtain a finite number:
# $$\Bemd{} = P(R_A < R_B) \,.$$
# We consider this a better assessment quantity, less prone to over/underconfidence than a ratio with diverging denominator. However, since other criteria have no equivalent form, for the purpose of comparability, we will convert $\Bemd{}$ into a ratio-like quantity $\underline{B}^{\mathrm{EMD}}$ (we use $\approx^*$ to denote “conceptual equivalence”, rather a strict mathematical equality):
# $$\begin{align}
# \Bemd{} &\approx^* \frac{P(A)}{P(A) + P(B)} \\
# \therefore \; \underline{B}^{\mathrm{EMD}} &\approx^* \frac{P(A)}{P(B)} \approx^* \frac{\Bemd{}}{1 - \Bemd{}} \\
# \log \underline{B}^{\mathrm{EMD}} &\approx^* \log {\Bemd{}} - \log \bigl(1 - \Bemd{}\bigr) \,.
# \end{align}$$

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{note}
# :class: margin
# $\Bemd{}$ does not require specifying priors on the parameters or reserving test data.
# :::

# %%
def get_ppfs(𝓜, 𝒟, rng=None, L_synth=2**14):
    mixed_ppf = emd.make_empirical_risk_ppf(𝓜.Q(𝒟.get_data()))
    synth_ppf = emd.make_empirical_risk_ppf(𝓜.Q(𝓜.gen_data(L_synth, λmin=𝒟.λmin, λmax=𝒟.λmax, rng=rng)))
    return mixed_ppf, synth_ppf


# %% editable=true slideshow={"slide_type": ""}
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


# %% [markdown] editable=true slideshow={"slide_type": ""}
# | Symbol | Variable | Meaning |
# |--------|----------|---------|
# |$L$     | `L`  | Number of data samples used to fit the model. |
# |$L'$    | `Lᑊ`      | Number of *new* data samples used only to test the model |
# |$\mathcal{D}$ | `𝒟` | Dataset |
# |$\hat{Θ}$ | `θˆ` | Maximum likelihood parameters |
# |$\logL(Θ)$ | `𝓁(Θ)` | Log likelihood function |

# %% [markdown] editable=true slideshow={"slide_type": ""}
# $\logL$: Log likelihood function
# ~ Recall that the likelihood is a function of model parameters, so we can write
#   $$\logL_a(Θ)  := \sum_{i=1}^L -Q_{a}(y_i \mid x_i, Θ) \,,$$
#   since we defined the loss $Q$ to be the negative log probability.
#
#   To assign a likelihood to a model, some criteria use $\max_{Θ} \logL_a(Θ)$; i.e. they evaluate the at the fitted parameters. In the following we denote this $\logL_a(\hat{Θ})$.

# %% editable=true slideshow={"slide_type": ""}
def 𝓁(Θ, Q, 𝒟):
    return -Q[Θ](𝒟.get_data()).sum().squeeze()


# %% editable=true slideshow={"slide_type": ""}
def 𝓁(Θ, Q, 𝒟):  # This version can deal with larger datasets, by splitting the data into manageable chunks
    x, y = 𝒟.get_data()
    w = 2**12
    return sum(
        -Q[Θ]((x[i*w:(i+1)*w], y[i*w:(i+1)*w])).sum().squeeze()
        for i in range(int(np.ceil(len(x)/w)))
    )


# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{note}
# :class: margin
#
# When we check the fitted temperatures, they are all in the range 3200-4100 K. So these are reasonable datasets
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# $R$: Risk
# ~ The (true) risk is a scalar obtained by averaging the loss on _test_ samples:
#   $$\begin{aligned}
#   R_a &:= \lim_{L' \to \infty} \frac{1}{L'} \sum_{j=1}^{L'} Q_a(y_j \mid x_j) & \qquad \text{where}\quad (y_j, x_j) &\sim \mathcal{M}_{\mathrm{true}} \\
#       & \,= -\frac{1}{L'} \logL_a\Bigl(\{x_j, y_j\}_{j=1}^{L'}\Bigr) \,.
#   \end{aligned}$$
#   The $R$-distributions computed in our paper approximate the uncertainty on $R_Q$ due to modelling errors.  
#   Note although the true risk is technically an expectation over the true data distribution, we get a very good estimate by using an empirical average because:
#   - we can generate test data by using a different random seed from the training data;
#   - the models are very simple, and therefore we can make $L'$ very large.

# %% editable=true slideshow={"slide_type": ""}
@cache
def R(Q, Θ, 𝒟, Lᑊ=2**12, purpose="expected risk"):
    """Compute the expected risk by averaging it over _new_ data samples.
    Change the value of `purpose` to change the RNG seed used to generate new samples."""
    return Q[Θ](replace(𝒟, purpose=purpose, L=Lᑊ).get_data()).mean() * base_e_to_2


# %% [markdown]
# $π$: Prior
# ~ We assume the prior factorizes into an independent distribution for each variable:
#   $$π(Θ) = π_A(θ_A) π_B(θ_B) \dotsb$$
# ~ The `FactorizedPrior` class comes equipped with an `expect` method.
#   This computes a Monte Carlo estimate of $\int f(Θ) dπ(Θ)$: samples are drawn from the prior, and we track both their mean and variance.
# ~ Experience has shown that alredy for 2d priors, this is much more efficient than computing the marginal with direct integration with `dblquad`.
# ~ The variance is used to compute the standard error on the mean. Once this is below a certain threshold, we return the mean. A threshold of $2^{-6} \approx 1.5\%$ runs fairly fast; computation time becomes noticeable for thresholds $2^{-n}$ with $n > 6$, and increases exponentially in $n$.
# ~ The number of samples is increased doubled each time until the standard error threshold is reached. We use the [_Parallel algorithm_](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm) to update the estimated mean and variance.
# ~ __Important__ `expect` can average any function $f$ over the prior, but that function must be provided as $\log_2 f$ (i.e. it must return the _logarithm base 2_ the function we actually want to average).
#   This allows us to use `logaddexp2` to minimize rounding errors. The use of base 2 instead of $e$ is not only much more computationally efficient, but also reduces rounding errors.

# %% editable=true slideshow={"slide_type": ""}
# FactorizedPrior
# To use joblib.Memory, we need objects which can be hashed & pickled consistently
# Stats distributions don’t do this, but we can store their arguments instead
@dataclasses.dataclass(frozen=True)
class FactorizedPrior:
    """
    Container for a factorized prior.
    Each factor is given by an independent 1d distribution from scipy.stats.
    """
    distnames: list[str]
    args: list[tuple]
    rng: int|str|tuple[int|str]

    def __and__(self, other: 'FactorizedPrior'):
        return FactorizedPrior(self.distnames + other.distnames,
                               self.args + other.args,
                               self.rng + other.rng)
    
    @cached_property  # cached property avoids resetting the RNG to the same value
    def rv_list(self):
        _rv_list = []
        for i, (distname, _args) in enumerate(zip(self.distnames, self.args)):
            rv = getattr(stats, distname)(*_args)
            rv.random_state = utils.get_rng(self.rng, i)
            _rv_list.append(rv)
        return _rv_list
    def rvs(self, size=1, random_state=None): return np.stack([rv.rvs(size=size, random_state=random_state)
                                                               for rv in self.rv_list]).T
    def pdf(self, x): return np.prod([rv.pdf(xi)
                                      for xi, rv in zip(np.atleast_2d(x).T, self.rv_list)], axis=0)
    def logpdf(self, x): return np.sum([rv.logpdf(xi)
                                        for xi, rv in zip(np.atleast_2d(x).T, self.rv_list)], axis=0)
    @property
    def prior_transform(self):
        def prior_transform(u):
            return np.array([rv.ppf(ui) for ui, rv in zip(u, self.rv_list)])
        return prior_transform
    def _avg_M2(self, log2f, args=(), rtol=2**-6, Lπ_min=2**10, num_doublings=6):
        """
        Return an estimate of the mean and of the sum of squared differences of f under the prior.
        For numerical reasons we take a function which returns the _logarithm_ of f in base 2.
        """
        def one_block(Lπ):
            log2farr = np.fromiter((log2f(Θ, *args) for Θ in self.rvs(size=Lπ)),
                                   dtype=float, count=Lπ)
            log2farr.sort()                        # Minimize rounding errors by sorting values before summing
            μ = 2**np.logaddexp2.reduce(log2farr) / Lπ  # Faster but requires more memory than reduce(np.logaddexp2, log2farr)
            M2arr = (2**log2farr - μ)**2           # NB: using logM2arr could run into issues for points where the difference is near zero
            M2arr.sort()                           # Idem: minimize rounding errors
            return μ, M2arr.sum()

        Lπ = Lπ_min
        μ, M2 = one_block(Lπ)
        stderr = np.sqrt(M2) / Lπ
        if stderr < rtol * μ:
            return μ, M2, Lπ

        # Keep doubling the number of samples until the relative error is below the tolerance
        for _ in range(num_doublings):
            μB, M2B = one_block(Lπ)
            δ = μ - μB
            μ = (μ + μB)/2
            M2 += M2B + δ**2 * Lπ/2
            Lπ *= 2
            
            stderr = np.sqrt(M2) / Lπ
            if stderr < rtol * μ:
                break
        else:
            logger.warning("`Prior.expect` did not achieve the target relative accuracy:\n"
                           f"Std err is {stderr/μ*100:.3}% of mean, but it should be less than {rtol*100:.3}%.")

        #print("# prior samples:", Lπ)
        #print("       std. err:", stderr)
        return μ, M2, Lπ

    def expect(self, log2f, args=()):
        """
        Return the expectation of f under the prior.
        
        .. Caution:: For numerical reasons we take a function which returns the
           _logarithm_ of f in base 2.
        """
        μ, M2, Lπ = self._avg_M2(log2f, args)
        return μ

    def variance(self, log2f, args=()):
        """
        Return an estimator for the variance of f under the posterior.
        
        .. Caution:: For numerical reasons we take a function which returns the
           _logarithm_ of f in base 2.
        """
        μ, M2, Lπ = self._avg_M2(log2f, args)
        return M2/Lπ


# %% [markdown] editable=true slideshow={"slide_type": ""}
# $\eE$: Model evidence
# ~ The Bayesian model evidence is obtained by marginizaling the likelihood over the prior.
#   $$\eE_a = \int\!dπ(Θ)\, \logL_a(Θ) = \EE_π\bigl[ \logL_a(Θ) \bigr] \,.$$

# %% [markdown] editable=true slideshow={"slide_type": ""}
# An initial attempt to compute evidence by Monte Carlo fails, because even in this simple example, the likelihood is too peaked: none (or not enough) of the samples are drawn in the high-probability region.
# ```python
#     @memory.cache
#     def logℰ(Q, π, 𝒟):
#         return np.log(π.expect(partial(𝓁, Q=Q, 𝒟=𝒟)))
# ```
# Instead we use [dynesty](https://dynesty.readthedocs.io), a package specialized for computed the model evidence using slice sampling. Slice sampling goes through a series of refinements, to locate the mode(s) of the posterior and focus the samples on those regions.

# %% editable=true slideshow={"slide_type": ""}
@memory.cache(ignore=["ncores"])
def sample_posterior(Q, π, 𝒟, dlogz=0.01, nlive=1024, sample='rwalk', enlarge=1.25, bootstrap=0, ncores=ncores):
    """
    Sample the posterior using slice sampling (specifically with Dynesty).
    The resulting samples can be used for evaluating both the model evidence
    and, after uniform resampling, more usual Bayesian statistics like WAIC and posterior variance.
    
    Optional arguments are passed on to `dynesty.NestedSampler`; the defaults define
    a more robust and accurate sampler than the dynesty defaults. This allows it to
    accommodate a more complex posterior, at the cost of longer convergence times.

    .. Hint:: The dynesty documentation recommends that `nlive` be an integer multiple
       of the number of cores.
    """
    with dynesty.pool.Pool(ncores, 𝓁, π.prior_transform) as pool:
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform,
                                        logl_kwargs={'Q':Q, 'D':𝒟},
                                        ndim=π.rvs().size, pool=pool,
                                        nlive=nlive, sample=sample, enlarge=enlarge, bootstrap=bootstrap)
        sampler.run_nested(print_progress=False, dlogz=dlogz)
            # Within a tqdm loop, the default print_func spams the console (it prints below the old progress, instead of on top of it)
            #   Possible solution: set up our own tqdm progbar, and pass it as the `pbar` argument to dynesty.results.print_fn
        return sampler.results


# %% editable=true slideshow={"slide_type": ""}
def logℰ(Q, π, 𝒟) -> tuple[float, float]:
    """
    Returns estimate of the log evidence and the standard error on that estimate,
    as computed by `dynesty` during sampling.

    The standard de
    """
    dynres = sample_posterior(Q, π, 𝒟)
    return dynres.logz[-1], dynres.logzerr[-1]


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ::::{margin}
# :::{note}
# Dynesty documents `logzerr` both as the approximate error on `logz`, but also as its standard deviation.
# AFAICT they actually mean its standard *error*; at least `logzerr` is the quantity they use for the "+/-" uncertainty in the progress display, so clearly this is how they interpret it.
#
# Possibly confusing at first is that the sequence of `logzerr` is monotonically _increasing_.
# This is because it is estimated as the cumulative uncertainty (on the log z) for the contribution to the evidence from each slice. Once a slice is “frozen”, we no longer add samples to it, and therefore the uncertainty on its total evidence does not decrease.
# :::
# ::::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# $\mathrm{elpd}$: Expected log pointwise predictive density, WAIC, LOO
# ~ The expected log pointwise predictive density ($\mathrm{elpd}$) on unseen data is often considered a gold standard when it comes to comparing models. Directly estimating the elpd requires putting aside a large amount of data for testing, which is rarely feasible; hence approximations have been developed, like the widely applicable information criterion (WAIC) and leave-one-out (LOO) cross-validation {cite:p}`vehtariPracticalBayesianModel2017`. In this example however we can generate data at will, so there is no need for approximations: we can compute the $\mathrm{elpd}$ directly.
#
# ~ In the following we use $\{x_i, y_i\} := \{x_i, y_i\}_{i=1}^L$ to denote the original data points used to fit the model, and $x_j'$, $y_j'$ (with $j \in \{1,\dotsc,L'\}$) to denote the *new* data points on which we evaluate the $\mathrm{elpd}$.
# ~ The prior and posterior over parameters are denoted respectively $π(Θ)$ and $p(Θ \mid \{x_i, y_i\})$.
# ~ The likelihood and posterior are denoted respectively $\logL(Θ) \stackrel{\mathrm{def}}{=} p(\{x_i, y_i\} | Θ)$ and $p(x_j', y_j' | \{x_i, y_i\}, Θ)$.
#
# ~ The $\mathrm{elpd}$ is closely related to the expected risk $R$; in fact, if we define $Q$ to be the log *posterior* instead of the log *likelihood*, it becomes equivalent.
#   $$\begin{aligned}
#   p(Θ | \{x_i, y_i\})
#       &= \frac{p(\{x_i, y_i\} | Θ) π(Θ)}{p(\{x_i, y_i\})} \\
#       &= α' π(Θ) \, p(\{x_i, y_i\} | Θ) && \quad\text{where} & α' &:= \frac{1}{p(\{x_i, y_i\})} \\
#   p\bigl(x_j', y_j' | \{x_i, y_i\}\bigr)
#       &= \int\!dΘ\, p\bigl(x_j', y_j' | Θ\bigr)\, p(Θ | \{x_i, y_i\})  \\
#       &= α' \int\!dπ(Θ)\, p\bigl(x_j', y_j' | Θ\bigr) \hphantom{p(x_j)} \; p(\{x_i, y_i\} | Θ)  \\
#       &= α' \int\!dπ(Θ)\, p\bigl(y_j' | x_j', Θ\bigr) p(x_j) \; \, p(\{y_i\} | \{x_i\}, Θ) p(\{x_i\}) \\ 
#       &= α \int\!dπ(Θ)\, p\bigl(y_j' | x_j', Θ\bigr)  \; p(\{y_i\} | \{x_i\}, Θ) && \quad\text{where} & α &:= \frac{p(x_j)p(\{x_i\}) }{p(\{x_i, y_i\})}  \\
#   \mathrm{elpd}_a &\approx \frac{1}{L'} \sum_{j=1}^{L'} \log p\bigl(x_j', y_j' | \{x_i, y_i\}\bigr) \\
#       &= \frac{1}{L'} \sum_{j=1}^{L'} \log \Biggl[ α \int\!dπ(Θ)\, p\bigl(y_j' | x_j' Θ\bigr)\, p(\{y_i\} | \{x_i\} Θ) \Biggr] \\
#       &= \frac{1}{L'} \sum_{j=1}^{L'} \log \Bigl[ π.\mathtt{expect}\Bigl[\log_2 p\bigl(y_j' | x_j', Θ\bigr) + \log_2 p(\{y_i\} | \{x_i\}, Θ)\Bigr] \Bigr] + \frac{\log α}{L'} && \mathrlap{\Bigl( π.\mathtt{expect} \text{ assumes log-transformed argument } \Bigr)} \\
#       &= \frac{1}{L'} \sum_{j=1}^{L'} \log \Bigl[ π.\mathtt{expect}\Bigl[\log p\bigl(y_j' | x_j', Θ\bigr) + \underbrace{\log p(\{y_i\} | \{x_i\}, Θ)}_{\logL(Θ]} \bigm/ \log 2 \Bigr] \Bigr] + \frac{\log α}{L'}
#   \end{aligned}$$
#

# %% [markdown] editable=true slideshow={"slide_type": ""}
# $B^l$: Relative likelihood / AIC
# ~ Computing the ratio of likelihoods is equivalent to the difference of *log* likelihoods:
#   $$\log B^l = \logL_{\mathrm{P}}(\hat{σ}_{\mathrm{P}}, \hat{T}_{\mathrm{P}}) - \logL_{\mathrm{RL}}(\hat{σ}_{\mathrm{P}}, \hat{T}_{\mathrm{P}}) \,.$$
# ~ The Akaike information criterion (AIC) for a model with $k$ parameters reads
#   $$AIC_a = 2k - 2\logL_a(\hat{σ}_a, \hat{T}_a) \,.$$
#   Since the Rayleigh-Jeans and Planck models both have the same number of parameters, this is equivalent to the likelihood ratio (up to a factor of 2).

# %%
# NB: We can’t just use len(Θ) because Θ may contain lists
def num_params(Θ): return sum(1 for _ in utils.flatten(Θ))


# %% editable=true slideshow={"slide_type": ""}
def AIC(Q, Θ, 𝒟)   : return -2*𝓁(Θ, Q, 𝒟) + 2*num_params(Θ)


# %% editable=true slideshow={"slide_type": ""}
def BIC(Q, Θ, 𝒟)   : return -2*𝓁(Θ, Q, 𝒟) + np.log(𝒟.L)*num_params(Θ)


# %% editable=true slideshow={"slide_type": ""}
#@memory.cache
def DIC(Q, Θ, π, 𝒟):
    #return -2*𝓁(Θ, Q, 𝒟) + 4*π.variance(partial(𝓁, Q=Q, 𝒟=𝒟))
    samples = sample_posterior(Q, π, 𝒟)
    weights = samples.importance_weights()
    var = (samples.logl**2).dot(weights) - samples.logl.dot(weights)**2
    return -2*𝓁(Θ, Q, 𝒟) + 4*var


# %% [markdown] editable=true slideshow={"slide_type": ""}
# WAIC and PSIS-LOO-CV assume uniformly drawn MCMC samples. We _could_ use an MCMC sampler like *pymc* or *emcee*, but since we already run *dynesty*’s slice sampler to compute the model evidence, this seems wasteful.
# Instead we subsample the log likelihood values returned by *dynesty* according to their probability, thus emulating what a uniform MCMC sampler would return.
#
# For the calculations themselves, we use the implementations [ArviZ.waic](https://python.arviz.org/en/stable/api/generated/arviz.waic.html) or [ArviZ.loo](https://python.arviz.org/en/stable/api/generated/arviz.loo.html). At least for PSIS-LOO-CV, this is the recommendation of that method’s [official implementation repo](https://github.com/avehtari/PSIS).
#
# ::::{margin}
# :::{note}
# For this simple model, it seems that both WAIC and PSIS-LOO-CV work well (no warning is raised) and give nearly identical results.
# :::
# ::::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{admonition} Difference with *dynesty*’s `resample_equal` function
# :class: note dropdown
#
# *dynesty* provides the `resample_equal` function which also generates an MCMC-like set of uniform samples. The difference is that `resample_equal` samples with replacement to create a new dataset of the same size as the original: so it generates a bigger dataset, but with repeated values.
# I found that the estimated number of parameters using subsampling was more plausible (1.4 instead of 2.3), which is why I use this approach.
# Ultimately this choice does not really matter, because the important values (elpd and its standard error) are effectively the same for both methods.
# :::

# %% editable=true slideshow={"slide_type": ""}
@memory.cache
def elpd(Q, π, 𝒟, purpose="elpd",
         scale: Literal["log", "negative_log", "deviance"]="log",
         method: Literal["waic", "loo"]="waic"
        ) -> tuple[float, float]:
    """
    Return the elpd, estimated using either WAIC or PSIS-LOO-CV.
    Returns:
        estimated elpd
        standard error
    """
    rng = utils.get_rng(purpose)
    if method not in {"waic", "loo"}: raise ValueError(f"`method` must be either 'waic' or 'loo'. Received {method}")

    dynresult = sample_posterior(Q, π, 𝒟)
    select = rng.binomial(1, np.exp(dynresult.logl - dynresult.logl.max())).astype(bool)
    θsamples = dynresult.samples[select]
    #θsamples = dynresult.samples_equal()

    # Massage the log likelihood data into the form expected by ArviZ.
    # Note that we want pointwise log likelihoods, so neither can we use the values in dynresult.logl nor our own log likelihood function 𝓁
    logL_dataset = xarray.Dataset(
        {"logL": (["chain", "draw", "B"], [[-Q[θ](𝒟.get_data()) for θ in θsamples]])},
        coords = {"chain": (["chain"], [0]),
                  "draw": (["draw"], np.arange(len(θsamples))),
                  "B": (["B"], np.arange(𝒟.L)),
                 }
    )
    idata = arviz.InferenceData(log_likelihood=logL_dataset)

    if method == "waic":
        elpd_res = arviz.waic(idata, pointwise=True, scale=scale)
        return elpd_res.elpd_waic, elpd_res.se
    else:
        weights = np.exp(logL)
        weights /= weights.sum()
        reff = 1/(weights**2).sum() / len(weights)
        elpd_res = arviz.loo(idata, pointwise=True, reff=reff, scale=scale)
        return elpd_res.elpd_loo, elpd_res.se


# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{dropdown} Old implementation using direct expectation
#
# The idea here was to compute the elpd “directly”, to avoid possible confounds from the approximations inherent in other methods like WAIC or PSIS-LOO-CV.
# Unfortunately direct evaluation of the integral is still too slow, even for this simple problem, and the Monte Carlo implementation below doesn’t draw enough samples from the high-probability region to produce an accurate estimate.
#
# ```python
# # Functions must be defined in module scope in order to be pickleable
# def h(Θ, xy, Q, 𝒟): return (𝓁(Θ, Q, 𝒟) - Q[Θ](xy)) * base_e_to_2
# def lpd(xy, π, h): return π.expect(partial(h, xy=xy))
#
# @memory.cache
# def elpd(Q, π, 𝒟, Lᑊ=16, purpose="elpd"):
#     # Collecting multiple param draws before summing them allows to sort and reduce rounding errors
#     #Lπ = 1024
#     
#     𝒟test = replace(𝒟, L=Lᑊ, purpose=f"{𝒟.purpose} - {purpose}")
#     #xytest = 𝒟test.get_data()
#
#     #def h(Θ, xy): return (𝓁(Θ, Q, 𝒟) - Q[Θ](xy)) * base_e_to_2
#     #def lpd(xy): return π.expect(partial(h, xy=xy))
#     _lpd = partial(lpd, π=π, h=partial(h, Q=Q, D=𝒟))
#
#     args = tqdm(zip(*𝒟test.get_data()), desc="elpd test sample", total=Lᑊ)
#     with multiprocessing.Pool(ncores) as pool:
#         lpd_arr = np.fromiter(pool.imap_unordered(_lpd, args, chunksize=max(1, Lᑊ//8)),
#                               dtype=float, count=Lᑊ)
#     
#     #lpd_arr = np.zeros(Lᑊ, dtype=float)
#     #for j, (xj, yj) in tqdm(enumerate(zip(*𝒟test.get_data())), desc="elpd test sample"):
#     #    lpd_arr[j] = π.expect(partial(h, xy=(xj, yj)))
#
#     lpd_arr.sort()
#     return np.log(lpd_arr).mean()
# ```
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# $B^{\mathrm{Bayes}}$: Bayes factor
# ~ The Bayes factor is the ratio of the evidence for each model:
#   $$\begin{aligned}
#   B^{\mathrm{Bayes}} &= \frac{\eE_{\mathrm{P}}}{\eE_{\mathrm{RL}}} \,, \\
#   \log B^{\mathrm{Bayes}} &= \log \eE_{\mathrm{P}} - \log \eE_{\mathrm{RL}}
#   \end{aligned}$$

# %% editable=true slideshow={"slide_type": ""}
def BBayes(𝒟, πlogσ, πlogT): return Criterion(logℰ("Planck", 𝒟, πlogσ, πlogT),
                                              logℰ("Rayleigh-Jeans", 𝒟, πlogσ, πlogT))


# %% [markdown] editable=true slideshow={"slide_type": ""}
# $B^{\mathrm{elpd}}$: $\mathrm{elpd}$ criterion
# ~ The $\mathrm{elpd}$ is usually reported as-is, but since it does scale like a log probability, we can make it comparable to other criteria by defining
#
#   $$\begin{aligned}
#   \log B^{\mathrm{elpd}}
#   &\coloneqq \mathrm{elpd}_{\mathrm{P}} - \mathrm{elpd}_{\mathrm{RL}} \\
#   &= - \log \eE_{\mathrm{P}} + \log \eE_{\mathrm{RL}} \\
#   &\quad \begin{aligned}
#       \;+\, \frac{1}{L'}\sum_{j=1}^{L'} \Bigg\{
#       &\log \left[
#       \iint \!dTdσ\; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, \logL_a(σ, T) \, π(T)π(σ)
#       \right] \\
#       &- \log \left[
#        \iint \!dTdσ\; p\bigl(λ_j, \Bspec_j' \mid T, σ, a\bigr) \, \logL_a(σ, T) \, π(T)π(σ)
#       \right] \Biggr\}
#     \end{aligned}
#   \end{aligned}$$
#
#   In practice, a large positive value for $\log B^{\mathrm{elpd}}_{AB}$ would be interpreted as strong evidence for model $A$.

# %% editable=true slideshow={"slide_type": ""}
def Belpd(𝒟, πlogσ, πlogT): return Criterion(elpd("Planck", 𝒟, πlogσ, πlogT),
                                             elpd("Rayleigh-Jeans", 𝒟, πlogσ, πlogT))


# %% [markdown] editable=true slideshow={"slide_type": ""}
# $\underline{B}^R$: Ratio of expected risk
# ~ As with the $\mathrm{elpd}$, the expected risk is more commonly given directly. However since we chose $Q$ to be the negative log likelihood, it is reasonable to present it as a ratio to make it comparable with other criteria:
#   :::{margin}
#   Signs are flipped because our criteria are interpreted as ratios of probabilities (i.e. negative loss).
#   :::
#   $$\log \underline{B}^R = -R_{\mathrm{P}} + R_{\mathrm{RJ}}$$

# %% editable=true slideshow={"slide_type": ""}
def BR(𝒟): return Criterion(-R("Planck", 𝒟), -R("Rayleigh-Jeans", 𝒟))


# %% editable=true slideshow={"slide_type": ""} tags=["remove-input"]
import emdcmp as emd
emd.utils.GitSHA()

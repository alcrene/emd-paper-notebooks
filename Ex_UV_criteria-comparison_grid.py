# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
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
#     '\MP'    : '\mathcal{M}_{\mathrm{P}}'
#     '\MRJ'   : '\mathcal{M}_{\mathrm{RJ}}'
#     '\NML'  : '\mathrm{NML}'
#     '\D'    : '\mathcal{D}'
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
# # Comparison with other criteria

# %% [markdown] editable=true slideshow={"slide_type": ""}
# {{ prolog }}
#
# %{{ startpreamble }}
# %{{ endpreamble }}

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
from Ex_UV import *
from numpy.polynomial import Polynomial
from collections.abc import Generator
from itertools import product, cycle
import logging
import math
import multiprocessing as mp
import re
import dataclasses
import shelve
import psutil
#from myst_nb import glue
#from viz import glue

# %% editable=true slideshow={"slide_type": ""}
from emd_falsify import draw_R_samples
from other_criteria import FactorizedPrior, get_ppfs, R, AIC, BIC, DIC, logℰ, elpd, 𝓁
import mdl_uv

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
logging.getLogger("mdl_uv").setLevel(logging.INFO)
do_long_computations = True
refresh_shelves = True  # If true, recreate shelves from joblib caches. For MDL, skips calculations if they aren’t already in the cache

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# # This cell only executed in notebooks
# do_long_computations = False
# refresh_shelves = False

# %% editable=true slideshow={"slide_type": ""}
import viz  # Local module
from paul_tol_rainbow import discrete_rainbow_scheme  # Local module
import matplotlib as mpl
import matplotlib.pyplot as plt
import panel as pn
import holoviews as hv
from cycler import cycler

hv.extension("matplotlib")


# %%
def pow2_formatter(x, pos):
    p = math.log2(x)
    if p.is_integer():
        p = int(p)
    return f"$2^{{{p}}}$"


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Use sans-serif for LaTeX axis labels.
# - STIX is maybe not the prettiest font, but it is very legible.
# - Bitstream Vera Sans is the matplotlib default for non-TeX labels.
# - See https://stackoverflow.com/a/27697390

# %%
params = {'text.usetex': False,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stixsans'
          #'mathtext.fontset': 'custom',
          #'mathtext.rm'     : 'Bitstream Vera Sans',
          #'mathtext.it'     : 'Bitstream Vera Sans:italic',
          #'mathtext.bf'     : 'Bitstream Vera Sans:bold',
         }
mpl.rcParams.update(params)


# %% [markdown]
# Extend the color scheme from `Ex_UV`. Some decent options for candidates, from most to least discriminable. The greater variety of colours in the most discriminable palettes makes them a bit jarring.
#
#     hv.Palette("TolRainbow", range=(0.1, 1.), reverse=True)
#     hv.Palette("Sunset", range=(0., 1.), reverse=True)
#     hv.Palette("Magma", range=(0., .9), reverse=False)
#     hv.Palette("TolYlOrBr", range=(0.3, 1.), reverse=True)

# %% editable=true slideshow={"slide_type": ""}
@dataclass
class colors(colors):
    #candidates : hv.Cycle = hv.Cycle(config.figures.colors["bright"]["cycle"])
    #candidates = hv.Palette("TolRainbow", range=(0.1, 1.), reverse=True)
    candidates : hv.Cycle = hv.Cycle(values = discrete_rainbow_scheme(4)[::-1])


# %% editable=true slideshow={"slide_type": ""}
colors

# %% [markdown]
# | Criterion    | Not restricted to nested models | No restriction on noise model | Not restricted to log likelihood risk | Not asymptotic | Allows singular models | Allows Bayesian models | Symmetric | Information-theoretic | Consistent | Robust viz. misspecification | Source of variability |
# |--------------|---------------------------------|-------------------------------|---------------------------------------|---------------|----------------------------|------------|--------------|-----------------------|------------|----------------------|----------------|
# | $\Bemd{}$    | ✔      | ✔ | ✔      | ✔ | ✔ | ✔ | ✔ | ✘ | ✔   | ✔    | Imperfect replications |
# | WAIC         | ✔      | ✔ | ✔      | ✔ | ✔ | ✔ | ✔ | ✘ | ✔   | ✘⁽²⁾ | Posterior              |
# | Bayes factor | ✔      | ✔ | ✘      | ✔ | ✘ | ✔ | ✔ | ✘ | ✔   | ✘⁽²⁾ | Posterior              |
# | MDL          | ✔/✘⁽¹⁾ | ✔ | ✔/✘⁽¹⁾ | ✔ | ✘ | ✔ | ✘ | ✔ | ✔   | ✘⁽³⁾ | ???                    |
# | (BIC)        | ✔      | ✔ | ✘      | ✘ | ✘ | ✔ | ✔ | ✘ | ✔/✘ | ✘    | Posterior              |
# | DIC          | ✘      | ✘ | ✘      | ✘ | ✘ | ✔ | ✘ | ✔ | ✔   | ✘    | Posterior              |
# | AIC          | ✘      | ✘ | ✘      | ✘ | ✘ | ✘ | ✘ | ✔ | ✘   | ✘    | Perfect replications   |
#
#
# ⁽¹⁾ Some entries are marked ✔/✘ for MDL because although the general form may not have these restrictions, they are usually required to obtain tractable approximations.  
# ⁽²⁾ Bayesian methods are sometimes described as accounting for epistemic errors, but only for errors which are within the hypothesis class. So specifically not misspecification.  
# ⁽³⁾ There has been at least one attempt to make MDL robust in the case of misspecified models, but it is far from standardized. It is also unclear whether it would be broadly applicable, since it depends on an ad hoc scaling of the likelihood, which itself introduces an additional hyperparameter.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Data
#
# For this experiment we want one of the candidates to be the true model. Also, in order to compute NML complexity exactly, we want to be able to enumerate all possible datasets. This is possible thanks to how we set the problem:
#
# - The use of Poisson noise means that possible values are discretized. Technically there are infinitely many possible values, but their probability quickly vanishes, so we need to consider only a dozen or so.
# - Fixed, exact $λ$ values: for a given dataset size $L$, we always generate data with the abscissa $λ_k = μλ_{\mathrm{min}} + k \frac{λ_{\mathrm{max}} - λ_{\mathrm{min}}}{L-1}$, where $k=0, \dotsc, L-1$.
#
# This means we can implement enumeration as follows:
# 1. For each $λ_k$, determine a set of possible values for $\Bspec_k$. Below we use an interval containing 0.9998% of the probability mass. Each $\Bspec_k$ is a sequential array with equal steps.
# 2. A dataset is generated by picking a random value from each $\Bspec_k$: $\{(λ_k, b_k) : b_k \sim \Bspec_k\}$.
#    The total number of datasets is therefore $\prod_{k=0}^{L-1} \lvert \Bspec_k \rvert$.
# 3. In practice there are still an unfeasibly large number of datasets. Therefore we generate them from most to least likely: the hope is that the integrand in an NML integral (which in this case is a series) should correlate with the true data likelihood, and that we can truncate the series once we have enough dominant terms.

# %% [markdown]
#     sizes = np.asarray(sizes, dtype=int)
#     tot_size = int(np.prod(sizes.astype(float)))  # float prevents overflow; casting to Python int is safe (worst case we get a bigint)
#     max_idx = sizes - 1
#     L = len(sizes)
#     for k in range(max_idx.sum()+1):
#         example_idx = get_unif_idx(sizes, k)
#         print(mult(sizes, k), example_idx)

# %% editable=true slideshow={"slide_type": ""}
𝒟 = mdl_uv.EnumerableDataset(
    "comparison with other criteria",
    L   =100,
    #λmin=0.1*μm,
    λmin=0.6*μm,
    λmax=2  *μm,
    #σ   =4e-5 * Bunits,
    #s   =(2e-2*Bunits)**-1,
    #s   =4*Bunits**-1,
    s   =16*Bunits**-1,
    T   =data_T)

# %% editable=true slideshow={"slide_type": ""}
#data   = Dataset("", L=4000, λmin=0.1*μm, λmax=2*μm, s=(2e-3*Bunits)**-1, T=3000*K)
𝒟_mean = replace(𝒟, s=(4e-5*Bunits)**-1)
dims = dict(kdims=hv.Dimension("λ", label="λ — wavelength", unit="μm"),
            vdims=hv.Dimension("B", label="$\\tilde{B}$ – radiance", unit="kW/m² nm sr"))
fig = hv.Scatter(𝒟.get_data(strip_units=True), **dims) * hv.Curve(𝒟_mean.get_data(strip_units=True), **dims)
fig.opts(
    hv.opts.Curve(color="#AA0022"),
    hv.opts.Scatter(size=1.5, backend="bokeh"),
    hv.opts.Scatter(s=12, backend="matplotlib"),
    hv.opts.Overlay(fig_inches=tuple(2*((40, 34)*ureg.mm).to("in").m),
                    fontscale=2, backend="matplotlib"),
    hv.opts.Overlay(width=500, backend="bokeh")
)

# %%
hv.save(fig, config.paths.figures/"compare-other-methods_sample-data.svg")


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Note that in contrast to the main setup, we consider a different regime, with wavelength window closer to the distribution’s peak and lower relative noise. This makes the likelihood essentially Gaussian, up to a normalization factor.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Candidate models
#
# Most established model comparison methods are meant to compare nested models, where preferably one of the alternatives is the true model. For criteria typically interpreted in terms of information theory, like AIC and MDL, that interpretation usually only holds when the true model is also the simplest.
# Therefore to give existing criteria the fairest comparison, we setup a model comparison problem where the model alternatives are variants of the Planck radiance model augmented with a polynomial:
#
# $$\tilde{\Bspec} \mid λ \sim \nN\biggl(\MP(λ\,;\, T) + \sum_{j=0}^{m-1} b_j λ^j \;,\;\; \frac{σ^2}{\MP(λ\,;\, T)}\biggr)$$
#
# Note that in contrast to the main setup, the difference between Poisson and Gaussian noise is not too important, for two reasons:
#
# - We increased the signal to noise ratio, partly by choosing a regime (i.e. a $λ$ window) which contains the largest radiance values.
# - We scale the variance in accordance with the Poisson distribution of the underlying data.
#
# This ensures that whether we fit the model using Poisson or Gaussian likelihood, the true model is recovered when $m=0$.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# It is worth emphasizing that in order to accommodate some criteria, we are posing a different problem than the one we actually want to solve. First, instead of comparing the Planck ($\MP$) and Rayleigh-Jeans ($\MRJ$) models, we compare some artificially augmented version of $\MP$. Second, we need to be in the regime where the Poisson is effectively a discretized Gaussian. This allows us to substitute the Poisson’s discrete PMF with the Gaussian’s continuous PDF for evaluating the loss. (If we tried to fit a discrete model, its PMF would vanish as soon as one data point is not in the support.) This also ensures that the asymptotic criteria, which assume that we have reached the regime where the likelihood is “Gaussian-like”, are valid. Finally, the discreteness of the data distribution nevertheless allows us to enumerate all possible datasets, a feature which is convenient for computing the NML distribution.
#
# Making these choices of regime and data-generating model ensures no method is disadvantaged, but of course in practice a practitioner does not get to choose their data, and should choose their model(s) based on what makes sense for those data, not what simplifies the statistical analysis.
#
# In practice of course we would like to be able to compare models without restriction on their structure or the type of noise, as we did in [Fig. ...] comparing the Planck and Rayleigh-Jeans models with $\Bemd{}$. Bayesian methods are also agnostic to the choice of model. Bayesian methods however consider a different type of epistemic error, and in particular ignore errors due to misspecification. We illustrate this with a comparison table in [Supplementary ...]. For completeness, this table also includes values obtained with the other criteria – since in practice they are often used even when they are invalid.

# %% editable=true slideshow={"slide_type": ""}
@dataclass(frozen=True)
class CandidateModel:
    physmodel: Literal["Plank","Rayleigh-Jeans"]
    σ : float|PintQuantity
    T : PintQuantity  # units: K
    coeffs: list[float]

    @property
    def m(self): 
        """Returns the order of the model’s spurious polynomial."""
        return len(self.coeffs)

    def __post_init__(self):
        # If parameters were given as plain floats, convert to default units
        if not isinstance(self.σ, pint.Quantity):
            object.__setattr__(self, "σ", self.σ * Bunits)   # IMO bypassing frozen dataclass in __post_init__ is acceptable
        if not isinstance(self.T, pint.Quantity):
            object.__setattr__(self, "T", self.T * ureg.kelvin)

    def Q(self, λ_B):
        """
        The loss is computed from differences between the observed data `λ_B`
        and predictions of the _physical_ model. The logpdf of the _observation_
        model is used to convert those differences into a loss.
        """
        λ, Bdata = λ_B
        _, Btheory = self.apply_physmodel(λ)
        #return -stats.poisson.logpdf((Bdata - Btheory).m , 
        Bphys = phys_models[self.physmodel](λ, self.T).to(Bunits)  # HACK: Copied from apply_physmodel, but this isn’t DRY
        return -stats.norm.logpdf((Bdata - Btheory).m, loc=0, scale=self.σ.m*np.sqrt(Bphys.m))  # IMPORTANT: Must match def in obsmodel

    def apply_obsmodel(self, λ_B, rng=None):
        λ, Bdata = λ_B
        Bphys = phys_models[self.physmodel](λ, self.T).to(Bunits)  # HACK: Copied from apply_physmodel, but this isn’t DRY
        return λ, Bdata + stats.norm.rvs(size=len(Bdata), loc=0, scale=self.σ.m*np.sqrt(Bphys.m), random_state=rng)*Bunits
    
    def apply_physmodel(self, λarr: ArrayLike, rng=None) -> tuple[ArrayLike, ArrayLike]:
        if isinstance(λarr, tuple):  # Allow passing output of data directly
            assert len(λarr) == 2, "CandidateModel expects either a single array `λarr`, or a tuple `(λarr, Barr)`."
            λarr = λarr[0]
        Barr = phys_models[self.physmodel](λarr, self.T).to(Bunits)
        if self.coeffs:
            Barr += Polynomial(self.coeffs)(λarr.m) * Bunits
        return λarr, Barr

    def gen_data(self, L, λmin=data_λ_min, λmax=data_λ_max, rng=None):
        λarr = np.linspace(λmin, λmax, L)
        return self.apply_obsmodel(self.apply_physmodel(λarr), rng=rng)
    
    def __call__(self, λarr, rng=None):
        return self.apply_obsmodel(self.apply_physmodel(λarr), rng=rng)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Criteria & loss definitions

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Risk functional
#
# We use the negative log likelihood as our risk functional, with a Gaussian observation model.
#
# (For technical reasons the risk functional is defined above as a method of the candidate model. The comparison functions expect a separate functional, so `Q[phys,σ,T,coeffs]` is a wrapper which calls `Q` from a properly parameterized model.)

# %% editable=true slideshow={"slide_type": ""}
class QMeta(type):
    def __getitem__(cls, Θ):
        σ, T, *coeffs = Θ
        return cls.__call__(σ, T, coeffs)
    def __call__(cls, σ, T, coeffs=()):
        if len(coeffs) == 1 and isinstance(coeffs[0], (list, tuple)):
            coeffs = coeffs[0]
        #return Q(σ, T, coeffs)
        model = CandidateModel(cls.physmodel, σ, T, coeffs)
        return model.Q
#@dataclass
class QPlanck(metaclass=QMeta):
    physmodel="Planck"
class QRJ(metaclass=QMeta):
    physmodel="Rayleigh-Jeans"
Q = {"Planck": QPlanck, "Rayleigh-Jeans": QRJ}


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Fitting function
# Given some observed data and the order of the extra polynomial, returns a tuple `(σ, T, coeffs)` of fitted parameters.
#
# Parameters are fitted by minimizing $Q$, subject to some mild regularization on $σ$ and $T$.

# %% editable=true slideshow={"slide_type": ""}
def _fitΘ(λℬ, physmodel, m):
    log2_T0    = np.log2(data_T.m)
    priorT_std = 12  # |log₂ 4000 - log₂ 5000| ≈ 12
    def f(Θtilde, physmodel=physmodel, λℬ=λℬ):
        log2σ, log2T, *coeffs = Θtilde
        σ = 2**log2σ; T = 2**log2T
        risk = Q[physmodel](σ, T, coeffs)(λℬ).mean()
        priorσ = 2**(-log2σ/128)  # Soft floor on σ so it cannot go too low
        priorT = (log2T - log2_T0)**2 / (2*priorT_std**2)  
        #priorσ = 0
        #priorT = 0
        return risk + priorσ + priorT

    res = optimize.minimize(f, np.r_[np.log2([.1, data_T.m]), [0]*m], tol=1e-5)
    log2σ, log2T, *coeffs = res.x
    σ = 2**log2σ; T = 2**log2T

    return σ, T, tuple(coeffs)

_fitΘ_cache = {}
def fitΘ(λℬ, physmodel, m):
    key = (tuple(λℬ[0].m), tuple(λℬ[1].m), physmodel, m)
    res = _fitΘ_cache.get(key)
    if res is None:
        res = _fitΘ(λℬ, physmodel, m)
    return res


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ::::{margin}
# :::{note}
# We initialize the fits practically at the MLE, to ensure we select the global optimum.
# :::
# ::::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Bayesian prior

# %% editable=true slideshow={"slide_type": ""}
π_phys_loose = FactorizedPrior(["loguniform", "loguniform"],
                               [(2**-10, 2**-1), (1000, 10000)],
                               rng="prior - compare - models")
π_phys_tight = FactorizedPrior(["loguniform", "loguniform"],
                               [(2**-5, 2**-4), (2000, 5000)],
                               rng="prior - compare - models")


# %% editable=true slideshow={"slide_type": ""}
def π_coeffs_loose(m):
    return FactorizedPrior(["norm"]*m, [(0, 5)]*m,
                           rng="prior - compare - models - coeffs")
def π_coeffs_tight(m):
    return FactorizedPrior(["norm"]*m, [(0, 2)]*m,
                           rng="prior - compare - models - coeffs")


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### NML distribution

# %% editable=true slideshow={"slide_type": ""}
def logL_mdl(*Θ, Q): return lambda λB: -Q(*Θ)(λB)

def MDL_criterion(Q, θˆ, 𝒟, physmodel, m):
    #return Q[θˆ](𝒟.get_data()) \
    comp = mdl_uv.comp(𝒟, r=30, m=m, logL=partial(logL_mdl, Q=Q),
                       fitΘ=lambda λℬ, m: fitΘ(λℬ, physmodel, m), 
                       cache_key="UV - Plank + poly", rng=3014,
                       no_compute=refresh_shelves)
    return None if comp is None else -𝓁(θˆ, Q, 𝒟) + comp


# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# ```python
# df_mdl = pd.DataFrame(
#     [(L, m, mdl_uv.comp(replace(𝒟, L=L), r=30, m=m, logL=partial(logL_mdl, Q=Q),
#                         fitΘ=fitΘ,
#                         cache_key="UV - Plank + poly", rng=3014))
#      for L in tqdm(Llist[:2], desc="L")
#      for m in tqdm(mlist, desc="m")],
#     columns=["L", "m", "comp"])
# ```

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Test criterion evaluations

# %% [markdown] editable=true slideshow={"slide_type": ""}
# θˆ = fitΘ(𝒟.get_data(), "Planck", m=1)
# 𝓜 = CandidateModel("Planck", *θˆ)
# π = π_phys_loose & π_coeffs_loose(m=1)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# R(Q["Planck"], θˆ, 𝒟)

# %% [markdown]
# AIC(Q["Planck"], θˆ, 𝒟)

# %% [markdown]
# BIC(Q["Planck"], θˆ, 𝒟)

# %% [markdown]
# π_coeffs_loose(m=1).rvs()

# %% [markdown]
# DIC(Q["Planck"], θˆ, π, 𝒟)

# %% [markdown]
# logℰ(Q["Planck"], π, 𝒟)

# %% [markdown]
# elpd(Q["Planck"], π, 𝒟, Lᑊ=4)

# %% [markdown]
# mixed_ppf, synth_ppf = get_ppfs(𝓜, 𝒟, rng=utils.get_rng("synth ppf - compare - models"))
# draw_R_samples(mixed_ppf, synth_ppf, 0.5)

# %% [markdown]
# :::{admonition} TODO
# MDL
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Compute criteria

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Note: We don’t change the dataset’s seed, so there is correlation in the data, but not exactly the same $(x_i, y_i)$ pairs are drawn. (Because the $x_i$ are distributed evenly across $[λ_{min}, λ_{max}]$.)

# %% editable=true slideshow={"slide_type": ""}
#Llist = (2**4, 2**6, 2**8, 2**10, 2**12)#, 2**14)
#mlist = range(0, 9)
#Llist = (2**2, 2**3, 2**4, 2**5, 2**6, 2**8, 2**10, 2**12)
#mlist = (0, 4, 16, 64)
#Lᑊ_elpd = 4
mlist = [0,2,4,6]
Llist       = [2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12]   # np.logspace(6, 12, 7, base=2)
Llist_short = [2**6,       2**8,       2**10,        2**12] 
Llist       = np.logspace(6, 12, 25, base=2).astype(int)
assert len(mlist) <= len(colors.candidates)

# %% [markdown]
# The PPF curves are useful diagnostics in and of themselves, to check that the fitted model behaves as expected.

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input", "active-ipynb"]
# Φarr = np.linspace(0, 1)
# ppf_panels = {}
# scatter_panels = {}
# def no_units(args): return tuple(a.m for a in args)
# for physmodel in ["Planck", "Rayleigh-Jeans"]:
#     for L in Llist:
#         for m in mlist:
#             𝒟 = replace(𝒟, L=L, purpose=𝒟.purpose + f" - {L=}")
#             θˆ = fitΘ(𝒟.get_data(), physmodel, m=m)
#             𝓜 = CandidateModel(physmodel, *θˆ)
#             mixed_ppf, synth_ppf = get_ppfs(𝓜, 𝒟, rng=utils.get_rng("synth ppf - compare - models"))
#             ppf_panels[physmodel, L, m] = \
#                 hv.Curve((Φarr, mixed_ppf(Φarr)), kdims="Φ", vdims="PPF", label="mixed") \
#                 * hv.Curve((Φarr, synth_ppf(Φarr)), kdims="Φ", vdims="PPF", label="synth")
#             scatter_panels[physmodel, L, m] = \
#                 hv.Scatter(no_units(𝓜.gen_data(D.L, λmin=D.λmin, λmax=D.λmax)), kdims="λ", vdims="B", label="model") \
#                 * hv.Scatter(no_units(𝒟.get_data()), kdims="λ", vdims="B", label="true")

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-input"]
# layout = hv.HoloMap(ppf_panels, kdims=["physical model", "L", "m"]) \
#          + hv.HoloMap(scatter_panels, kdims=["physical model", "L", "m"])
# layout.opts(fig_inches=2.5, sublabel_format="", fontsize={"title": 8}, fontscale=1.2)
# layout.opts(hv.opts.Scatter(alpha=0.5))

# %% [markdown]
# :::{note}
# Unbiased estimators should "flip-flop" between equivalent models.
# To show this, we change the seed for each dataset, i.e. for each different $L$.
# (We _don’t_ change for different $m$, because for a given dataset size, all models must be fitted to the same data.)
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{note}
# `elpd` is reported with “deviance” scaling, so that it is comparable with other information criteria.
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""}
# :::{note}
# Managing our own cache under `data/*-model-compare` makes it easier to archive results and package them with our code.
# :::
#
# ::::{margin}
# :::{hint}
# To get the location of the cache for $R$-distributions, check
# `emd_falsify.config.caching.joblib.location`
#
# The code from `other_criteria` caches itself under `.joblib-cache/joblib/other_criteria`
# :::
# ::::

# %% editable=true slideshow={"slide_type": ""}
import multiprocessing as mp
import psutil

def compute_MDL(phys_L_m, 𝒟=𝒟):
    physmodel, L, m = phys_L_m
    𝒟 = replace(𝒟, L=L, purpose=f"compare models - nested -- data -- {L}")
    θˆ = fitΘ(𝒟.get_data(), physmodel, m=m)
    _Q = Q[physmodel]

    try:
        mdl_score = MDL_criterion(_Q, θˆ, 𝒟, physmodel, m)
    except ValueError:
        # If computations for a particular set of args fails, don’t let the exception terminate other computations
        # Instead store a flag value, so we can diagnose this later
        mdl_score = None
    return physmodel, L, m, mdl_score

MDL_scores = {}
arglist = [(phys,L,m) for phys in ["Planck", "Rayleigh-Jeans"] for L in Llist for m in mlist
           if L <= 2**10   # Computations too slow otherwise
          ]
dellist = []  # indices of args which have already been computed, and can thus be removed from arglist
with shelve.open(str(config.paths.data/"MDL-model-compare")) as shelf:
    if refresh_shelves:
        shelf.clear()
    else:
        for i, (phys,L,m) in enumerate(arglist):
            shelfkey = f"{phys} - {L} - {m}"
            if (mdl_score := shelf.get(shelfkey)) is not None:
                MDL_scores["MDL", phys,m,L] = mdl_score
                dellist.append(i)
    
        for i in sorted(dellist, reverse=True):
            del arglist[i]

if arglist and do_long_computations:
    cores = min(psutil.cpu_count(logical=False), len(arglist))
    with mp.Pool(cores) as pool:
        for phys, L, m, mdl_score in tqdm(pool.imap_unordered(compute_MDL, arglist),
                                           desc="MDL scores", total=len(arglist), miniters=1):
            # By opening & closing the shelf every time we ensure a) that results are written immediately and b) that we don’t lock the shelf
            # TODO: Implement timeout & retry in case the shelf is locked by another process
            with shelve.open(str(config.paths.data/"MDL-model-compare")) as shelf:
                shelfkey = f"{phys} - {L} - {m}"
                shelf[shelfkey] = MDL_scores[phys, m, L] = mdl_score

# %% editable=true slideshow={"slide_type": ""} tags=["remove-output"]
# scores = {crit: {m: [] for m in range(4)}
#           for crit in ["AIC", "BIC", "EMD"]}
scores = {}
#scores = []
#_tqdm = lambda x, *a, **kw: x
crits_long = ["R", "AIC", "DIC", "BIC", "logE", "elpd"]
crits_short = []#["elpd"]
with shelve.open(str(config.paths.data/"criteria-model-compare")) as shelf_scores:
    for physmodel in ["Planck", "Rayleigh-Jeans"]:
        print(physmodel)
        progbar_L = tqdm(desc="L", total=len(Llist), miniters=1)
        progbar_m = tqdm(desc="m", total=len(mlist), miniters=1)
        for L in Llist:
            progbar_m.reset()
            for m in mlist:
                #tqdm.write(f"{L=}, {m=}")
                crits = (crits_long + crits_short) if L in Llist_short else crits_long
                keys = [(C, physmodel, m, L) for C in crits]
                strkeys = {key: ",".join(str(k) for k in key) for key in keys}
                _scores = {key: shelf_scores.get(strkeys[key]) for key in keys}
                missing = {C for (C, *_), s in _scores.items() if s is None}

                subkey = (physmodel, m, L)
                
                if missing:
                    π = π_phys_loose & π_coeffs_loose(m)
                    _𝒟 = replace(𝒟, L=L, purpose=f"compare models - nested -- data -- {L}")
                    _Q = Q[physmodel]
                    if missing & {"R", "AIC", "DIC", "BIC"}:
                        θˆ = fitΘ(𝒟.get_data(), physmodel, m=m)
                    else:
                        θˆ = None

                    if "R"    in missing: _scores[(key:=("R"   , *subkey))] = shelf_scores[strkeys[key]] = R(_Q, θˆ, _𝒟)
                    if "AIC"  in missing: _scores[(key:=("AIC" , *subkey))] = shelf_scores[strkeys[key]] = AIC(_Q, θˆ, _𝒟)
                    if "DIC"  in missing: _scores[(key:=("DIC" , *subkey))] = shelf_scores[strkeys[key]] = DIC(_Q, θˆ, π, _𝒟)
                    if "BIC"  in missing: _scores[(key:=("BIC" , *subkey))] = shelf_scores[strkeys[key]] = BIC(_Q, θˆ, _𝒟)
                    if "logE" in missing: _scores[(key:=("logE", *subkey))] = shelf_scores[strkeys[key]] = logℰ(_Q, π, _𝒟)
                    if "elpd" in missing: _scores[(key:=("elpd", *subkey))] = shelf_scores[strkeys[key]] = elpd(_Q, π, _𝒟, method="waic", scale="log")
                    #if L in Llist_short:
                    #    #scores[("MDL", physmodel, m, L)] = MDL_criterion(_Q, θˆ, _𝒟, m)
                    #    scores[("elpd", physmodel, m, L)], scores[("elpd_se", physmodel, m, L)] = elpd(_Q, π, _𝒟, method="waic", scale="log")
                    #    #𝓜 = CandidateModel(physmodel, *θˆ)
                    #    #mixed_ppf, synth_ppf = get_ppfs(𝓜, _𝒟, rng=utils.get_rng("synth ppf - compare - models"))

                    del π, _𝒟, _Q, θˆ
                    
                scores[("R", *subkey)]   = _scores[("R", *subkey)]
                scores[("AIC", *subkey)] = _scores[("AIC", *subkey)]
                scores[("DIC", *subkey)] = _scores[("DIC", *subkey)]
                scores[("BIC", *subkey)] = _scores[("BIC", *subkey)]
                scores[("logE", *subkey)], scores[("logE_se", *subkey)] = _scores[("logE", *subkey)]
                if L in ("elpd", *subkey) in _scores:
                    scores[("elpd", *subkey)], scores[("elpd_se", *subkey)] = _scores[("elpd", *subkey)]

                progbar_m.update()
            progbar_L.update()


# %% editable=true slideshow={"slide_type": ""}
def compute_Rdist(phys_L_m, 𝒟=𝒟):
    physmodel, L, m = phys_L_m
    𝒟 = replace(𝒟, L=L, purpose=f"compare models - nested -- data -- {L}")
    θˆ = fitΘ(𝒟.get_data(), physmodel, m=m)
    𝓜 = CandidateModel(physmodel, *θˆ)

    #scores[("MDL", m, L)] = MDL_criterion(Q, θ^, 𝒟, m)
    mixed_ppf, synth_ppf = get_ppfs(𝓜, 𝒟, rng=utils.get_rng("synth ppf - compare - models", "nested", "EMD", L, m))
    return physmodel, L, m, draw_R_samples(mixed_ppf, synth_ppf, c=0.5, max_M=2**14)

Rdists = {}
arglist = [(phys,L,m) for phys in ["Planck", "Rayleigh-Jeans"] for L in Llist_short for m in mlist]
dellist = []  # indices of args which have already been computed, and can thus be removed from arglist
with shelve.open(str(config.paths.data/"Rdists-model-compare")) as shelf:
    for i, (phys,L,m) in enumerate(arglist):
        shelfkey = f"{phys} - {L} - {m}"
        Rsamples = shelf.get(shelfkey)
        if Rsamples is not None:
            Rdists[phys,m,L] = Rsamples
            dellist.append(i)

    for i in sorted(dellist, reverse=True):
        del arglist[i]

    if arglist:
        cores = min(psutil.cpu_count(logical=False), len(arglist))
        with mp.Pool(cores) as pool:
            for phys, L, m, Rsamples in tqdm(pool.imap_unordered(compute_Rdist, arglist),
                                       desc="R distributions", total=len(arglist)):
                shelfkey = f"{phys} - {L} - {m}"
                shelf[shelfkey] = Rdists[phys, m, L] = Rsamples

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell", "active-py"] raw_mimetype=""
import sys; sys.exit()

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# :::{dropdown} Fully parallelized version
#
# The functions `logℰ` and `elpd` use MP unternally, but if we have a lot of CPUs, it may make sense to
# compute multiple scores at once (say each score using 4 CPUs).
#
# Unresolved issue:
# - Avoid nesting MP pools (this is not allowed).
# - Specify to subprocesses how many CPUs they can use, so that together they don’t use too many.
#
# ```python
# import multiprocessing as mp
# import psutil
#
# def compute_scores(L_m, 𝒟=𝒟):
#     L, m = L_m
#     𝒟 = replace(𝒟, L=L)
#     θˆ = fitΘ(𝒟.get_data(), m=m)
#     𝓜 = CandidateModel(*θˆ)
#
#     scores = {}
#     scores[("R", m, L)] = R(Q, θˆ, 𝒟)
#     scores[("AIC", m, L)] = AIC(Q, θˆ, 𝒟)
#     scores[("BIC", m, L)] = BIC(Q, θˆ, 𝒟)
#     scores[("logE", m, L)] = logℰ(Q, π, 𝒟)
#     scores[("elpd", m, L)] = elpd(Q, π, 𝒟, Lᑊ=Lᑊ_elpd)
#     #scores[("MDL", m, L)] = MDL_criterion(Q, θ^, 𝒟, m)
#     mixed_ppf, synth_ppf = get_ppfs(𝓜, 𝒟, rng=utils.get_rng("synth ppf - compare - models"))
#     scores[("Rdist", m, L)] = draw_R_samples(mixed_ppf, synth_ppf, c=0.5, max_M=2**14)
#
#     return scores
#
# scores = {}
# arglist = [(L,m) for L in Llist for m in mlist]
# cores = min(psutil.cpu_count(logical=False), len(arglist))
# with mp.Pool(cores) as pool:
#     for scoreset in tqdm(pool.imap_unordered(compute_scores, arglist),
#                          desc="Computing scores", total=len(arglist)):
#         scores.update(scoreset)
# ```
# :::

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
#     relscores = []
#     for key, s in scores.items():
#         crit, m, L = key
#         # For scores with arbitrary zero, compute relative values
#         #if crit in {"AIC", "BIC"}: s -= scores[crit, 0, L]
#         # For Rdist, unpivot/convert to a narrow format
#         if crit == "Rdist":
#             relscores.extend([(crit, m, L, _s) for _s in s])
#         else:
#             relscores.append((crit, m, L, s))

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# ```python
# tab = hv.Table(scores, kdims=["criterion", "phys", "m", "L"], vdims=["score"])
# df = tab.dframe().pivot(index=["L"], columns=["criterion", "m", "phys"], values=["score"]) \
#                  .droplevel(0, axis="columns")
# ```

# %% editable=true slideshow={"slide_type": ""}
df = pd.DataFrame([(*k, v) for k,v in (scores|MDL_scores).items()],
                  columns=["criterion", "phys", "m", "L", "score"])
df = df.pivot(index=["L"], columns=["criterion", "m", "phys"], values=["score"]) \
       .droplevel(0, axis="columns")

# %% editable=true slideshow={"slide_type": ""}
df


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Computed columns
#
# For comparability, we take the negative log evidence and negative elpd, so that for each criterion, _smaller is better_.
#
# We also display criteria with respect to a reference column (the one corresponding to the model with $m=0$).

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
def add_Δ_cols(df: pd.DataFrame, vdim, ref_col=None):
    """
    Add columns to `df` corresponding to the values of vdim subtracted by the reference column.
    By default, the `ref_col` is (`vdim`, 0).
    The new columns will have the criterion name 'Δ {vdim}'.
    """
    Δvdim = f"Δ {vdim}"
    if Δvdim in df.columns:
        # Δ cols have already been added
        return df
    if ref_col is None: ref_col = (vdim, 0)
    cols = df[vdim].columns
    Δdf = df[vdim].sub(df[ref_col], axis="index")
    cols_df = cols.to_frame()
    cols_df.insert(0, "criterion", Δvdim)
    #Δdf.columns = pd.MultiIndex.from_arrays([[Δvdim]*len(cols), cols], names=["criterion", *cols.names])
    Δdf.columns = pd.MultiIndex.from_frame(cols_df)
    return pd.concat((df, Δdf), axis="columns")

def add_neg_cols(df: pd.DataFrame, vdim):
    """
    Add columns to `df` corresponding to the negation of the subtracted by the reference column.
    The new columns will have the criterion name '-{vdim}'.
    """
    neg_vdim = f"-{vdim}"
    if neg_vdim in df.columns:
        # neg cols have already been added
        return df
    cols = df[vdim].columns
    neg_df = -df[vdim]
    cols_df = cols.to_frame()
    cols_df.insert(0, "criterion", neg_vdim)
    #neg_df.columns = pd.MultiIndex.from_arrays([[neg_vdim]*len(cols), cols], names=["criterion", *cols.names])
    neg_df.columns = pd.MultiIndex.from_frame(cols_df)
    return pd.concat((df, neg_df), axis="columns")


# %% editable=true slideshow={"slide_type": ""} tags=["remove-output"]
for vdim in ["BIC", "AIC", "logE", "elpd", "MDL"]:
    df = add_Δ_cols(df, vdim).sort_index()
for vdim in ["Δ logE", "Δ elpd"]:
    df = add_neg_cols(df, vdim).sort_index()


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Criteria comparison grid

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
# :::{dropdown} Plot curves with Holoviews
# ```python
# def plot_curve_w_se(df, vdim, vdim_se, orientation: Literal["horizontal","vertical"]="horizontal"):
#     assert orientation[:3] in {"hor", "ver"}
#     df = pd.concat((df[vdim], df[vdim] - df[vdim_se], df[vdim] + df[vdim_se]),
#                    axis="columns", keys=[vdim, f"{vdim}_low", f"{vdim}_high"])
#     xvals = df.index
#     yvals = df[vdim, m]
#     kdims = "L"
#     
#     if orientation.startswith("ver"):
#         xvals, yvals = yvals, xvals
#     curves = hv.NdOverlay({m: hv.Curve((xvals, yvals), kdims="L", vdims=vdim)
#                            for m in df.columns.unique("m")},
#                           kdims=["m"])
#     areas = hv.NdOverlay({m: hv.Area((df.index, df[f"{vdim}_low", m], df[f"{vdim}_high", m]),
#                                      kdims="L", vdims=[vdim, f"{vdim}2"])
#                           for m in df.columns.unique("m")},
#                          kdims=["m"])
#     return areas * curves
# ```
# :::

# %% editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
def plot_curve(df, vdim, vdim_se=None, mdim="m",
               orientation: Literal["horizontal","vertical"]="horizontal",
               error_style: Literal["fill", "bar", None]=None, offset: bool=False,
               ax=None, figsize=(30, 20)*ureg.mm, err_kwargs=None, **kwargs):
    """
    `offset` can be either a bool or a real value. If True, it defaults to 0.025.
    `err_kwargs` are combined with (and override) `kwargs`, so common args (like color)
       only need to be set once.
    """
    assert orientation[:3] in {"hor", "ver"}
    assert error_style in {"fill", "bar", None}
    vertical = bool(orientation.startswith("ver"))

    if isinstance(figsize, pint.Quantity):
        figsize = figsize.to(ureg.inch).m
    #figsize = ((60,40)*ureg.mm).to(ureg.inch).m

    if offset is True:
        offset = 0.025
    
    if ax is None:
        # Avoid using the `plt` interface to avoid memory leaks.
        # See https://panel.holoviz.org/reference/panes/Matplotlib.html#using-the-matplotlib-pyplot-interface
        fig = mpl.figure.Figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None

    if error_style=="fill":
        df = pd.concat((df[vdim], df[vdim] - df[vdim_se], df[vdim] + df[vdim_se]),
                       axis="columns", keys=[vdim, f"{vdim}_low", f"{vdim}_high"])
    if err_kwargs is None: err_kwargs = {}

    mvals = df.columns.unique(mdim)
    for i, m in enumerate(mvals):
        xvals = df.index
        yvals = df[vdim, m]
        if offset:
            xvals = xvals * 10**(offset*(i-len(mvals)/2))
        if orientation.startswith("ver"):
            xvals, yvals = yvals, xvals
        if error_style == "bar":
            _kwargs = kwargs | err_kwargs
            ax.errorbar(xvals, yvals,
                        **{("xerr" if vertical else "yerr"): df[vdim_se, m]},
                        **_kwargs)
            continue  # Skip plotting the center line
        if error_style == "fill":
            _kwargs = kwargs | {"alpha": 0.65} | err_kwargs
            ax._fill_between_x_or_y("y" if vertical else "x",
                                    df.index, df[f"{vdim}_low", m], df[f"{vdim}_high", m],
                                    **_kwargs)
        ax.plot(xvals, yvals, **kwargs)
    
    if vertical:
        kaxis = ax.yaxis
        vaxis = ax.xaxis
        ax.invert_yaxis()
    else:
        kaxis = ax.xaxis
        vaxis = ax.yaxis
    
    kaxis._set_axes_scale("log")
    kaxis.minorticks_off()
    kaxis._set_tick_locations(df.index)
    kaxis.set_major_formatter(mpl.ticker.FuncFormatter(pow2_formatter))
    
    kaxis.set_label_text("$L$")
    vaxis.set_label_text(vdim)

    return fig


# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
def plot_Rdists(Rdists: dict,
                orientation: Literal["horizontal","vertical"]="horizontal",
                offset: bool=False,
                ref_m=0,
                width=0.25,
                ylabel="$R$ distribution (EMD)",
                colors=colors.candidates.values,
                alpha=0.65,
                linewidth=1,
                ax=None):
    """
    `ref_m`: The grey reference line will be the mean of this distribution
    `width`: width or height of distributions. Scaled with L so they are visually consistent.
    """
    assert orientation[:3] in {"hor", "ver"}
    vertical = bool(orientation.startswith("ver"))
    assert isinstance(colors, list)
    
    if offset is True:
        offset = 0.025
    
    if ax is None:
        # Avoid using the `plt` interface to avoid memory leaks.
        # See https://panel.holoviz.org/reference/panes/Matplotlib.html#using-the-matplotlib-pyplot-interface
        fig = mpl.figure.Figure(figsize=(4,2))
        ax = fig.subplots()
    else:
        fig = None

    mlist = sorted({m for (m, L) in Rdists})
    Llist = sorted({L for (m, L) in Rdists})
    
    data = []
    Lvals = set()  # L values before any offset
    positions = []
    widths = []
    #n_m = len(np.unique(df.index.get_level_values("m")))
    #for (L,m), Rvals in df.groupby(["L", "m"]):
    # NB: Color cycle logic assumes the fast loop is over m
    for i, (L, m) in enumerate(product(Llist, mlist)):
        Rvals = Rdists[m,L]
        data.append(Rvals.flatten())
        if offset:
            L = L * 10**(offset*(i-len(mlist)/2))
        positions.append( L )
        widths.append( width * L )

    if vertical:
        ax.axvline(Rdists[ref_m, Llist[-1]].mean(), color="#AAAAAA", alpha=0.15, zorder=-1)
    else:
        ax.axhline(Rdists[ref_m, Llist[-1]].mean(), color="#AAAAAA", alpha=0.15, zorder=-1)

    # NB: A horizontal plot (L along x axis) needs *vertical* distributions
    parts = ax.violinplot(data, positions, widths=widths, orientation="horizontal" if vertical else "vertical",
                          showextrema=False, showmeans=False, side="low")
    for i, (m, polycoll, c) in enumerate(zip(cycle(mlist), parts["bodies"], cycle(colors[:len(mlist)]))):
        polycoll.set_facecolor(c)
        polycoll.set_edgecolor("none")
        polycoll.set_alpha(alpha=alpha)
        # AFAICT, it is not possible to set separate alpha values for face and edge colors (PolyCollection has its own alpha setting, which overrides the alpha of both facecolor and edgecolor)
        # Moreover, it draws the line down the middle of the violin, which in our case adds a useless flat line to the bottom of each plot.
        # So instead we extract the edge path and draw it ourselves
        path = polycoll._paths[0]  # Path is formed of two parts: first the curved edge, then the straight line in the middle of the violin
        edge = mpl.patches.PathPatch(mpl.path.Path(path.vertices[:len(path)//2], path.codes[:len(path)//2]),  # Draw only the curved part
                                     edgecolor=c, linewidth=linewidth,
                                     label=str(m) if i < len(mlist) else "_")  # Only apply one label for each m value, otherwise we can duplicate entries in the legend
        ax.add_patch(edge)

    if vertical:
        kaxis = ax.yaxis
        vaxis = ax.xaxis
        ax.invert_yaxis()
    else:
        kaxis = ax.xaxis
        vaxis = ax.yaxis
    
    kaxis._set_axes_scale("log")
    kaxis.minorticks_off()
    kaxis._set_tick_locations(Llist)
    kaxis.set_major_formatter(mpl.ticker.FuncFormatter(pow2_formatter))
    
    kaxis.set_label_text("$L$")
    vaxis.set_label_text(ylabel)

    return fig


# %% editable=true slideshow={"slide_type": ""}
_Rdists = {(m, L): Rdist for (phys, m, L), Rdist in Rdists.items() if phys == "Planck"}
pn.pane.Matplotlib(
    plot_Rdists(_Rdists, "vert", offset=0.05, width=1, colors=discrete_rainbow_scheme(len(mlist)), alpha=.65),
    tight=True
)

# %% [markdown]
# :::{note}
# Even with the Planck model, there is still some small level of mismatch because the candidates use Gaussian noise, while the data use Poisson.
# This is almost certainly why higher-order polynomials obtain tighter $R$-distributions.
# :::

# %% editable=true slideshow={"slide_type": ""}
_Rdists = {(m, L): Rdist for (phys, m, L), Rdist in Rdists.items() if phys == "Rayleigh-Jeans"}
pn.pane.Matplotlib(
    plot_Rdists(_Rdists, "vert", offset=0.05, ref_m=6, width=1, colors=discrete_rainbow_scheme(len(mlist)), alpha=.65),
    tight=True
)

# %% editable=true slideshow={"slide_type": ""}
_Rdists = {(phys, L): Rdist for (phys, m, L), Rdist in Rdists.items() if m == 0}
pn.pane.Matplotlib(
    plot_Rdists(_Rdists, "vert", offset=0, ref_m="Planck", width=1, colors=discrete_rainbow_scheme(2), alpha=.65),
    tight=True
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Planck is favoured by $\Bemd{}$, although the difference is less when we add polynomials (which can partially compensate for the incorrect Rayleigh-Jeans model).

# %% editable=true slideshow={"slide_type": ""}
P_index  = pd.MultiIndex.from_tuples([("Planck", m_P) for m_P in mlist], names=["phys model", "m"])
RJ_index = pd.MultiIndex.from_tuples([("Rayleigh-Jeans", m_P) for m_P in mlist], name=["phys model", "m"])
pd.DataFrame(
    [[np.less.outer(Rdists[(*P_model, 2**12)], Rdists[(*RJ_model, 2**12)]).mean()
      for RJ_model in RJ_index] for P_model in P_index],
    index=P_index, columns=RJ_index
).style.set_caption(r"$B^{\mathrm{EMD}}$ between Planck & Rayleigh-Jeans models")

# %% editable=true slideshow={"slide_type": ""}
w = 30.548
fig = mpl.figure.Figure(figsize=((6*w + 50, 3*40)*ureg.mm).to("in").m)
axes = fig.subplots(3,6)

## Nested models with true (Planck + poly) ##
axes_row = axes[0]
color_cycle = cycler(color=discrete_rainbow_scheme(len(mlist)))
for ax in axes_row:
    ax.set_prop_cycle(color_cycle)
_df = df.loc[:,(np.s_[:],np.s_[:],"Planck")].droplevel("phys", axis="columns")
_Rdists = {(m, L): Rd for (phys, m, L), Rd in Rdists.items() if phys=="Planck"}
plot_curve(_df, "Δ BIC" ,       None, "m", "vert", ax=axes_row[0])
plot_curve(_df, "-Δ logE", "logE_se", "m", "vert", ax=axes_row[1], error_style="fill")
# Server issues corrupted the results for L=256 and L=64. Don’t drop L=64 because it would slightly shift the axis limits
plot_curve(_df.drop([256]), "Δ MDL" ,       None, "m", "vert", ax=axes_row[2])
plot_curve(_df, "Δ AIC" ,       None, "m", "vert", ax=axes_row[3])
# One doesn’t see anything when plotting the elpd as curves, so plot only the error bars instead
plot_curve(_df.loc[Llist_short], "Δ elpd", "elpd_se", "m", "vert", ax=axes_row[4], error_style="bar", linestyle="", offset=0.025)
plot_Rdists(_Rdists,                 "vert", ax=axes_row[5], offset=0.05, width=1, colors=color_cycle.by_key()["color"], alpha=.15)
axes_row[-1].legend(loc="upper left", bbox_to_anchor=(1,1))

## Nested models without true (Rayleigh-Jeans + poly) ##
axes_row = axes[1]
color_cycle = cycler(color=discrete_rainbow_scheme(len(mlist)))
for ax in axes_row:
    ax.set_prop_cycle(color_cycle)
_df = df.loc[:,(np.s_[:],np.s_[:],"Rayleigh-Jeans")].droplevel("phys", axis="columns")
_Rdists = {(m, L): Rd for (phys, m, L), Rd in Rdists.items() if phys=="Rayleigh-Jeans"}
plot_curve(_df, "Δ BIC" ,       None, "m", "vert", ax=axes_row[0])
plot_curve(_df, "-Δ logE", "logE_se", "m", "vert", ax=axes_row[1], error_style="fill")
plot_curve(_df.drop([256]), "Δ MDL" ,       None, "m", "vert", ax=axes_row[2])
plot_curve(_df, "Δ AIC" ,       None, "m", "vert", ax=axes_row[3])
plot_curve(_df, "-Δ elpd", "elpd_se", "m", "vert", ax=axes_row[4], error_style="fill")
plot_Rdists(_Rdists,                 "vert", ax=axes_row[5], offset=0.05, width=1, colors=color_cycle.by_key()["color"], alpha=.15)

## Structurally different models (Planck vs Rayleigh-Jeans, m=0) ##
axes_row = axes[2]
color_cycle = cycler(color=discrete_rainbow_scheme(2))
for ax in axes_row:
    ax.set_prop_cycle(color_cycle)
_df = df.loc[:,(np.s_[:],0)].droplevel("m", axis="columns")
_Rdists = {(phys, L): Rd for (phys, m, L), Rd in Rdists.items() if m==0}
# Recompute relative values wrt to Planck, m=0
_df.drop([C for C in df.columns.unique("criterion") if C[0] in "-Δ"],
          axis="columns", inplace=True)
for vdim in ["BIC", "AIC", "logE", "elpd", "MDL"]:
    _df = add_Δ_cols(_df, vdim, ref_col=(vdim, "Planck")).sort_index()
for vdim in ["Δ logE", "Δ elpd"]:
    _df = add_neg_cols(_df, vdim).sort_index()
# Now do the plots
plot_curve(_df, "Δ BIC" ,       None, "phys", "vert", ax=axes_row[0])
plot_curve(_df, "-Δ logE", "logE_se", "phys", "vert", ax=axes_row[1], error_style="bar")
plot_curve(_df.drop([256]), "Δ MDL" ,       None, "phys", "vert", ax=axes_row[2])
plot_curve(_df, "Δ AIC" ,       None, "phys", "vert", ax=axes_row[3])
plot_curve(_df, "-Δ elpd", "elpd_se", "phys", "vert", ax=axes_row[4], error_style="fill")
plot_Rdists(_Rdists,                 "vert", ax=axes_row[5], offset=0, ref_m="Planck", width=1, colors=color_cycle.by_key()["color"], alpha=.15)
axes_row[-1].legend(loc="upper left", bbox_to_anchor=(1,1))

for ax in axes[:-1,:].flat:
    ax.set_xlabel(None)
#    ax.xaxis.set_visible(False)
for ax in axes[:,1:].flat:
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
for ax in axes[:,0]:
    ax.set_yticks(Llist_short)

pn.pane.Matplotlib(fig, tight=True, width=800)

# %% editable=true slideshow={"slide_type": ""}
fig.savefig(config.paths.figures/"compare-other-methods_grid_raw.svg")

# %% [markdown]
# Standard error on $\log \eE$ is negligible:

# %%
df["logE_se"].max(None)

# %% editable=true slideshow={"slide_type": ""} tags=["remove-input"]
emd.utils.GitSHA()

# %% editable=true slideshow={"slide_type": ""}

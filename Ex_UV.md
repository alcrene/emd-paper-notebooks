---
jupytext:
  formats: ipynb,py:percent,md:myst
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
    '\eE'   : '\mathcal{E}'
    '\ll'   : '\mathcal{l}'
    '\nN'   : '\mathcal{N}'
    '\Unif' : '\operatorname{Unif}'
    '\Poisson': '\operatorname{Poisson}'
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Code for comparing models of blackbody radiation

{{ prolog }}

%{{ startpreamble }}
%{{ endpreamble }}

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

> **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).

> **NOTE** This notebook is synced with a Python file using [Jupytext](https://jupytext.readthedocs.io/). **That file is required** to run this notebook, and it must be in the current working directory.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

:::{hint}
When running within Jupyter, don’t run the first half of the notebook (unless you specifically want to run those parts).
Skip to the cell containing
```python
from Ex_UV import *
hv.extension("matplotlib", "bokeh")
```
and start executing from there. This is much faster than running the entire notebook from the start, since although most of the preceding cells will still be executed, those producing plots will be skipped.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
import logging
import time
import numpy as np
import pandas as pd
import pint
import holoviews as hv

from numpy.typing import ArrayLike
from scipy import stats
from scipy import optimize

from collections import OrderedDict, namedtuple
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache, lru_cache, wraps, partial
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from warnings import filterwarnings, catch_warnings

from tqdm.notebook import tqdm
from addict import Dict
from joblib import Memory
from scityping.base import Dataclass
from scityping.functions import PureFunction
from scityping.numpy import RNGenerator
from scityping.scipy import Distribution
from scityping.pint import PintQuantity

logger = logging.getLogger(__name__)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

EMD imports

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import emd_falsify as emd
import emd_falsify.tasks
import emd_falsify.viz
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Project imports

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import utils
import viz
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

Notebook imports

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell, active-ipynb]
---
import itertools
import scipy.integrate
import scipy.special
#from myst_nb import glue
from viz import glue
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Configuration

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The project configuration object allows to modify not just configuration options for this project, but also exposes the config objects dependencies which use [*ValConfig*](https://validating-config.readthedocs.io).

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from config import config
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
memory = Memory(".joblib-cache", verbose=0)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Dimensions

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
ureg = pint.get_application_registry()
ureg.default_format = "~P"
if "photons" not in ureg: ureg.define("photons = count")
Q_  = pint.Quantity
K   = ureg.K
μm  = ureg.μm
nm  = ureg.nm
kW  = ureg.kW
c   = ureg.c
h   = ureg.planck_constant
kB  = ureg.boltzmann_constant
photons = ureg.photons

Bunits = ureg.kW / ureg.steradian / ureg.m**2 / ureg.nm
sunits = photons / Bunits
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
dims = viz.dims.matplotlib
```

### Notebook parameters

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Dataset sizes
L_small    = 2**9     # 512
L_med      = 2**12    # 4096
L_large    = 2**15    # 32768  –  Unecessarily large for most cases.
                      # Used only to check that a criterion saturates.
L_synth    = 2**12    # Dataset size for constructing empirical PPFs.
                      # Generally this is cheap, so we want to make this large enough
                      # to make numerical errors in the PPF negligible.
# Dataset parameters
data_λ_min   = 15*μm
#data_λ_min   = 20*μm
data_λ_max   = 30*μm
data_T       = 4000*K
data_noise_s = 1e5 * sunits  # Determines variance of the Poisson noise
                             # (Inverse proportional to std dev)
data_B0      = 2.5e-6 * Bunits

# Fitting parameters (used for the illustrative temperature fits)
Nfits = 64
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
#config.emd.mp.max_cores = 2
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell, skip-execution]
---
viz.save.update_figure_files = False
```

### Plotting

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
@dataclass
class colors(viz.ColorScheme):
    scale  : str = "#222222"
    data   : str = "#b2b2b2"  # Greyscale version of "high-constrast:yellow"
    RJ     : str = config.figures.colors["high-contrast"].red
    Planck : str = config.figures.colors["high-contrast"].blue
    calib_curves: hv.Palette = hv.Palette("YlOrBr", range=(0.1, .65), reverse=True)

@dataclass
class color_labels:
    data   : str = "grey"
    RJ     : str = "red"
    Planck : str = "blue"

dash_patterns = ["dotted", "dashed", "solid"]
sanitize = hv.core.util.sanitize_identifier_fn
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
hv.extension("matplotlib", "bokeh")
```

```{code-cell} ipython3
colors
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Model definition

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Generative physical model

To compare candidate models using the EMD criterion, we need three things:

- A **generative model** for each candidate, also called a “forward model”.
- A **risk function** for each candidate. When available, the negative log likelihood is often a good choice.
- A **data generative process**. This may be an actual experiment, or a simulated experiment as we do here.
  In the case of a simulated experiment, we may use the shorthand “true model” for this, but it is important to remember that it is different from the candidate models.

:::{important}
The true model is treated as a black box, because in real applications we don’t have access to it. We have no equations for the true model – only data. Consequently, there are no “true parameters” against which to compare, nor is the notion of “true risk” even defined. **The only thing we can do with the true model is request new data.**

Nevertheless, in the case of simulated experiments where we actually do have equations for the true model, it is still useful as a sanity check to compare parameters of the true and candidate models. However such a comparison cannot be part of any quantitative assessment.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Model signatures & implementation


- The **data generating model** should take one argument – an integer $L$ – and return a dataset with $L$ samples:
  $$\begin{aligned}
  \texttt{data-model}&:& L &\mapsto
      \begin{cases}
          \bigl[(x_1, y_1), (x_2, y_2), \dotsc, (x_L, y_L)\bigr] \\
          (xy)_1, (xy)_2, \dotsc, (xy)_L
      \end{cases}
  \end{aligned}$$
  :::{note}
  :class: margin
  
  A process with no independent variable (like drawing from a static distribution) is equivalent to the $(xy)_i$ format with a dummy value for $x$.
  :::
  This dataset will normally be composed of pairs of independent ($x$) and dependent ($y$) variables. Whether these are returned as separate values $(x,y)$ or combined $(xy)$ is up to user preference.

- The **candidate models** attempt to predict the set of $\{y_i\}$ from the set of $\{x_i\}$; we denote these predictions $\{\hat{y}_i\}$. Therefore they normally take one of the following forms:
  $$\begin{aligned}
  \texttt{candidate-model}&:& \{(x_i,y_i)\} &\mapsto \{\hat{y}_i\} \\
  \texttt{candidate-model}&:& \bigl\{(xy)_i\bigr\} &\mapsto \{\hat{y}_i\}
  \end{aligned}$$
  *In addition*, they may also accept the simplified form
  $$\begin{aligned}
  \texttt{candidate-model}&:& \{x_i\} &\mapsto \{\hat{y}_i\} \,.
  \end{aligned}$$
  This can be done by inspecting the argument, to see if it matches the form $\{(x_i,y_i)\}$ or $\{x_i\}$.
  :::{important}
  The *output* of the data model must be a valid *input* for a candidate model. Generally the latter will disregard the $y_i$ component when making its prediction $\hat{y}_i$. (There may be cases where $y_i$ is actually used, for example if we want to test models based on their one-step-ahead prediction.)
  :::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{important}
Our implementation assumes that a) any arbitrary number of samples $L$ can be requested, and b) that samples are all of equivalent quality.

For example, with data provided by solving an ODE, the correct way to increase the number of samples is to keep the time step $Δt$ fixed and to increase the integration time. Decreasing $Δt$ would NOT work: altough it results in more samples, they are also more correlated, and thus of “lesser quality”.

In contrast, for a static relationship like the radiance of a black body, then we *do* want to keep the same upper and lower bounds but increase the density of points within those bounds. (Extending the bounds then would mean testing a different physical regime.) For static relationships, points can be generated independently, so generating more points within the same interval does increase statistical power.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{note}
:class: margin

In theory data and candidate models can also be defined using only plain functions, but classes make it easier to control how the model is pickled. (Pickling is used to send data to multiprocessing subthreads.)
:::

```{code-cell} ipython3
def BRayleighJeans(λ, T, c=c, kB=kB): return 2*c*kB*T/λ**4
def BPlanck(λ, T, h=h, c=c, kB=kB): return 2*h*c**2/λ**5 / ( np.exp(h*c/(λ*kB*T)) - 1 )

phys_models = {
    "Rayleigh-Jeans":BRayleighJeans,
    "Planck"        :BPlanck
}
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@dataclass(frozen=True)
class DataModel:
    T         : PintQuantity    # units: K
    λ_min     : PintQuantity    # units: nm
    λ_max     : PintQuantity    # units: nm
    phys_model: Literal["Rayleigh-Jeans", "Planck"]
    
    def __call__(self, L: int, rng=None) -> tuple[ArrayLike, ArrayLike]:
        λarr = np.linspace(self.λ_min, self.λ_max, L)
        # NB: Returning as tuple (rather than 2D array) allows to keep units
        return λarr, phys_models[self.phys_model](λarr, self.T).to(Bunits)

@dataclass(frozen=True)
class CandidateModel:
    """Model variant used to compute candidate predictions given observed data.
    Uses the {(x,y)} -> {ŷ} signature. Also accepts {x} -> {ŷ}.

    Instead of an array of time points, takes previously computed (or recorded)
    data, extracts the time points, and computes the candidate model’s prediction.
    """
    phys_model: Literal["Rayleigh-Jeans", "Planck"]
    T         : PintQuantity    # units: K
    
    def __call__(self, λarr: ArrayLike, rng=None) -> tuple[ArrayLike, ArrayLike]:
        if isinstance(λarr, tuple):  # Allow passing output of data directly
            assert len(λarr) == 2, "CandidateModel expects either a single array `λarr`, or a tuple `(λarr, Barr)`."
            λarr = λarr[0]
        return λarr, phys_models[self.phys_model](λarr, self.T).to(Bunits)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_uv_observation-models)=
### Generative observation model

An observation model is a function which takes both the independent and dependent variables (generically $x$ and $y$), and returns a transformed value $\tilde{y}$.
Processes which are time-independent (such as simple additive Gaussian noise) can simply ignore their $x$ argument.

The pattern for defining a noise model is a pair of nested functions. The outer function sets the noise parameters, the inner function applies the noise to data:

$$\begin{aligned}
&\texttt{def public-noise-name}([arguments...], rng): \\
&\qquad\texttt{def }ξ(x, y, rng=rng): \\
&\qquad\qquad\texttt{return}\; \tilde{y} \\
&\qquad\texttt{return}\; ξ
\end{aligned}$$

For consistency, noise arguments should always include a random number generator `rng`, even if the noise is actually deterministic. The value passed as `rng` should be a `Generator` instance, as created by `numpy.random.default_rng` or `numpy.random.Generator`.

+++

We use two types of noise for this example:

Poisson noise
~ is used to generate the actual data. (Eq. {eq}`eq_UV_true-obs-model`)

  $$\tilde{\Bspec} \mid \Bspec \sim \frac{1}{s}\Poisson\bigl(s \, \Bspec\bigr) + \Bspec_0$$ (eq_code_poisson-noise)

Gaussian noise
~ is used by the candidate models (since we use squared difference to quantify error)
  
  $$\tilde{\Bspec} \mid \Bspec \sim \nN\bigl(\Bspec + \Bspec_0, σ^2\bigr)$$

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@dataclass(frozen=True)
class poisson_noise:
    label="poisson"
    B0 : PintQuantity
    s  : PintQuantity
    rng: RNGenerator|None=None
    def __call__(self, λ_B: tuple[ArrayLike, ArrayLike], rng: np.random.Generator=None) -> tuple[ArrayLike, ArrayLike]:
        λ, B = λ_B
        rng = rng or self.rng
        sB = self.s*B
        assert sB.dimensionless; "Dimension error: s*B should be dimensionless"
        assert rng, "No RNG was specified"
        return λ, rng.poisson(sB.m) / self.s + self.B0
        #return xy + self.rv.rvs(size=xy.shape, random_state=rng)

@dataclass(frozen=True)
class gaussian_noise:
    label="gaussian"
    B0 : PintQuantity
    σ  : PintQuantity
    rng: RNGenerator|None=None
    def __call__(self, λ_B: tuple[ArrayLike, ArrayLike], rng: np.random.Generator=None) -> tuple[ArrayLike, ArrayLike]:
        λ, B = λ_B
        rng = rng or self.rng
        assert rng, "No RNG was specified"
        return λ, B + self.B0 + self.σ*rng.normal(0, 1, size=B.shape)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Full data model definition

Our full data generating model is the Planck model composed with Poisson noise.

:::{important}
The `purpose` attribute serves as our RNG seed.
We take care not to make the RNG itself an attribute: sending RN generators is fragile and memory-hungry, while sending seeds is cheap and easy.
Within the `Calibrate` task, each data model is shipped to an MP subprocess with pickle: by making `Dataset` callable, we can ship the `Dataset` instance (which only needs to serialize a seed) instead of the `utils.compose` instance (which would serialize an RNG)
:::

:::{hint}
:class: margin

`utils.compose(g, f)` works like $g \circ f$.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

```python
data_model = utils.compose(poisson_noise(s=data_noise_s, B0=data_B0),
                           DataModel(λ_min=data_λ_min, λ_max=data_λ_max, T=data_T, phys_model="Planck"))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
@dataclass(frozen=True)
class Dataset:
    purpose: str                    # Used to seed the RNG
    L      : int
    λmin   : PintQuantity
    λmax   : PintQuantity
    s      : PintQuantity
    T      : PintQuantity=data_T
    B0     : PintQuantity=0*Bunits
    phys_model: str="Planck"        # Only changed during calibration
    @property
    def rng(self): return utils.get_rng("UV", self.purpose)
    @property
    def data_model(self):
        return utils.compose(
            poisson_noise(s=self.s, B0=self.B0, rng=self.rng),
            DataModel(λ_min=self.λmin, λ_max=self.λmax, T=self.T, phys_model=self.phys_model) )
    @cache
    def get_data(self, rng=None):
        return self.__call__(self.L, rng)
    def __call__(self, L, rng=None):
        return self.data_model(L, rng=(rng or self.rng))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Observed data set

Using the data model, we generate a unique dataset which simulates the real recorded data.
The goal for the rest of this notebook is to compare candidate models against this dataset

+++ {"editable": true, "slideshow": {"slide_type": ""}}

```python
λ_data, B_data = data_model(L_med, rng=utils.get_rng("UV", "data"))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
observed_dataset = Dataset("data", L_med, data_λ_min, data_λ_max, data_noise_s, data_T, data_B0)
λ_data, B_data = observed_dataset.get_data()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Loss function

Loss functions follow from the _observation model_ assumed by a candidate model. In this example we assume Gaussian observation noise and choose the risk to be the negative log likelihood – so for a model candidate $a$, observed data point $\tilde{\Bspec}$ and model prediction $\hat{\Bspec}$, the risk $q$ is given by
$$q_a(λ,\tilde{\Bspec}) = -\log p\Bigl(\tilde{\Bspec} \,\Bigm|\, \hat{\Bspec}_{a}(λ; T), σ\Bigr) = \log \sqrt{2π}σ + \frac{(\tilde{\Bspec}-\hat{\Bspec}_{a}(λ; T))^2}{2σ^2} \,.$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The variables $T$ and $σ$ are model parameters – $T$ is a parameter of the physical model itself, and $σ$ of the observation model we assumed. To get the fairest comparison between models, we fit $σ$ to the data.

::::{important}
Observation models – like the Gaussian noise we assume here – are added by necessity: we mostly care about the physical model (here $\Bspec_{\mathrm{RJ}}$ or $\Bspec_{\mathrm{P}}$), but since that doesn’t predict the data perfectly, we need the observation model to account for discrepancies. Therefore, barring a perfect physical model, a candidate model is generally composed of both:
$$\text{candidate model} = \text{physical model} + \text{observation model} \,.$$
(If the inputs to the physical model are also unknown, then the r.h.s. may also have a third _input model_ term.)

The concrete takeaway is that **to define the candidate models, we need to determine the value not only of $T$ but also of $σ$.** Both of these must be fitted.

Important also is to tailor the observation model to the specifics of the system  – consider for example the difference between testing a model using 1000 measurements from the same device, versus using 1000 single measurements from 1000 different devices. In the latter case, it might make more sense to use a posterior probability as an observation model.

:::{admonition} On the appropriateness of maximum likelihood parameters
:class: dropdown

For simple models with a convex likelihood, like the one in this example, fitting parameters by maximizing the likelihood is often a good choice. This is however not always the case, especially with more complex models: these tend to have a less regular likelihood landscape, with sharp peaks and deep valleys. Neural networks are one notorious example of a class of models with very sharp peaks in their likelihood.

To illustrate one type of situation which can occur, consider the schematic log likelihood sketched below, where the parameter $θ$ has highest likelihood when $θ=3$. However, if there is uncertainty on that parameter, then the steep decline for $θ>3$ would strongly penalize the expected risk. In contrast, the value $θ=1$ is more robust against variations, and its expected risk less strongly penalized.

```{glue:figure} fig_sensitivity_max_likelihood
```

This sensitivity of the risk is exactly what the EMD criterion uses to assess candidate models. 
However it only does so for the selected candidate models – if we omit to include the best models in our candidate pool, the EMD won’t find them. Thus we should only use maximum likelihood parameters if we expect them to yield the best models.
:::

::::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
xarr = np.linspace(-6, 6)
yarr = stats.norm.logpdf(xarr, loc=-2, scale=1.5)/2 + stats.cauchy.logpdf(xarr, loc=3, scale=0.05)/2
curve = hv.Curve(zip(xarr, yarr), kdims=["θ"], vdims=["log likelihood"])
curve.opts(hooks=[viz.no_yticks_hook, viz.despine_hook, viz.no_spine_hook("left")],
           backend="matplotlib")
_kdims = ["θ", "log likelihood"]
θmax=2.96; θrobust=-1  # Obtained by inspecting xarr and np.diff(yarr)
fig = hv.VLine(θrobust, kdims=_kdims) * hv.VLine(θmax, kdims=_kdims) * curve
fig.opts(hv.opts.VLine(color="#BBBBBB", linestyle="dashed", linewidth=1.5))
glue("fig_sensitivity_max_likelihood", fig)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Loss functions $q$ typically only require the observed data as argument, so depending whether `data_model()` returns $x$ and $y$ separately as `x` and `y` or as single object `xy`, the signature for $q$ should be either

    q(x, y)

or

    q(xy)

Additional parameters can be fixed at definition time.

In this notebook we use the `q(x, y)` signature.

```{code-cell} ipython3
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
---
@dataclass(frozen=True)
class Q:
    """Loss of a model assuming Gaussian observation noise.

    :param:candidate_model: One of the phys_model labels; will be passes `CandidateModel`.
    :param:σ: The standard deviation of the assumed Gaussian noise.
        Lower values mean that deviations between data and model are
        less strongly penalized.
    """
    candidate_model: str
    σ              : float|PintQuantity=1.*Bunits   # Used in fitting function, so must
    T              : float|PintQuantity=data_T      # support plain floats as well

    def __post_init__(self):
        # If parameters were given as plain floats, convert to default units
        if not isinstance(self.σ, pint.Quantity):
            object.__setattr__(self, "σ", self.σ * Bunits)   # IMO bypassing frozen dataclass in __post_init__ is acceptable
        if not isinstance(self.T, pint.Quantity):
            object.__setattr__(self, "T", self.T * ureg.kelvin)
        # We use σ in logpdf, which needs a plain float. Might as well convert it now
        object.__setattr__(self, "σ", self.σ.to(Bunits).magnitude)
    def __call__(self, λ_B):
        λ, Bdata = λ_B
        physmodel = CandidateModel(self.candidate_model, T=self.T)
        _, Btheory = physmodel(λ)
        return -stats.norm.logpdf((Bdata - Btheory).m, loc=0, scale=self.σ)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_uv_candidate-models)=
## Candidate models

+++ {"editable": true, "slideshow": {"slide_type": ""}}

As mentioned above, the observation model parameters (here $σ$) are also part of the candidate models, so to construct candidate models we also need to set $σ$. We do this by maximizing the likelihood of $σ$.

:::{admonition} The true observation model is not always the best
:class: hint
It would actually be a poor choice to use the log likelihood of the true Poisson noise to compute the risk: since the Poisson distribution is discrete, any discrepancy between model and data would lead to an infinite risk.
In general, *even when the data are generated with **discrete** noise, candidate models must be assessed with a **continuous** noise model*. Likewise, an observation model with **finite support** (such as a *uniform* distribution) is often a poor choice, since data points outside that support will also cause the risk to diverge.

On the flip side, some distributions are perfectly fine for fitting but make things more challenging when used to generate data. For example, when estimating the expected negative log likelihood from samples, the number of samples required is about 10x greater if they are generated from a Cauchy compared to a Normal distribution. 
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{hint}
:class: margin

Fitting $\log σ$ and $\log T$ is justified on ontological grounds: they are positive quantities, and we are fitting their *scale* as much as we are their precise value. But fitting logarithms is hugely beneficial for numerical stability.

For similar reasons, we use priors to regularize the fit and prevent runoff towards values like 5 K or $10^{10} K$. This is also easily justified by the fact that an experimenter would have *some* idea of the temperature of their black body source and noise of their device.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
FitResult = namedtuple("FitResult", ["σ", "T"])
def fit_gaussian_σT(data) -> Dict[str, FitResult]:
    """
    The candidate models depend on the temperature T and use a Gaussian
    observation model with std dev σ to compute their risk.
    Both T and σ are chosen by maximizing the likelihood.
    """
    
    fitted_σT = Dict()

    log2_T0    = np.log2(data_T.m)
    priorT_std = 12  # |log₂ 4000 - log₂ 5000| ≈ 12
    def f(log2_σT, candidate_model, _λ_B):
        σ, T = 2**log2_σT
        risk = Q(candidate_model, σ, T=T)(_λ_B).mean()
        priorσ = 2**(-log2_σT[0]/128)  # Soft floor on σ so it cannot go too low
        priorT = (log2_σT[1] - log2_T0)**2 / (2*priorT_std**2)  
        return risk + priorσ + priorT
    
    res = optimize.minimize(f, np.log2([1e-4, data_T.m]), ("Rayleigh-Jeans", data), tol=1e-5)
    σ, T = 2**res.x
    fitted_σT.RJ = FitResult(σ*Bunits, T*K)
    
    res = optimize.minimize(f, np.log2([1e-4, data_T.m]), ("Planck", data))
    σ, T = 2**res.x
    fitted_σT.Planck = FitResult(σ*Bunits, T*K)

    return fitted_σT
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fitted = fit_gaussian_σT((λ_data, B_data))
candidate_models = Dict({
    "Rayleigh-Jeans": utils.compose(gaussian_noise(0, fitted.RJ.σ), 
                                    CandidateModel("Rayleigh-Jeans", T=fitted.RJ.T)),
    "Planck": utils.compose(gaussian_noise(0, fitted.Planck.σ),
                            CandidateModel("Planck", T=fitted.Planck.T))
})
Qrisk = Dict({
    "Rayleigh-Jeans": Q(candidate_model="Rayleigh-Jeans", σ=fitted.RJ.σ),
    "Planck"        : Q(candidate_model="Planck",         σ=fitted.Planck.σ)
})
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{dropdown} We could alternatively use moment matching, which yields similar results.

In that case, since
$$\operatorname{Var}\biggl[\frac{1}{s} \Poisson(s \Bspec) \biggr] = \frac{\Bspec}{s} \,,$$
we would either set $σ$ to
$$σ = \sqrt{\frac{\langle \Bspec \rangle_{\text{data}}}{s}} \approx 3\times 10^{-5} \mathrm{kW}/\mathrm{m}^2 \cdot \mathrm{nm} \cdot \mathrm{sr}$$
or
$$σ(\Bspec) = \sqrt{\frac{\Bspec}{s}} \approx 3\times 10^{-5} \mathrm{kW}/\mathrm{m}^2 \cdot \mathrm{nm} \cdot \mathrm{sr} \,.$$
The first expression gives very similar results than the maximum likelihood fit. The second will remain more accurate further from the data used for fitting. In both cases of course this presumes that we know the parameters of the data generation process.

```python
σ = np.sqrt(B_data.mean()/data_noise_s)

candidate_models = Dict(
    RJ     = utils.compose(gaussian_noise(0, σ), CandidateModel("Rayleigh-Jeans", T=data_T)),
    Planck = utils.compose(gaussian_noise(0, σ), CandidateModel("Planck", T=data_T))
)
Qrisk = Dict(
    RJ    =Q(candidate_model="Rayleigh-Jeans", σ=σ),
    Planck=Q(candidate_model="Planck",         σ=σ)
)
```
However, moment matching isn’t as reliable when we don’t know the data generation model.
Therefore it does not work as well during calibration experiments.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
rng = utils.get_rng("uv", "compare ML vs moment matching")
σ_ml = fitted.Planck.σ.m  # max likelihood
σ_mm = np.sqrt(B_data.mean()/data_noise_s).m  # moment matched
_, B = poisson_noise(s=data_noise_s, B0=0*Bunits, rng=rng)((None, B_data))
ξ = (B - B_data).m

centers = np.sort(np.unique(ξ)); Δ = np.diff(centers).mean()
centers = np.concatenate((centers[:1], centers[1:][np.diff(centers)>Δ]))  # Combine values which are actually very close
edges = np.concatenate((centers[:1] - np.diff(centers)[0],
                        (centers[:-1] + centers[1:])/2,
                        centers[-1:] + np.diff(centers)[-1:]))

bins, edges = np.histogram(ξ, bins=edges, density=True)
pois = hv.Histogram((bins, edges), kdims="ξ", label="data").opts(xlabel="deviation from theory")
ξarr = np.linspace(edges[0], edges[-1])
norm_maxL  = hv.Curve(zip(ξarr, stats.norm(0, σ_ml).pdf(ξarr)), label="max likelihood")
norm_match = hv.Curve(zip(ξarr, stats.norm(0, σ_mm).pdf(ξarr)), label="moment matched")
fig = pois*norm_maxL*norm_match
fig.opts(
    hv.opts.Histogram(line_color=None, fill_color=colors.data, backend="bokeh"),
    hv.opts.Curve(f"Curve.{sanitize('max likelihood')}", backend="bokeh",
                  color=config.figures.colors.vibrant.orange),
    hv.opts.Curve(f"Curve.{sanitize('moment matched')}", backend="bokeh",
                  color=config.figures.colors.vibrant.blue, line_dash="dashed"),
    hv.opts.Overlay(width=500, height=300, legend_position="top", backend="bokeh")
)
hv.output(fig, backend="bokeh")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{hint}
:class: margin

`utils.compose(g, f)` works like $g \circ f$.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Example spectra

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, hide-input]
---
def plot_data(L, T=data_T, λmin=data_λ_min, λmax=data_λ_max, s=data_noise_s, B0=0*Bunits):
    """
    Alternative signatures:
    
    - plot_data(Dataset)
    - plot_data(L, utils.compose[Obsmodel, DataModel])
    - plot_data(L, DataModel, s=s, B0=B0)
    """
    if isinstance(L, Dataset):
        𝒟 = L
        del L, T, λmin, λmax, s, B0  # Ensure no code accidentally uses these variables
    elif isinstance(T, utils.compose):
        𝒟 = Dataset("data", L, T[1].λ_min, T[1].λ_max, T[0].s, T[1].T, T[0].B0)
        del T, λmin, λmax, s, B0     # Ensure no code accidentally uses these variables
    elif isinstance(T, DataModel):
        𝒟 = Dataset("data", L, T.λ_min, T.λ_max, s, T.T, B0)
        del T, λmin, λmax            # Ensure no code accidentally uses these variables
    else:
        𝒟 = Dataset("data", L, λmin, λmax, s, T, B0)
    λ_data, B_data = 𝒟.get_data()
    
    candidate_spectra = Dict(
        RayleighJeans = CandidateModel("Rayleigh-Jeans", T=𝒟.T)(λ_data),
        Planck        = CandidateModel("Planck", T=𝒟.T)(λ_data)
    )
    
    scat_data   = hv.Scatter({"λ":λ_data.m, "B":B_data.m}, kdims=[dims.λ], vdims=[dims.B],
                              label="observed data")
    λ, B = candidate_spectra.RayleighJeans
    curve_RJ     = hv.Curve({"λ":λ.m, "B":B.m}, kdims=[dims.λ], vdims=[dims.B],
                            group="candidate", label="Rayleigh-Jeans")
    λ, B = candidate_spectra.Planck
    curve_Planck = hv.Curve({"λ":λ.m, "B":B.m}, kdims=[dims.λ], vdims=[dims.B],
                            group="candidate", label="Planck")
    
    return scat_data * curve_RJ * curve_Planck
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, hide-input, active-ipynb]
---
Tlist = [data_T]
λmin_list = [data_λ_min]
slist = [data_noise_s]
#B0list = [0*Bunits]
B0list = [data_B0]
# Uncomment the lines below to check the bounds of the calibration distribution
# Uncomment also the line at the bottom of this cell
# Tlist = [1000, 3000, 5000]*K
# λmin_list = [5, data_λ_min]*μm
# slist = np.array([2**8, 2**6, 2**3, 1])*data_noise_s
#B0list = 1e-4 * np.array([-0.025, 0, 0.025]) * Bunits

panelA = hv.HoloMap({(T.m, λmin.m, s.m, B0.m): plot_data(L_small, T=T, λmin=λmin, s=s, B0=B0)
                     for T in Tlist for λmin in λmin_list for s in slist for B0 in B0list},
                    kdims=["T", "λmin", "s", "B₀"])
panelB = hv.HoloMap({(T.m, λmin.m, s.m, B0.m): plot_data(L_large, T=T, λmin=λmin, s=s, B0=B0)
                     for T in Tlist for λmin in λmin_list for s in slist for B0 in B0list},
                    kdims=["T", "λmin", "s", "B₀"])

fig = panelA + panelB
xticks = [14, 16, 18, 20, 22, 24, 26, 28, 30]
fig.opts(
    hv.opts.Scatter(edgecolor="none", facecolors=colors.data,
                    hooks=[viz.set_xticks_hook(xticks), viz.despine_hook],
                    backend="matplotlib"),
    hv.opts.Scatter(color=colors.data, xticks=xticks, backend="bokeh"),
    hv.opts.Curve(f"Candidate.{sanitize('Rayleigh-Jeans')}", color=colors.RJ, backend="matplotlib"),
    hv.opts.Curve(f"Candidate.{sanitize('Rayleigh-Jeans')}", color=colors.RJ, backend="bokeh"),
    hv.opts.Curve("Candidate.Planck", color=colors.Planck, linestyle="dashed", backend="matplotlib"),
    hv.opts.Curve("Candidate.Planck", color=colors.Planck, line_dash="dashed", backend="bokeh"),
    hv.opts.Layout(hspace=0.05, vspace=0.05, fontscale=1.3,
                   fig_inches=config.figures.matplotlib.defaults.fig_inches,
                   sublabel_position=(0.3, .8), sublabel_format="({alpha})",
                   backend="matplotlib")
)#.opts(fig_inches=config.figures.matplotlib.defaults.fig_inches, backend="matplotlib")
panelB.opts(
    hv.opts.Overlay(show_legend=False),
)
panelA.opts(
    hv.opts.Curve(hooks=[viz.xaxis_off_hook]),  # Have to do this last to leave panelB untouched
)

#hv.output(fig.opts(legend_position="right", width=500, clone=True, backend="bokeh"), backend="bokeh")
fig.cols(1)

# Use this when checking calibration distribution
#panelA.opts(framewise=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
viz.save(fig, config.paths.figures/"uv_example-spectra.pdf")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
# Rearrange figure for HTML output
panelA.opts(hv.opts.Curve(hooks=[]))  # Make xaxis visible again
panelB.opts(hv.opts.Curve(hooks=[viz.yaxis_off_hook]))  # Hide yaxis instead
figrow = panelA * hv.Text(30, 0.00055, f"$s = {viz.format_pow10(data_noise_s, format='latex')}$\n\n$L = {viz.format_pow2(L_small, format='latex')}$", halign="right") \
         + (panelB * hv.Text(30, 0.00055, f"$s = {viz.format_pow10(data_noise_s, format='latex')}$\n\n$L = {viz.format_pow2(L_large, format='latex')}$", halign="right")).opts(
             show_legend=False)
figrow.opts(
    hv.opts.Layout(hspace=0.05, vspace=0.05, fontscale=1.3,
                   fig_inches=7.5/2,
                   sublabel_position=(0.3, .8), sublabel_format="({alpha})",
                   backend="matplotlib")
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
viz.save(figrow, config.paths.figures/"uv_example-spectra.svg")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Standard error of fits

To get an estimate of the fitting error for $T$, we repeat the fit multiple times and use a bootstrap estimate.
Fits also estimate $σ$, the standard deviation of the Gaussian noise process assumed by candidate models.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
def f(log2σ_log2T, phys_model, data):
    return Q(phys_model, *(2**log2σ_log2T))(data).mean()
def fit(phys_model, data_model, L, rng):
    res = optimize.minimize(f, [-16., 12.], (phys_model, data_model(L, rng=rng)))
    assert res.success
    return 2**res.x
def do_fits(phys_model, data_model, L):
    rng = utils.get_rng("uv", "T fit", L)
    return np.array(
        [fit(phys_model, data_model, L, rng)
         for _ in tqdm(range(Nfits), f"{phys_model} - {L=}")])
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, active-ipynb]
---
data_model = observed_dataset.data_model
fit_results = {(phys_model, L): do_fits(phys_model, data_model, L)
               for phys_model in ["Rayleigh-Jeans", "Planck"]
               for L in [L_small, L_large]}
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-cell]
---
df = pd.DataFrame({(phys_model, L, varname): values
                  for (phys_model, L), results in fit_results.items()
                  for varname, values in zip(["σ", "T"], results.T)})
df
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
fit_stats = pd.DataFrame({"mean": df.mean(axis="index"),
              "std": df.std(axis="index")}).T
fit_stats.style.format(partial(viz.format_scientific, sig_digits=4, min_power=5))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Candidate model PPFs

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} Remark: Obtaining the PPF from samples is almost always easiest 
:class: dropdown

Even though the error model here is just unbiased Gaussian noise, the PPF of the *loss* (i.e. the negative log likelihood) is still non-trivial to calculate. To see why, let’s start by considering how one might compute the CDF. Relabeling $y$ to be the *difference* between observation and prediction, the (synthetic/theoretical) data are distributed as $y \sim \mathcal{N}(0, σ)$. With the risk $q = \log p(y)$, we get for the CDF of $q$:
$$Φ(q) = \int_{\log p(y) < q} \hspace{-2em}dy\hspace{1em} p(y) \log p(y) \,.$$
Substituting the probability of the Gaussian $p(y) = \frac{1}{\sqrt{2πσ}} \exp(-y^2/2σ^2)$, follows then
$$
Φ(q) = \int_{y^2 >  {-2σ^2( q + \log \sqrt{2 π} σ})} \hspace{-5em}dy\hspace{4em}
    \frac{1}{\sqrt{2πσ}} \exp(-y^2/2σ^2)
    \biggl[ -\log \sqrt{2π}σ  - \frac{y^2}{2σ^2} \biggr]
    \,.
$$
The result can be written in terms of error functions (by integrating the second term by parts to get rid of the $y^2$), but it’s not particularly elegant. And then we still need to invert the result to get $q(Φ)$.

A physicist may be willing to make simplifying assumptions, but at that point we might as well use the approximate expression we get by estimating the PPF from samples.

All this to say that in the vast majority of cases, we expect that the most convenient and most accurate way to estimate the PPF will be to generate a set of samples and use `make_empirical_risk_ppf`.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In order to create a set of risk samples for each candidate model, we define for each a generative model which includes the expected noise (in this case additive Gaussian noise with parameter $σ$ fitted above). For each model, we then generate a **synthetic** dataset, evaluate the risk $q$ at every data point, and get a distribution for $q$. We represent this distribution by its quantile function, a.k.a. percent point function (PPF).

:::{hint}
:class: margin
Note how we use different random seeds for each candidate model to avoid accidental correlations.
:::

```{code-cell} ipython3
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
---
def generate_synth_samples(model: CandidateModel, L_synth: int=L_synth, λmin: float=data_λ_min, λmax: float=data_λ_max):
    rng = utils.get_rng("uv", "synth_ppf", model[1].phys_model)
    λarr = np.linspace(λmin, λmax, L_synth)
    return model(λarr, rng=rng)
```

```{code-cell} ipython3
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
---
synth_ppf = Dict({
    "Rayleigh-Jeans": emd.make_empirical_risk_ppf(
        Qrisk["Rayleigh-Jeans"](generate_synth_samples(candidate_models["Rayleigh-Jeans"]))),
    "Planck": emd.make_empirical_risk_ppf(
        Qrisk["Planck"](generate_synth_samples(candidate_models["Planck"])))
})
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
Φarr = np.linspace(0, 1, 200)
curve_synth_ppfRJ = hv.Curve(zip(Φarr, synth_ppf["Rayleigh-Jeans"](Φarr)), kdims=dims.Φ, vdims=dims.q, group="synth", label="synth PPF (Rayleigh-Jeans)")
curve_synth_ppfP = hv.Curve(zip(Φarr, synth_ppf["Planck"](Φarr)), kdims=dims.Φ, vdims=dims.q, group="synth", label="synth PPF (Planck)")
fig_synth = curve_synth_ppfRJ * curve_synth_ppfP
# Plot styling
curve_synth_ppfRJ.opts(color=colors.RJ)
curve_synth_ppfP.opts(color=colors.Planck, linestyle="dashdot")
fig_synth.opts(hv.opts.Curve(linewidth=3, backend="matplotlib"),
               hv.opts.Overlay(show_title=False, legend_position="top_left"),
               hv.opts.Overlay(legend_position="right", fontscale=1.75),
               hv.opts.Overlay(fig_inches=4, backend="matplotlib"))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The EMD criterion works by comparing a *synthetic PPF* with a *mixed PPF*. The mixed PPF is obtained with the same risk function but evaluated on the actual data.
- If the theoretical models are good, differences between synthetic and mixed PPFs can be quite small. They may only be visible by zooming the plot. This is fine – in fact, models which are very close to each other are easier to calibrate, since it is easier to generate datasets for which either model is an equally good fit.
- Note that the mixed PPF curves are below the synthetic ones at low. Although there are counter-examples (e.g. if a model overestimates the variance of the noise), this is generally expected, especially if models are fitted by minimizing the expected risk.

```{code-cell} ipython3
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
---
mixed_ppf = Dict({
    "Rayleigh-Jeans": emd.make_empirical_risk_ppf(Qrisk["Rayleigh-Jeans"]((λ_data, B_data))),
    "Planck"        : emd.make_empirical_risk_ppf(Qrisk["Planck"]((λ_data, B_data)))
})
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
curve_mixed_ppfRJ = hv.Curve(zip(Φarr, mixed_ppf["Rayleigh-Jeans"](Φarr)), kdims=dims.Φ, vdims=dims.q, group="mixed", label="mixed PPF (Rayleigh-Jeans)")
curve_mixed_ppfP = hv.Curve(zip(Φarr, mixed_ppf["Planck"](Φarr)), kdims=dims.Φ, vdims=dims.q, group="mixed", label="mixed PPF (Planck)")
fig_emp = curve_mixed_ppfRJ * curve_mixed_ppfP
fig_emp.opts(hv.opts.Curve("mixed", linestyle="dashed", backend="matplotlib"),
             hv.opts.Curve("mixed", line_dash="dashed", backend="bokeh"))
curve_mixed_ppfRJ.opts(color=colors.RJ)
curve_mixed_ppfP.opts(color=colors.Planck)

panel = fig_synth * fig_emp
fig = panel# + panel.redim.range(risk=(0.5, 1.5))
fig.opts(hv.opts.Overlay(fontscale=1.75, fig_inches=4, legend_position="upper_left", backend="matplotlib"),
         hv.opts.Overlay(width=600, legend_position="top_left", backend="bokeh"),
         hv.opts.Layout(sublabel_format=""))
#hv.output(fig.redim.range(q=(-10, 200)), backend="bokeh")
hv.output(fig, backend="bokeh")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

::::{admonition} Technical validity condition (WIP)
:class: important dropdown

:::{caution} Something about this argument feels off; I need to think this through more carefully.
:::

We can expect that the accuracy of a quantile function estimated from samples will be poorest at the extrema near $Φ=0$ and $Φ=1$: if we have $L$ samples, than the "poorly estimated regions" are roughly $[0, \frac{1}{L+1})$ and $(\frac{L}{L+1}, 1]$. Our goal ultimate is to estimate the expected risk by computing the integral $\int_0^1 q(Φ)$. The contribution of the low extremum region will scale like
$$ \int_0^{1/L} q(Φ) dΦ \,,$$
and similarly for the high extremum region. Since we the estimated $q$ function is unreliable within these regions, we their contribution to the full integral to become negligible once we have enough samples:
$$\int_0^{1/L} q(Φ) dΦ \approx q\Bigl(\frac{1}{L}\Bigr) \cdot \frac{1}{L} \xrightarrow{L\to\infty} 0 \,.$$
In other words, $q$ may approx $\pm \infty$ at the extrema, but must do so at a rate which is sublinear.

Interestingly, low-dimensional models like the 1-d examples studied in this work seem to be the most prone to superlinear growth of $q$. This is because on low-dimensional distributions, the mode tends to be included in the typical set, while the converse is true in high-dimensional distributions (see e.g. chapter 4 of {cite:t}`mackayInformationTheoryInference2003`). Nevertheless, we found that even in this case, the EMD distribution seems to assign plausible, and most importantly stable, distributions to the expected risk $R$.

Where things get more dicey is with distributions with no finite moments. For example, if samples are generated by drawing from a Cauchy distribution, then the value of the function in the “poorly estimated region” remains, because as we draw more samples, we keep getting new samples with such enormous risk that they outweigh all the accumulated contributions up to that point. One solution in such cases may simply be to use a risk function which is robust with regards to outliers.
::::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{hint}

We recommend always inspecting quantile functions visually. For examples, if the risk function is continuous, then we expect the PPF to be smooth (since it involves integrating the risk) – if this isn’t the case, then we likely need more samples to get a reliable estimate
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def get_ppfs(_dataset, _fitted: None|Dict[str, FitResult]=None) -> tuple[Dict[str, emd.interp1d], Dict[str, emd.interp1d]]:
    """
    Returns a total of four quantile functions (PPFs) organized into two dictionaries:
    
    - The first contains mixed PPFs.
    - The second contains synthetic PPFs.

    If the fitted σ and T are not given through `_fitted`, they are obtained with
    a call to `fit_gaussian_σT`.

    Dictionary keys are the labels used to identify the models.
    """
    _data = _dataset.get_data()
    if _fitted is None:
        _fitted = fit_gaussian_σT(_data)
    _candidate_models = Dict({
        "Rayleigh-Jeans": utils.compose(gaussian_noise(0, _fitted.RJ.σ),
                                      CandidateModel("Rayleigh-Jeans", T=_fitted.RJ.T)),
        "Planck"       : utils.compose(gaussian_noise(0, _fitted.Planck.σ),
                                      CandidateModel("Planck", T=_fitted.Planck.T))
    })
    _Qrisk = Dict({
        "Rayleigh-Jeans": Q(candidate_model="Rayleigh-Jeans", σ=_fitted.RJ.σ),
        "Planck": Q(candidate_model="Planck",         σ=_fitted.Planck.σ)
    })

    _synth_ppf = Dict({
        "Rayleigh-Jeans": emd.make_empirical_risk_ppf(
            _Qrisk["Rayleigh-Jeans"](
                generate_synth_samples(_candidate_models["Rayleigh-Jeans"],
                                       λmin=_dataset.λmin, λmax=_dataset.λmax))),
        "Planck": emd.make_empirical_risk_ppf(
            _Qrisk["Planck"](
                generate_synth_samples(_candidate_models["Planck"],
                                       λmin=_dataset.λmin, λmax=_dataset.λmax)))
    })

    _mixed_ppf = Dict({
        "Rayleigh-Jeans": emd.make_empirical_risk_ppf(
            _Qrisk["Rayleigh-Jeans"](_data)),
        "Planck": emd.make_empirical_risk_ppf(
            _Qrisk["Planck"](_data))
    })

    return _mixed_ppf, _synth_ppf
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## How many samples do we need ?

Samples are used ultimately to estimate the expected risk. This is done in two ways:

- By integrating the PPF.
- By directly averaging the risk of each sample.

When computing the $\Bemd{AB;c}$ criterion we use the first approach. During calibration we compare this with $\Bconf{AB}$, which uses the second approach. When computing $\Bconf{AB}$, we ideally use enough samples to reliably determine which of $A$ or $B$ has the highest expected risk.

In the figure below we show the expected risk as a function of the number of samples, computed either by constructing the PPF from samples and then integrating it (left) or averaging the samples directly (right). 

The takeaway from this verification is that a value `Linf=10000` ($\approx 2^{13}$) should be appropriate for the calibration.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
Rvals_ppf = {"Rayleigh-Jeans": {}, "Planck": {}}
Rvals_avg = {"Rayleigh-Jeans": {}, "Planck": {}}
for model in Rvals_ppf:
    for _L in np.logspace(2, 6, 30):  # log10(40000) = 4.602
        _L = int(_L)
        if _L % 2 == 0: _L+=1  # Ensure L is odd
        # q_list = Qrisk[model](data_model(_L, rng=nsamples_rng))
        q_list = Qrisk[model](
            replace(observed_dataset, L=_L, purpose=f"n samples ppf - {model}")
            .get_data())
        ppf = emd.make_empirical_risk_ppf(q_list)
        Φ_arr = np.linspace(0, 1, _L+1)
        Rvals_ppf[model][_L] = scipy.integrate.simpson(ppf(Φ_arr), Φ_arr)
        Rvals_avg[model][_L] = q_list.mean()
```

:::{note}
:class: margin

Although these curves look identical, they really computed in two different manners. This similarity confirms that estimating the expected risk by empirically constructing the quantile function (PPF) and then integrating it is just as accurate as averaging samples.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
curves = {"ppf": [], "avg": []}
for model in Rvals_ppf:
    curves["ppf"].append(
        hv.Curve(list(Rvals_ppf[model].items()),
                 kdims=hv.Dimension("L", label="num. samples"),
                 vdims=hv.Dimension("R", label="exp. risk"),
                 label=f"{model} model - true samples"))
    curves["avg"].append(
        hv.Curve(list(Rvals_avg[model].items()),
                 kdims=hv.Dimension("L", label="num. samples"),
                 vdims=hv.Dimension("R", label="exp. risk"),
                 label=f"{model} model - true samples"))
fig_ppf = hv.Overlay(curves["ppf"]).opts(title="Integrating $q$ PPF")
fig_avg = hv.Overlay(curves["avg"]).opts(title="Averaging $q$ samples")
fig_ppf.opts(show_legend=False)
fig = fig_ppf + fig_avg
# Plot styling
fig.opts(
    hv.opts.Overlay(logx=True, fig_inches=5, aspect=3, fontscale=1.7, legend_position="right", backend="matplotlib")
)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code-uv-calibration)=
## Calibration

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Calibration distributions

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Calibration is a way to test that $\Bemd{AB;c}$ actually does provide a bound on the probability that $R_A < R_B$, where the probability is over a variety of experimental conditions.

There is no unique distribution over experimental conditions. Since this example is meant for illustration we define only one distribution, but in normal practice we expect one would define many, and look for a value of $c$ which satisfies all of them. Here again, knowledge of the system being modelled is key: for any given $c$, it is always possible to define an epistemic distribution for which calibration fails – for example, a model which is 99% random noise would be difficult to calibrate against. The question is whether the uncertainty on experimental conditions justifies including such a model in our distribution. Epistemic distributions are how we quantify our uncertainty in experimental conditions, or describe what we think are reasonable variations of those conditions.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The epistemic distribution we use varies the parameters $s$, $\Bspec_0$ and $T$ of the Poisson noise (Eq. {eq}`eq_code_poisson-noise`):
$$\begin{align}
\log_2 \frac{s}{10^5 \, [\Bspec]^{-1}}  &\sim \Unif(0, 8) \\
\frac{\Bspec_0}{[\Bspec]} &\sim \nN(0, (10^4)^2) \\
\frac{T}{[T]} &\sim \Unif(1000, 5000)
\end{align}$$
(Here $[x]$ indicates the units of $x$; units are the same we use elsewhere in this analysis.) The actual data were generated with $s = 10^5 [\Bspec]^{-1}$.

We like to avoid Gaussian distributions for calibration, because often Gaussian are “too nice”, and may hide or suppress rare behaviours. (For example, neural network weights are often not initialized with Gaussians: distributions with heavier tails tend to produce more exploitable initial features.) Uniform distributions are convenient because they can produce extreme values with high probability, without also producing unphysically extreme values. This is of course only a choice, and in many cases a Gaussian calibration distribution can also be justified: in this example we use a Gaussian distribution for $\Bspec_0$ because it actually promotes a more even distribution of $\Bemd{}$. (Datasets with large biases are unambiguous and so all end up in the same $(\Bemd{}, \Bconf{})$ bin.)

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{important}
Calibration is concerned with the transition from "certainty in model $A$" to "equivocal evidence" to "certainty in model $B$". It is of no use to us if all calibration datasets are best fitted with the same model. The easiest way to avoid this is to use the candidate models to generate the calibration datasets, randomly selecting which candidate for each dataset. All other things being equal, the calibration curve will resolve fastest if we select each candidate with 50% probability.

We also need datasets to span the entire range of certainties, with both clear and ambiguous decisions in favour of $A$ or $B$. One way to do this is to start from a dataset we expect to be ambiguous, and identify a control parameter which can reduce that ambiguity. In this example, the calibration datasets are generated with different $λ_{min}$: this is effective at generating a range of certainties, since for high $λ_{min}$ the two models almost perfectly overlap, while for low $λ_{min}$ the differences are unmistakable. Equivalently, we could start from a dataset with near-perfect discriminability, and identify a control parameters which makes the decision problem ambiguous – this is what we do in the [Prinz example](./Ex_Prinz2004.ipynb).

Note that since we use the candidate models to generate the datasets during calibration, we don’t need to know the true data model. We only need to identify a regime where model predictions differ.

We can conclude from these remarks that calibration works best when the models are close and allow for ambiguity. However this is not too strong a limitation: if we are unable to calibrate the $\Bemd{}$ because there is no ambiguity between models, then we don’t need the $\Bemd{}$ to falsify one of them.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
@dataclass(frozen=True)
class FittedCandidateModels:
    """
    Candidate models need to be tuned to a dataset by fitting T and σ.
    This encapsulates that functionality.
    """
    dataset: Dataset

    @property
    @cache
    def fitted(self):  # Fitting is deffered until we actually need it.
        return fit_gaussian_σT(self.dataset.get_data())

    @property
    def Planck(self):
        rng = utils.get_rng(*self.dataset.purpose, "candidate Planck")
        return utils.compose(
            gaussian_noise(0, self.fitted.Planck.σ, rng=rng),
            CandidateModel("Planck", T=self.fitted.Planck.T)
        )
    @property
    def RJ(self):
        rng = utils.get_rng(*self.dataset.purpose, "candidate RJ")
        return utils.compose(
            gaussian_noise(0, self.fitted.RJ.σ, rng=rng),
            CandidateModel("Rayleigh-Jeans", T=self.fitted.RJ.T)
        )

    @property
    def QPlanck(self):
        return Q("Planck", σ=self.fitted.Planck.σ, T=self.dataset.T)
    @property
    def QRJ(self):
        return Q("Rayleigh-Jeans", σ=self.fitted.RJ.σ, T=self.dataset.T)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{hint}
:class: margin

A calibration distribution type doesn’t need to subclass `emd.tasks.EpistemicDist`, but it should be a dataclass satisfying the following:

- Iterating over it yields data models.
  + Iteration must be non-consuming, because we iterate over models multiple times.
- `__len__` is defined.
- All parameters are serializable.[^serializable]
- Created with ``frozen=True``.

To view these requirements in IPython or Jupyter, along with sample code, type `emd.tasks.EpistemicDist??`.

[^serializable]: Serializable variable types include plain data (int, float, string, tuple, dict), as well as any type supported by [Pydantic](https://pydantic-docs.helpmanual.io/) or [SciTyping](https://scityping.readthedocs.io/en/stable/getting_started.html).
:::

:::{hint}
:class: margin

The `Dataset` instance takes care of initializing its RNG based on its `purpose` arguments; here we use `("uv", "calibration", "fit candidates", n)` so each dataset uses a different seed.

We found it much more reliable to send an object which sets its own seed, compared to creating the RNGs already within `EpistemicDist.__iter__`.
:::

```{raw-cell}
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
tags: [active-py]
---
@dataclass(frozen=True)  # frozen allows dataclass to be hashed
class EpistemicDist(emd.tasks.EpistemicDist):
    N: int|Literal[np.inf] = np.inf

    s_p_range   : tuple[int,int] = (13, 20)  # log₂ data_s ≈ 16.6
    B0_std      : PintQuantity   = 1e-4 * Bunits
    T_range     : PintQuantity   = (1000, 5000) * K
    λmin_range  : PintQuantity   = (10, 20) * μm
    λwidth_range: PintQuantity   = (5, 20) * μm
    
    __version__: int       = 5  # If the distribution is changed, update this number
                                # to make sure previous tasks are invalidated
    def get_s(self, rng):
        p = rng.uniform(*self.s_p_range)
        return 2**p * Bunits**-1    # NB: Larger values => less noise
    def get_B0(self, rng):          # NB: sunits would be better, but we did the original runs with Bunits, and changing to sunits would change the task hash
        return self.B0_std * rng.normal()
    def get_T(self, rng):
        return rng.uniform(*self.T_range.to(K).m) * K
    def get_λmin(self, rng):
        return rng.uniform(*self.λmin_range.to(μm).m) * μm
    def get_λwidth(self, rng):
        return rng.uniform(*self.λwidth_range.to(μm).m) * μm

    ## Experiment generator ##

    def __iter__(self):
        rng = utils.get_rng("uv", "calibration")
        n = 0
        while n < self.N:
            n += 1
            dataset = Dataset(
                ("uv", "calibration", "fit candidates", n),
                L    = L_med,          # L only used to fit model candidates. `CalibrateTask` will
                λmin = (λmin:= self.get_λmin(rng)),
                λmax = λmin + self.get_λwidth(rng),
                s    = self.get_s(rng),
                T    = self.get_T(rng),
                B0   = self.get_B0(rng),
                phys_model = rng.choice(["Rayleigh-Jeans", "Planck"])
            )
            # Fit the candidate models to the data
            candidates = FittedCandidateModels(dataset)
            # Yield the data model, candidate models along with their loss functions
            yield emd.tasks.Experiment(
                data_model=dataset,
                candidateA=candidates.Planck, candidateB=candidates.RJ,
                QA=candidates.QPlanck, QB=candidates.QRJ)
```

+++ {"editable": true, "raw_mimetype": "", "slideshow": {"slide_type": ""}}

```python
@dataclass(frozen=True)  # frozen allows dataclass to be hashed
class EpistemicDist(emd.tasks.EpistemicDist):
    N: int|Literal[np.inf] = np.inf

    s_p_range   : tuple[int,int] = (13, 20)  # log₂ data_s ≈ 16.6
    B0_std      : PintQuantity   = 1e-4 * Bunits
    T_range     : PintQuantity   = (1000, 5000) * K
    λmin_range  : PintQuantity   = (10, 20) * μm
    λwidth_range: PintQuantity   = (5, 20) * μm
    
    __version__: int       = 5  # If the distribution is changed, update this number
                                # to make sure previous tasks are invalidated
    def get_s(self, rng):
        p = rng.uniform(*self.s_p_range)
        return 2**p * Bunits**-1    # NB: Larger values => less noise
    def get_B0(self, rng):          # NB: sunits would be better, but we did the original runs with Bunits, and changing to sunits would change the task hash
        return self.B0_std * rng.normal()
    def get_T(self, rng):
        return rng.uniform(*self.T_range.to(K).m) * K
    def get_λmin(self, rng):
        return rng.uniform(*self.λmin_range.to(μm).m) * μm
    def get_λwidth(self, rng):
        return rng.uniform(*self.λwidth_range.to(μm).m) * μm

    ## Experiment generator ##

    def __iter__(self):
        rng = utils.get_rng("uv", "calibration")
        n = 0
        while n < self.N:
            n += 1
            dataset = Dataset(
                ("uv", "calibration", "fit candidates", n),
                L    = L_med,          # L only used to fit model candidates. `CalibrateTask` will
                λmin = (λmin:= self.get_λmin(rng)),
                λmax = λmin + self.get_λwidth(rng),
                s    = self.get_s(rng),
                T    = self.get_T(rng),
                B0   = self.get_B0(rng),
                phys_model = rng.choice(["Rayleigh-Jeans", "Planck"])
            )
            # Fit the candidate models to the data
            candidates = FittedCandidateModels(dataset)
            # Yield the data model, candidate models along with their loss functions
            yield emd.tasks.Experiment(
                data_model=dataset,
                candidateA=candidates.Planck, candidateB=candidates.RJ,
                QA=candidates.QPlanck, QB=candidates.QRJ)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
from Ex_UV import EpistemicDist
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Execution

+++

:::{admonition} How many experiments
:class: hint margin

The function `emd.viz.calibration_plot` attempts to collect results into 16 bins, so making $N$ a multiple of 16 works nicely. (With the constraint that no bin can have less than 16 points.)

For an initial pilot run, we found $N=64$ or $N=128$ to be good numbers. These numbers produce respectively 4 or 8 bins, which is often enough to check that $\Bemd{}$ and $\Bconf{}$ are reasonably distributed and that the epistemic distribution is actually probing the transition from strong to equivocal evidence.
A subsequent run with $N \in \{256, 512, 1024\}$ can then refine and smooth the curve.
:::

+++

:::{margin}
`"microwave"` distribution is used to calibrate for the first row of the criteria comparison table.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
N = 1024
Ωdct = {"infrared": EpistemicDist(),
        "microwave": EpistemicDist(λmin_range   = (20, 1000) * μm,
                                   λwidth_range = (1000, 3000) * μm)
       }
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
tasks = {}
for Ωkey, Ω in Ωdct.items():
    task = emd.tasks.Calibrate(
        reason = f"UV calibration – RJ vs Planck – Gaussian obs. model",
        #c_list = [.5, 1, 2],
        #c_list = [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0],
        #c_list = [2**-8, 2**-6, 2**-4, 2**-2, 1],
        #c_list = [2**-8, 2**-2],
        #c_list = [1, 2**2, 2**4],
        #c_list = [2**-2, 2**-6, 2**-4, 2**-8, 2**-10, 2**-12, 2**-14, 2**-16],
        #c_list = [2**-16, 2**-12, 2**-8, 2**-4, 2**-2, 2**0, 2**2],
        #c_list = [2**-8, 2**-4, 2**0],
        c_list = [2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4],
        #c_list = [2**0],
        # Collection of generative data model
        #data_models = Ω.generate(1000),
        experiments = Ω.generate(N),
        # # Theory models
        # riskA           = Qrisk["Rayleigh-Jeans"],
        # riskB           = Qrisk["Planck"],
        # Calibration parameters
        Ldata = 1024,
        Linf = 12288  # 2¹³ + 2¹²
        #Linf = 32767 # 2¹⁵ - 1
    )
    tasks[Ωkey]=task
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The code below creates task files which can be executed from the command line with the following:

    smttask run --import config <task file>

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
    for key, task in tasks.items():
        if not task.has_run:  # Don’t create task files for tasks which have already run
            Ω = task.experiments
            taskfilename = f"uv_calibration_{key}_N={Ω.N}_c={task.c_list}"
            task.save(taskfilename)
```

### Analysis

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell, skip-execution]
---
from Ex_UV import *
hv.extension("matplotlib", "bokeh")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{dropdown} Workaround to be able run notebook while a new calibration is running
```python
# Use the last finished task
from smttask.view import RecordStoreView
rsview = RecordStoreView()
params = rsview.get('20231029-161729_cf5215').parameters  # Latest run with UV as of 09.11
if "models_Qs" in params.inputs:
    params.inputs.experiments = params.inputs.models_Qs
    del params.inputs["models_Qs"]
task = emd.tasks.Calibrate.from_desc(params)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
task = tasks["infrared"]
assert task.has_run, "Run the calibration from the command line environment, using `smttask run`. Executing it as part of a Jupyter Book build would take a **long** time."
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
calib_results = task.unpack_results(task.run())
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can check the efficiency of sampling by plotting histograms of $\Bemd{}$ and $\Bconf{}$: ideally the distribution of $\Bemd{}$ is flat, and that of $\Bconf{}$ is equally distributed between 0 and 1. Since we need enough samples at every subinterval of $\Bemd{}$, it is the most sparsely sampled regions which determine how many calibration datasets we need to generate. (And therefore how long the computation needs to run.)
Beyond making for shorter compute times, a flat distribution however isn’t in and of itself a good thing: more important is that the criterion is able to resolve the models when it should.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} Hint: Diagnosing $\Bemd{}$ and $\Bconf{}$ histograms
:class: hint dropdown

$\Bemd{}$ distribution which bulges around 0.5.
~ *May* indicate that $c$ is too large and the criterion underconfident.
~ *May also* indicate that the calibration distribution is generating a large number of (`data`, `modelA`, `modelB`) triples which are essentially undecidable. If neither model is a good fit to the data, then their $δ^{\mathrm{EMD}}$ discrepancies between mixed and synthetic PPFs will be large, and they will have broad distributions for the expected risk. Broad distributions overlap more, hence the skew of $\Bemd{}$ towards 0.5.

$\Bemd{}$ distribution which is heavy at both ends.
~ *May* indicate that $c$ is too small and the criterion overconfident.
~ *May also* indicate that the calibration distribution is not sampling enough ambiguous conditions. In this case the answer is *not* to increase the value of $c$ but rather to tighten the calibration distribution to focus on the area with $\Bemd{}$ close to 0.5. It may be possible to simply run the calibration longer until there have enough samples everywhere, but this is generally less effective than adjusting the calibration distribution.

$\Bemd{}$ distribution which is heavily skewed either towards 0 or 1.
~ Check that the calibration distribution is using both candidate models to generate datasets. The best is usually to use each candidate to generate half of the datasets: then each model should fit best in roughly half the cases.
The skew need not be removed entirely – one model may just be more permissive than the other.
~ This can also happen when $c$ is too small.

$\Bconf{}$ distribution which is almost entirely on either 0 or 1.
~ Again, check that the calibration distribution is using both models to generate datasets.
~ If each candidate is used for half the datasets, and we *still* see ueven distribution of $\Bconf{}$, then this can indicate a problem: it means that the ideal measure we are striving towards (true expected risk) is unable to identify that model used to generate the data. In this case, tweaking the $c$ value is a waste of time: the issue then is with the problem statement rather than the $\Bemd{}$ calibration. Most likely the issue is that the loss is ill-suited to the problem:
  + It might not account for rotation/translation symmetries in images, or time dilation in time-series.
  + One model’s loss might be lower, even on data generated with the other model. This can happen with a log posterior, when one model has more parameters: its higher dimensional prior "dilutes" the likelihood. This may be grounds to reject the more complex model on the basis of preferring simplicity, but it is *not* grounds to *falsify* that model. (Since it may still fit the data equally well.)

:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
hists_emd = {}
hists_conf = {}
for c, res in calib_results.items():
    hists_emd[c] = hv.Histogram(np.histogram(res["Bemd"], bins="auto", density=False), kdims=["Bemd"], label="Bemd")
    hists_conf[c] = hv.Histogram(np.histogram(res["Bconf"].astype(int), bins="auto", density=False), kdims=["Bconf"], label="Bconf")
frames = {viz.format_pow2(c): hists_emd[c] * hists_conf[c] for c in hists_emd}
    
hmap = hv.HoloMap(frames, kdims=["c"])
hmap.opts(
    hv.opts.Histogram(backend="bokeh",
                      line_color=None, alpha=0.75,
                      color=hv.Cycle(values=config.figures.colors.light.cycle)),
    hv.opts.Overlay(backend="bokeh", legend_position="top", width=400)
)
hv.output(hmap, backend="bokeh", widget_location="right")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{hint}
:class: margin

Default properties `calibration_plot` can be changed by updating the configuration object `config.emd.viz.matplotlib` (relevant fields are `calibration_curves`, `prohibited_areas`, `discouraged_area`).
As with any HoloViews plot element, they can also be changed on a per-object basis by calling their [`.opts` method](https://holoviews.org/user_guide/Applying_Customizations.html).
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} Hint: Diagnosing calibration curves
:class: hint dropdown

Flat calibration curve
~ This is the most critical issue, since it indicates that $\Bemd{}$ is actually not predictive of $\Bconf{}$ at all. The most common reason is a mistake in the definition of the calibration distribution, where some values are fixed when they shouldn’t be.
  - Remember that the model construction pipeline used on the real data needs to be repeated in full for each experimental condition produced by the calibration distribution. For example, in `EpistemicDist` above we refit both the temperature $T$ and the observation noise $σ$ for each experimental condition generated within `__iter__`.
  - Treat any global used within `EpistemicDist` with particular suspicion, as they are likely to fix values which should be variable.
    To minimize the risk of accidental global variables, you can define `EpistemicDist` in its own separate module.
~ To help investigate issues, it is often helpful to reconstruct conditions that produce the unexpected behaviour. The following code snippet recovers the first calibration dataset for which both `Bemd > 0.9` and `Bconf = False`; the recovered dataset is `D`:
  ```python
  Bemd = calib_results[1.0]["Bemd"]
  Bconf = calib_results[1.0]["Bconf"]
  i = next(iter(i for i in range(len(Bemd)) if Bemd[i] > 0.9))
    
  for j, D in zip(range(i+1), task.models_Qs):
      pass
  assert j == i
  ```

Calibration curve with shortened domain
~ I.e. $\Bemd{}$ values don’t reach 0 and/or 1. This is not necessarily critical: the calibration distribution we want to test may simply not allow to fully distinguish the candidate models under any condition. 
~ If it is acceptable to change the calibration distribution (or to add one to the test suite), then the most common way to address this is to ensure the distribution produces conditions where $\Bemd{}$ can achieve maximum confidence – for example by having conditions with negligeable observation noise.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
c_chosen = 2**-1
c_list = [2**-4, 2**-2, 2**-1, 2**0, 2**1, 2**3]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
_hists_emd = {c: h.relabel(group="Bemd", label=f"$c={viz.format_pow2(c, format='latex')}$")
              for c, h in hists_emd.items() if c in c_list}
for c in c_list:
    α = 1 if c == c_chosen else 0.8
    _hists_emd[c].opts(alpha=α)
histpanel_emd = hv.Overlay(_hists_emd.values())

calib_curves, prohibited_areas, discouraged_areas = emd.viz.calibration_plot(calib_results)

for c, curve in calib_curves.items():
    calib_curves[c] = curve.relabel(label=f"$c={viz.format_pow2(c, format='latex')}$")

for c in c_list:
    α = 1 #if c == c_chosen else 0.85
    w = 3 if c == c_chosen else 2
    calib_curves[c].opts(alpha=α, linewidth=w)

main_panel = prohibited_areas * discouraged_areas * hv.Overlay(calib_curves.select(c=c_list).values())

histpanel_emd.redim(Bemd=dims.Bemd, Bconf=dims.Bconf, c=dims.c)
main_panel.redim(Bemd=dims.Bemd, Bconf=dims.Bconf, c=dims.c)

main_panel.opts(
    legend_position="top_left", legend_cols=1,
    hooks=[viz.despine_hook],
)
histpanel_emd.opts(
    legend_cols=3,
    legend_opts={"columnspacing": .5, "alignment": "center",
                 "loc": "upper center"},
    hooks=[viz.yaxis_off_hook, partial(viz.despine_hook, left=True)]
)

fig = main_panel << hv.Empty() << histpanel_emd.opts(show_legend=True)

# Plot styling
# NB: If we also set the Bokeh options, somehow that corrupts the matplotlib options
fig.opts(
    hv.opts.Curve(color=colors.calib_curves),
    #hv.opts.Curve(color=colors.calib_curves, line_width=3, backend="bokeh"),
    hv.opts.Area(alpha=0.5),
    #hv.opts.Area(alpha=0.5, backend="bokeh"),
    hv.opts.Histogram(color="none", edgecolor="none", facecolor=colors.calib_curves),
    #hv.opts.Histogram(line_color=None, fill_color=colors.calib_curves, backend="bokeh"),
    hv.opts.Overlay(fig_inches=config.figures.defaults.fig_inches,
                    #fig_inches=4,
                    aspect=1,
                    xlabel="$B^{\mathrm{EMD}}$", ylabel="$B^{\mathrm{conf}}$",
                    #show_legend=False,
                    #legend_position="right", legend_cols=1,
                    fontscale=1.3,
                    backend="matplotlib"),
)
fig.opts(backend="matplotlib", fig_inches=5)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
Ω = task.experiments
desc = f"N={Ω.N}"
viz.save(fig, config.paths.figures/f"uv_calibration_{desc}_raw.svg")
# viz.save(fig.opts(fig_inches=5.5, backend="matplotlib"),
#                   config.paths.figures/f"uv_calibration_{desc}.svg")

f"uv_calibration_{desc}"
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Finalized with Inkscape:
- Align histogram axis with curves axis (use vertical lines with 0.8 pt width)
- ~~Improve placement of legends~~
- Put the curve corresponding to `uv_c_chosen` on top. Highlight curve with white surround (2x curve width).
- Remove whitespace

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} Additional comments on choosing $c$
:class: hint

- Both the calibration curve and the $\Bemd{}$ histogram should check out. For example, small values of $c$ may seem to have good statistics on average, but be heavily biased towards the ends at 0 and 1.  
  (In fact $\Bemd{}$ becomes asymptotically equivalent to $\Bconf{}$ when $c$ approaches 0. We don’t want this: the point of $\Bemd{}$ is that it estimates the probability of $\Bconf{}$, without having to average over an ensemble of datasets.)
- Larger values of $c$, in addition to being more conservative, may also take longer to sample: Since they sample a wider variety of quantile curves, it takes more samples to achieve the target accuracy on $R$.
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## EMD model comparison

Based on the figure above, we choose the value {glue:}`uv_c_chosen` to compute the $\Bemd{}$ criterion between models.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
R_samples = Dict({
    "Rayleigh-Jeans": emd.draw_R_samples(mixed_ppf["Rayleigh-Jeans"], synth_ppf["Rayleigh-Jeans"], c=c_chosen),
    "Planck": emd.draw_R_samples(mixed_ppf["Planck"], synth_ppf["Planck"], c=c_chosen),
})
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
Rdists = [hv.Distribution(_Rlst, kdims=[dims.R], label=f"{a}")
          for a, _Rlst in R_samples.items()]
Rcurves = [hv.operation.stats.univariate_kde(dist).to.curve()
           for dist in Rdists]
fig = hv.Overlay(Rdists) * hv.Overlay(Rcurves)
# Plot styling
color_cycle = hv.Cycle(values=[colors.RJ, colors.Planck])
#fig = fig.redim.range(R=(-9.08, -8.89))
fig.opts(
    hv.opts.Distribution(alpha=.3),
    hv.opts.Distribution(facecolor=color_cycle, color="none", edgecolor="none", backend="matplotlib"),
    hv.opts.Curve(color=color_cycle),
    hv.opts.Curve(linestyle="solid", backend="matplotlib"),
    hv.opts.Overlay(legend_position="top_left", legend_cols=1,
                    fontscale=1.3, hooks=[viz.despine_hook], backend="matplotlib")
)
fig.opts(clone=True, backend="matplotlib", fig_inches=5)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell, active-ipynb]
---
viz.save(fig, config.paths.figures/f"uv_Rdists.pdf")
viz.save(fig.opts(fig_inches=5.5, backend="matplotlib", clone=True),
              config.paths.figures/f"uv_Rdists.svg")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

EMD estimates for the probabilities $P(R_a < R_b)$ are best reported as a table:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
df = emd.utils.compare_matrix(R_samples)
df.index = pd.MultiIndex.from_tuples([("a", a) for a in df.index])
df.columns = pd.MultiIndex.from_tuples([("b", b) for b in df.columns])
df.style.format(precision=3)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

For pasting into a MyST file, we reformat as a [list-table](https://jupyterbook.org/en/stable/reference/cheatsheet.html?highlight=list-table#tables):

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution, remove-cell, active-ipynb]
---
print(":::{list-table}")
model_labels = list(R_samples)
print(":header-rows: 1")
print(":stub-columns: 1")
print("")

print(f"* - ")
for a in model_labels:
    print(f"  - {a}")
for a in model_labels:
    print(f"* - {a}")
    for Pab in df.loc[("a", a),:]:
        print(f"  - {Pab:.3f}")
print(":::")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} `compare_matrix` implementation
:class: note dropdown

The `compare_matrix` function provided by `emd_falsify` simply loops through all $(a,b)$ model pairs, and counts the number of $R_a$ samples which are less than $R_b$:

```python
def compare_matrix(R_samples: Dict[str, ArrayLike]) -> pd.DataFrame:
    R_keys = list(R_samples)
    compare_data = {k: {} for k in R_keys}
    for i, a in enumerate(R_keys):
        for j, b in enumerate(R_keys):
            if i == j:
                assert a == b
                compare_data[b][a] = 0.5
            elif j < i:
                compare_data[b][a] = 1 - compare_data[a][b]
            else:
                compare_data[b][a] = np.less.outer(R_samples[a], R_samples[b]).mean()
    return pd.DataFrame(compare_data)
```
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_uv-dataset-and-Rdist-grids)=
## Dependence of $R$ distributions on bandwidth and noise level

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Approximate noise level ranges (low $s$ ⤇ high noise):
- High $λ$: $\frac{s}{[s]} \in [2, 40]$.
- Low $λ$: $\frac{s}{[s]} \in [500, 15000]$.

:::{note}
:class: margin

- The Poisson noise (Eq. {eq}`eq_code_poisson-noise`) has variance $\frac{\Bspec}{s}$, and therefore its standard deviation scales only as $\Bspec^{1/2}$. Since different $λ$ ranges lead to different orders of magnitude for $\Bspec$, if we kept the same $s$ for all rows, we would not see similar amounts of spread. This is why we decrease $s$ in the last row, where wavelength ($λ$) is lower.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def panel_param_iterator(L=L_small, purpose="data", progbar=True):
    #λmin_list = [6. , 1. , .2] * μm
    λmin_list = [6. , 6. , .2] * μm
    λmax_list = [20., 20., 3.] * μm
    B0_list = [2**-10+2**-11, 0, 0] * Bunits
    #s_list = [2**-11, 2**-8, 2**-6] * (10e5 * Bunits**-1)
    
    for λmin, λmax, B0 in tqdm(zip(λmin_list, λmax_list, B0_list), desc="λ", total=len(λmin_list), disable=not progbar):
        if λmin < 2 * μm:
            # 3rd row
            s_list = [2**-11, 2**-9, 2**-7] * (2**12 * Bunits**-1)
            #B0 = 2**-5 * Bunits
            #B0 = (2**-10 + 2**-9) * Bunits
        else:
            # 1st & 2nd row
            s_list = [2**-3 , 2**0 , 2**2]  * (2**12 * Bunits**-1)
            #B0 = (2**-10 + 2**-9) * Bunits
        for s in tqdm(s_list, desc="s", leave=False, disable=not progbar):
            yield Dataset(purpose, L, λmin, λmax, s, T=data_T, B0=B0)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
data_panels = [plot_data(L=D.L, T=D.T, λmin=D.λmin, λmax=D.λmax, s=D.s, B0=D.B0)
               for D in panel_param_iterator()]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb, remove-output]
---
legend_specs = hv.plotting.mpl.LegendPlot.legend_specs
legend_specs["top_orig"] = dict(legend_specs["top"])  # `dict` to force a copy
legend_specs["top"].update({"mode": None, 'bbox_to_anchor': (-0.3, 1.02, 1.0, 0.102)})
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{note}
:class: margin

For the last row we use the $σ$ and $T$ values obtained by fitting the original dataset.
This is because the Rayleigh-Jeans model is so bad in the UV region, that the fit doesn’t converge.
For pedagogical reasons we want to illustrate this extreme case, but in practice the fact that the fit doesn’t converge should be reason enough to reject the Rayleigh-Jeans model.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb, remove-output]
---
          #+ [panel.redim.range(B=(0, 5)) for panel in data_panels[3:6]] \
_panels = [panel.redim.range(B=(0, 0.03)) for panel in data_panels[:3]] \
          + [panel.redim.range(B=(0, 0.03)) for panel in data_panels[3:6]] \
          + [panel.redim.range(B=(0, 7)) for panel in data_panels[6:9]]
for (i, panel), D in zip(enumerate(_panels), panel_param_iterator(progbar=False)):
    if i < 6:
        panel *= hv.Text(10, 0.02, f"$s={viz.format_pow2(D.s.m, format='latex')}$",
                         halign="left")
    else:
        panel *= hv.Text(0.25, 6, f"$s={viz.format_pow2(D.s.m, format='latex')}$",
                         halign="center")
    _panels[i] = panel
data_layout = hv.Layout(_panels)
for i, panel in enumerate(_panels):
    hooks = []
    if i != 0:
        panel.opts(show_legend=False)
    if i == 0:
        panel.opts(legend_position="top", legend_cols=3)
    if i != 6:
        panel.opts(ylabel="")
    if i == 6:
        hooks.append(viz.set_ylabel_hook(f"{dims.B.label} ({dims.B.unit})", loc="bottom"))
    if i != 7:
        panel.opts(xlabel="")
    if i % 3:
        hooks.extend([viz.no_spine_hook("left"), viz.no_yticks_hook])
    if i < 6:
        def major_formatter(x, pos): return f"{x:.3g}"
        def minor_formatter(x, pos): return r"$6$" if x == 6 else r"$20$" if x == 20 else ""
        hooks.extend([viz.hide_minor_ticks_hook,
                      viz.set_minor_xticks_formatter(minor_formatter),
                      viz.set_major_xticks_formatter(major_formatter)])
    else:
        def major_formatter(x, pos): return f"{x:.3g}"
        def minor_formatter(x, pos): return r"$0.2$" if x == 0.2 else r"$3$" if x == 3 else ""
        hooks.extend([viz.set_major_xticks_formatter(major_formatter),
                      viz.set_minor_xticks_formatter(minor_formatter)])
    panel.opts(logx=True, aspect=3, hooks=hooks, fontscale=1.3)
    #_panels[i] = panel
data_layout = hv.Layout(_panels)

data_layout.opts(
    hv.opts.Layout(backend="matplotlib", sublabel_format="({alpha})", sublabel_position=(0.3, 0.7),
                   hspace=0.04, vspace=0.5, fig_inches=7/6),  # full width fig is about 7in wide
    hv.opts.Scatter(backend="matplotlib", axiswise=True,
                    color="#888888", s=3),
    hv.opts.Overlay(axiswise=True),
    hv.opts.Text(axiswise=True),
    hv.opts.Curve(backend="matplotlib", axiswise=True, linewidth=3, alpha=.6),
    hv.opts.Curve(f"Candidate.{sanitize('Rayleigh-Jeans')}", color=colors.RJ),
    hv.opts.Curve(f"Candidate.Planck", color=colors.lighten(.1).Planck),
)
data_layout.cols(3)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-input]
---
data_layout \
    .opts(clone=True, backend="matplotlib", fig_inches=3) \
    .opts(hv.opts.Scatter(backend="matplotlib", s=8))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{hint}
:class: margin

Together, `get_ppfs` and `plot_Rdists` condense almost the entire EMD pipeline to two functions.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
def plot_Rdists(_dataset, c):
    _R_samples = _draw_R_samples(_dataset, c)

    _Rdists = [hv.Distribution(_Rlst, kdims=[dims.R], label=f"{a}")
          for a, _Rlst in _R_samples.items()]
    _Rcurves = [hv.operation.stats.univariate_kde(dist).to.curve()
               for dist in _Rdists]
    _fig = hv.Overlay(_Rdists) * hv.Overlay(_Rcurves)

    return _fig

@memory.cache
def _draw_R_samples(_dataset, c):  # This function mostly exists so it can be cached
    if _dataset.λmax <= 5*μm:
        _mixed_ppf, _synth_ppf = get_ppfs(_dataset, fitted)
    else:
        _mixed_ppf, _synth_ppf = get_ppfs(_dataset)
    return Dict({
        "Rayleigh-Jeans": emd.draw_R_samples(
            _mixed_ppf["Rayleigh-Jeans"], _synth_ppf["Rayleigh-Jeans"], c=c),
        "Planck": emd.draw_R_samples(
            _mixed_ppf["Planck"], _synth_ppf["Planck"], c=c),
    })
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
Rdist_panels = [plot_Rdists(D, c=c_chosen) for D in panel_param_iterator()]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{margin}
In the last row, the two distributions cannot really be plotted together: the Planck distributions is effectively a Dirac delta compared to the Rayleigh-Jeans distributions. To show this, we overlay the Planck distribution with a vertical line.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input, remove-output]
---
_panels = [panel.redim.range(R=(-7, 0), Density=(0, 7), R_density=(0,7)) for panel in Rdist_panels[:3]] \
          + [panel.redim.range(R=(-10, 20), Density=(0, 1.5), R_density=(0,1.5)) for panel in Rdist_panels[3:6]] \
          + [panel.redim.range(R=(-1e17, 6e17), Density=(0, 0.7e-17), R_density=(0,0.7e-17)) for panel in Rdist_panels[6:9]]
hooks = {i: [] for i in range(len(_panels))}
for i, panel in enumerate(_panels):
    if i < 3:
        panel.opts(xticks=[-6, -2], yticks=(0, 4))
    elif 3 <= i < 6:
        panel.opts(xticks=[-5, 15], yticks=(0, 1))
    elif 6 <= i < 9:
        panel *= hv.VLine(panel.Curve.Planck.data["R"].mean()).opts(color=colors.Planck, linewidth=2)
        hooks[i].extend([viz.set_yticks_hook([0, 6e-18], labels=["0", "$6\\times 10^{-18}$"], rotation=90),
                         #viz.set_yticklabels_hook(["0", "$4\\times 10^{-18}$"]),
                         viz.set_xticks_hook([0, 3e17 ], labels=["0", "$3\\times 10^{17}$"])])
                         #viz.set_xticklabels_hook(["0", "$3\\times 10^{17}$"])])
        _panels[i] = panel
    if i != 7:
        panel.opts(xlabel="")
    if i != 3:
        panel.opts(hv.opts.Curve(ylabel=""))
    if i % 3:
        hooks[i].extend([viz.yaxis_off_hook, viz.no_yticks_hook])
for i, hooklist in hooks.items():
    _panels[i].opts(hv.opts.Curve(hooks=hooklist))

Rdist_layout = hv.Layout(_panels)

Rdist_layout.opts(
    hv.opts.Layout(backend="matplotlib", sublabel_format="({alpha})", sublabel_position=(0.4, 0.6),
                   hspace=0.04, vspace=0.1, fig_inches=7/6),  # full width fig is about 7in wide
    hv.opts.Overlay(show_legend=False),
    hv.opts.Distribution(backend="matplotlib", axiswise=True),
    hv.opts.Curve(backend="matplotlib", axiswise=True, linewidth=1),
    # Colors
    hv.opts.Curve(f"Curve.{sanitize('Rayleigh-Jeans')}", color=colors.RJ),
    hv.opts.Distribution(f"Distribution.{sanitize('Rayleigh-Jeans')}", facecolor=colors.RJ, backend="matplotlib"),
    hv.opts.Curve(f"Curve.Planck", color=colors.Planck),
    hv.opts.Distribution(f"Distribution.Planck", facecolor=colors.Planck, backend="matplotlib"),
    hv.opts.Distribution(edgecolor="none", color="none", backend="matplotlib")
)
for i, panel in enumerate(Rdist_layout):
    if 3 <= i < 6:
        panel.redim.range(Density=(0, 2), R_density=(0,2))
Rdist_layout.cols(3)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-input]
---
Rdist_layout.opts(fig_inches=3, clone=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
viz.save(data_layout, config.paths.figures/"uv_dataset-grid.svg")
viz.save(Rdist_layout, config.paths.figures/"uv_Rdist-grid.svg")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(For insertion into the manuscript, these two figures should be combined into one wide figure with Inkscape and saved as *uv_dataset-and-Rdist-grids* (both as *Optimized SVG* and *PDF*).

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Comparison with other criteria

+++ {"editable": true, "slideshow": {"slide_type": ""}}

See [](./Ex_UV_criteria-comparison.ipynb).

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

## Exported notebook variables

These can be inserted into other pages.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
# Assert that all Calibrate tasks use the same values
task_args = SimpleNamespace(Ldata=set(), Linf=set())
for task in tasks.values():
    task_args.Ldata.add(task.Ldata)
    task_args.Linf.add(task.Linf)
assert len(task_args.Ldata) == len(task_args.Linf) == 1
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
glue("uv_c_chosen", f"${viz.format_pow2(c_chosen, 'latex')}$")

glue("data_T", **viz.formatted_quantity(data_T, 0))
glue("data_noise_s", data_noise_s.m, raw_html=viz.format_pow10(data_noise_s.m, 'latex'), # latex b/c used inside $…$ expression
                                    raw_latex=viz.format_pow10(data_noise_s.m, 'latex'))
glue("data_B0", data_B0.m, raw_html=viz.format_scientific(data_B0.m, 2, format='latex'),
                          raw_latex=viz.format_scientific(data_B0.m, 2, format='latex'))
glue("Bunits", f"{Bunits:~P}", raw_latex=f"{Bunits:~Lx}",
     raw_myst=viz.tex_frac_to_solidus(f"{Bunits:~L}"))
    # {:~L} uses \frac{}{}, but we want to use Bunits in a denom, so we replace with {}/{}
#sunits = data_noise_s.units
glue("sunits", f"{sunits:~P}", raw_latex=f"{sunits:~Lx}",
     raw_myst=viz.tex_frac_to_solidus(f"{sunits:~L}"))

glue("L_small", L_small, raw_html=viz.format_pow2(L_small, 'latex'), # latex b/c used inside $…$ expression
                        raw_latex=viz.format_pow2(L_small, 'latex'))
glue("L_med", L_med, raw_html=viz.format_pow2(L_med, 'latex'),
                    raw_latex=viz.format_pow2(L_med, 'latex'))
glue("L_large", L_large, raw_html=viz.format_pow2(L_large, 'latex'),
                        raw_latex=viz.format_pow2(L_large, 'latex'))

glue("Nfits", Nfits)
r = fit_stats[("Rayleigh-Jeans", L_small, "T")]
glue("T-fit_Rayleigh-Jeans_L-small", **viz.formatted_quantity(r["mean"]*K, r["std"]*K, 0))
r = fit_stats[("Rayleigh-Jeans", L_large, "T")]
glue("T-fit_Rayleigh-Jeans_L-large", **viz.formatted_quantity(r["mean"]*K, r["std"]*K, 0))
r = fit_stats[("Planck", L_small, "T")]
glue("T-fit_Planck_L-small", **viz.formatted_quantity(r["mean"]*K, r["std"]*K, 0))
r = fit_stats[("Planck", L_large, "T")]
glue("T-fit_Planck_L-large", **viz.formatted_quantity(r["mean"]*K, r["std"]*K, 0))  

glue("calib_Nsims", N)
glue("calib_Nbins", len(next(iter(calib_curves.values())).data))
Linf = next(iter(task_args.Linf))
Ldata = next(iter(task_args.Ldata))
glue("Linf", f"${viz.format_pow2(Linf, 'latex')}$")
glue("Ldata", f"${viz.format_pow2(Ldata, 'latex')}$")
glue("Lsynth", L_synth, display=True)

glue("calib_sp_low"     , Ω.s_p_range[0], raw_myst=True, raw_latex=True)
glue("calib_sp_high"    , Ω.s_p_range[1], raw_myst=True, raw_latex=True)
glue("calib_B0_std"     , **viz.formatted_quantity(Ω.B0_std))
_s = viz.format_pow10(Ω.B0_std.m); _sl = viz.format_pow10(Ω.B0_std.m, format="latex")
glue("calib_B0_std_inv_m", _s, raw_html=_sl, raw_latex=_sl)
glue("calib_T_low"      , **viz.formatted_quantity(Ω.T_range[0]))
glue("calib_T_high"     , **viz.formatted_quantity(Ω.T_range[1]))
glue("calib_lambdamin_low"   , **viz.formatted_quantity(Ω.λmin_range[0]))
glue("calib_lambdamin_high"  , **viz.formatted_quantity(Ω.λmin_range[1]))
glue("calib_lambdawidth_low" , **viz.formatted_quantity(Ω.λmin_range[0]))
glue("calib_lambdawidth_high", **viz.formatted_quantity(Ω.λmin_range[1]))

glue("color_RJ", color_labels.RJ)
glue("color_Planck", color_labels.Planck)
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

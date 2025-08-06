---
jupytext:
  formats: ipynb,md:myst,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python (emd-paper)
  language: python
  name: emd-paper
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_effect-of-c-on-Rdists)=
# Effect of $c$ on the $R$-distributions in the Prinz model

+++ {"editable": true, "slideshow": {"slide_type": ""}}

> **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from addict import Dict

import multiprocessing as mp
import pickle
import psutil
if __name__ == "__main__":
    mp.set_start_method("spawn")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
import holoviews as hv
hv.extension("matplotlib")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# Hack: For some reason, "spawn" or "forkserver" methods add emd-paper/viz to the path, which breaks some imports
import sys;
for i in reversed([i for i, p in enumerate(sys.path) if "emd-paper/viz" in p]):
    del sys.path[i]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import emdcmp
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{margin}
`emdcmp` library
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from Ex_Prinz2004 import (Q, fit_gaussian_σ, LP_data, phys_models, AdditiveNoise,
                          generate_synth_samples,
                          colors)
from config import config
import utils
import viz
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{margin}
Project code
:::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

First we recreate `synth_ppf` and `mixed_ppf`.
These are stored to disk so that computations are not repeated in subprocesses.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
cache_path = config.paths.data / "prinz_ppfs.pkl"
if not cache_path.exists():

    candidate_models = Dict()
    Qrisk = Dict()
    for a in "ABCD":
        fitted_σ = fit_gaussian_σ(LP_data, phys_models[a], "Gaussian")
        candidate_models[a] = utils.compose(AdditiveNoise("Gaussian", fitted_σ),
                                            phys_models[a])
        Qrisk[a] = Q(phys_model=phys_models[a], obs_model="Gaussian", σ=fitted_σ)

    synth_ppf = Dict({
        a: emdcmp.make_empirical_risk_ppf(Qrisk[a](generate_synth_samples(candidate_models[a])))
        for a in "ABCD"
    })
    mixed_ppf = Dict({
        a: emdcmp.make_empirical_risk_ppf(Qrisk[a](LP_data.get_data()))
        for a in "ABCD"
    })

    with open(cache_path, 'wb') as f:
        pickle.dump((synth_ppf, mixed_ppf), f)

else:
    with open(cache_path, 'rb') as f:
        synth_ppf, mixed_ppf = pickle.load(f)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Sample a set of expected risks ($R$) for each candidate model.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
c_list = [2**-6, 2**-4, 2**-2, 2**0, 2**2, 2**4]
arg_list = [(c, a) for c in c_list
                   for a in "ABCD"]
def draw_R_samples(args):
    c, a = args
    return emdcmp.draw_R_samples(mixed_ppf[a], synth_ppf[a], c=c, relstderr_tol=2**-6)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# In order to run with "spawn" in a Jupyter Notebook, the function needs to be
# defined in a separate module. So we use the one from the .py copy.
from Effect_of_c_on_R_distributions import draw_R_samples
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
cores = min(len(arg_list), psutil.cpu_count(logical=False))
with mp.Pool(cores) as pool:
    R_samples = {c: Dict() for c in c_list}
    for (c, a), res in zip(arg_list, pool.map(draw_R_samples, arg_list)):
        R_samples[c][a] = res
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Convert the samples into a distributions using a kernel density estimate (KDE).

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell, active-ipynb]
---
hv.Layout([
    hv.Overlay([hv.Distribution(samples, kdims=hv.Dimension("R", label="$R$")) for samples in R_samples[c].values()],
               label=f"$c={viz.format_pow2(c, format='latex')}$")
    for c in R_samples
]).cols(1).opts(
    hv.opts.Overlay(aspect=3),
    hv.opts.Distribution(facecolor=colors.LP_candidates, edgecolor=colors.LP_candidates),
    hv.opts.Layout(fig_inches=3.5, shared_axes=False, sublabel_format="", tight=True)
).cols(3)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
xticks      = {2**-6: [4.2  , 4.3, 4.4, 4.5], 2**-4: [4, 4.2, 4.4, 4.6]    , 2**-2: [3.8, 4.0, 4.2, 4.4, 4.6, 4.8], 2**0: [3.5, 4., 4.5, 5.0, 5.5], 2**2: [3,  4,  5, 6, 7,  8], 2**4: [0, 5, 10, 15]}
xticklabels = {2**-6: ["4.2", "", "", "4.5"], 2**-4: ["4.0", "", "", "4.6"], 2**-2: ["", "4.0", "", "",  "","4.8"], 2**0: ["", "4", "", "5", ""],   2**2: ["","4","","","","8"], 2**4: [0, "", "10", ""]}
yticks      = {2**-6: [0, 15, 30]    , 2**-4: [0, 5, 10, 15],      2**-2: [0,    4,  8 ], 2**0: [0, 2, 4]     , 2**2: [0, 1, 2]   , 2**4: [0, 0.5, 1]}
yticklabels = {2**-6: ["0", "", "30"], 2**-4: ["0", "", "", "15"], 2**-2: ["0", "", "8"], 2**0: ["0", "", "4"], 2**2: ["0","","2"], 2**4: ["0","","1"]}

yrange = {2**-6: (0, 40), 2**-4: (0, 16) , 2**-2: (0, 8),
          2**0 : (0, 5) , 2**2 : (0, 2.5),  2**4: (0, 1.2)}

def get_frac_pos(xfrac, yfrac, xrange, yrange):
    """
    Utility for getting the x,y coords within a box, given the box xrange and yrange.
    """
    return (xrange[0] + xfrac*(xrange[1]-xrange[0]),
            yrange[0] + yfrac*(yrange[1]-yrange[0]))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_effect-of-c-on-Rdists_plot)=

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
panels = {c: viz.make_Rdist_fig(R_samples[c], colors = colors.LP_candidates,
                                xticks = xticks[c], xticklabels = xticklabels[c],
                                yticks = yticks[c], yticklabels = yticklabels[c]
                                )
             .redim.range(Density=yrange[c])  # Without this, range("Density") is (NaN, NaN)
          for c in c_list}
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
xrange = {c: panel.range("R") for c, panel in panels.items()}
xrange[2**4] = (0, 15)
c_texts = {2**-6: (0.3, .75, "$c=2^{-6}$"),
           2**-4: (0.3, .75, "$c=2^{-4}$"),
           2**-2: (0.3, .75, "$c=2^{-2}$"),
           2** 0: (0.7, .75, "$c=2^{0}$"),
           2** 2: (0.7, .75, "$c=2^{2}$"),
           2** 4: (0.7, .75, "$c=2^{4}$")}
layout = hv.Layout([((p:=panels[c]) * hv.Text(*get_frac_pos(xfrac, yfrac, xrange[c], yrange[c]),
                                             text))
                    .opts(hooks=[viz.set_xticks_hook(xticks[c]), viz.set_xticklabels_hook(xticklabels[c]), viz.ylabel_shift_hook(5),
                                 viz.set_yticks_hook(yticks[c]), viz.set_yticklabels_hook(yticklabels[c]), viz.xlabel_shift_hook(7),
                                 viz.despine_hook(2)],
                          xlim=xrange[c])
                    for c, (xfrac, yfrac, text) in c_texts.items()])
layout.opts(
    hv.opts.Overlay(aspect=3, show_legend=False),
    hv.opts.Curve(xlabel="", ylabel=""), hv.opts.Distribution(xlabel="", ylabel=""),
    hv.opts.Distribution(facecolor=colors.LP_candidates, edgecolor=colors.LP_candidates),
    hv.opts.Layout(fig_inches=2.5, shared_axes=False, sublabel_format="", tight=False,
                   hspace=0.2, vspace=0.05)
)
layout[4].opts(xlabel="$R$")
layout[0].opts(ylabel="Density")
layout[3].opts(ylabel="Density")
layout.cols(3)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell, active-ipynb]
---
hv.save(layout, config.paths.figures/"R-dist_as-fn-c_prinz_raw.svg")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input, active-ipynb]
---
emdcmp.utils.GitSHA(packages=["emdcmp", "pyloric-network-simulator"])
```

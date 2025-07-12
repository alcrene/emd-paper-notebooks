---
jupytext:
  formats: ipynb,md:myst
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

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

---
math:
  '\Bemd' : 'B_{#1}^{\mathrm{EMD}}'
  '\BQ' : 'B_{#1}^{Q}'
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_effect-of-c-on-calibs)=
# Effect of $c$ on calibration curves

%{{ startpreamble }}
%{{ endpreamble }}

+++ {"editable": true, "slideshow": {"slide_type": ""}}

> **NOTE** Within Jupyter Lab, this notebook is best displayed with [`jupyterlab-myst`](https://myst-tools.org/docs/mystjs/quickstart-jupyter-lab-myst).

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Libraries

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import math
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The `emdcmp` package

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import emdcmp
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Proect packages

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from config import config
import utils
import viz
import viz.emdcmp

from Ex_UV import (
    Bunits, Dataset, Q, gaussian_noise, CandidateModel,
    L_med, data_T, data_λ_min, data_λ_max, data_noise_s,
    EpistemicDistBiasSweep,
    c_chosen
)
import task_bq
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Visualization

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import holoviews as hv
sanitize = hv.core.util.sanitize_identifier_fn
hv.extension("matplotlib")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Define the epistemic distribution

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
#c_list = [2**-3, 2**-2, 2**-1, 2**0, 2**1]
c_list = [2**-15, 2**-12, 2**-9, 2**-6, 2**-3, 2**0, 2**3, 2**6]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{margin}
Number of replication experiments;
compute time is linear in `N`.
Larger values can only realistically be run on a cluster.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
#N = 64
#N = 256
N = 4096
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{margin}
`EpistemicDistBiasSweep` is defined [here](#code_uv_calib-dists)
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
Ω = EpistemicDistBiasSweep()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Setup two tasks with the same distribution: one using $\Bemd{}$, the other using $\BQ{}$

:::{note}
$\BQ{}$ is an alternative criterion we consider in our [supplementary](#sec_why-not-BQ).
In the paper we use $\BQ{}$ plots from the [neuron model](./Calibrating-with-BQ.ipynb) to illustrate how it differs from the $\Bemd{}$, but as we show below, the same exercise can also be done with the black body radiation models.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
task = emdcmp.tasks.Calibrate(
    reason = "UV calibration – RJ vs Plank – bias sweep",
    c_list = c_list,
    #c_list = [c_chosen],
    experiments = Ω.generate(N),
    Ldata = 1024,
    Linf = 12288
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
task_BQ = task_bq.CalibrateBQ(
    reason = "UV calibration – RJ vs Plank – bias sweep",
    #c_list = [-2**1, -2**0, -2**-1, -2**-2, -2**-3, -2**-4, 0, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1],
    c_list = c_list,
    experiments = Ω.generate(N),
    Ldata = 1024,
    Linf = 12288,
    LQ = 4000
)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Run the calibration tasks

:::{hint}
If the tasks have not been run yet, they will refuse to run unless the notebook repository is clean.
This is to ensure the results cached to disk are reproducible.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
task.run(cache=True)
task_BQ.run(cache=True);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{margin}
Task results are stored in a compressed format (a flat array) which requires the task arguments to be unpacked.
Tasks provide the `unpack_results` method to do this.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
calib_results = task.unpack_results(task.run(cache=True))
calib_results_BQ = task_BQ.unpack_results(task_BQ.run(cache=True))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_calib-uv_c-effect_plots)=
## Plot the results

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The `emdcmp` provides a plotting function under `emdcmp.viz.calibration_plot`.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
calib_normal = viz.emdcmp.calibration_plot(calib_results, target_bin_size=32)

task_bq.calib_point_dtype.names = ("Bemd", "Bconf")  # Hack to relabel fields in the way calibration_plot expects them
calib_BQ = viz.emdcmp.calibration_plot(calib_results_BQ, target_bin_size=32)
task_bq.calib_point_dtype.names = ("BQ", "Bepis")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The object itself will render as a plot, but it actually provides a few different methods to plot the data in different ways.
In particular, the different Holoviews plot elements can be accessed individually, so that users can compose their plots in the way that suits them.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
# CalibrationPlotElements doesn’t use LaTeX labels by default, because they aren’t supported by Bokeh
# It also doesn’t know that one of the plots is BQ instead of Bemd, so we also use `redim` to fix that
calib_normal = calib_normal.redim(Bemd =hv.Dimension("Bemd" , label="$B^{\\mathrm{EMD}}$"),
                                  Bepis=hv.Dimension("Bepis", label="$B^{\\mathrm{epis}}$"))
calib_BQ = calib_BQ.redim(Bemd =hv.Dimension("BQ" , label="$B^Q$"),
                          Bepis=hv.Dimension("Bepis", label="$B^{\\mathrm{epis}}$"))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig = calib_normal.overlayed_scatters + calib_BQ.overlayed_scatters.redim(c="c_Q", Bemd="BQ")
fig
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
(calib_normal.scatters + calib_BQ.scatters).opts(
    fig_inches=2, sublabel_format="", show_title=False)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# Map the labels to their corresponding power of 2, which is much easier to work with
labels = {int(round(math.log2(float(curve.label.lstrip('c='))))): curve.label
          for curve in calib_normal.calibration_curves}
san_labels = {i: sanitize(lbl) for i, lbl in labels.items()}
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
calibopts = (
    hv.opts.Overlay(backend="matplotlib",
                    #legend_cols=3,
                    #legend_opts={"columnspacing": .5, "alignment": "center",
                    #             "loc": "upper center"},
                    #hooks=[partial(viz.despine_hook(), left=False)],
                    #fig_inches=config.figures.defaults.fig_inches,
                    aspect=1, fontscale=1.3),
    hv.opts.Scatter(backend="matplotlib", s=20),
    hv.opts.Layout(sublabel_format=""),
    hv.opts.Layout(backend="matplotlib", hspace=0.1, vspace=0.05,
                   fig_inches=0.65*config.figures.defaults.fig_inches)
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
left   = calib_normal.select(c=[2**-15, 2**-12])
middle = calib_normal.select(c=[2**-9, 2**-6, 2**-3])
right  = calib_normal.select(c=[2**0, 2**3, 2**6])
```

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": ["remove-cell"]}

Reverse some plots to make the different curves easier to distinguish.
Also restrict the range of the `copper` palette, so we don’t go so far into the yellow which is hard to see on white and pink.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
left.scatter_palette = hv.Palette("copper", range=(0., .7))
_curves = middle.calibration_curves
_curves.data = {k:_curves.data[k] for k in reversed(sorted(_curves.data))}
middle.scatter_palette = hv.Palette("copper", range=(0, .9), reverse=True)  # Keep the order of colours the same: dark = smaller
_curves = right.calibration_curves
_curves.data = {k:_curves.data[k] for k in reversed(sorted(_curves.data))}
right.scatter_palette = hv.Palette("copper", range=(0, .9), reverse=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
hooks = [viz.despine_hook, viz.set_xticks_hook([0, 0.5, 1]), viz.set_yticks_hook([0, 0.5, 1]),
         viz.set_xticklabels_hook(["$0$", "$0.5$", "$1$"])]
left_hooks   = hooks + [viz.set_yticklabels_hook(["$0$", "$0.5$", "$1$"])]
middle_hooks = hooks + [viz.yaxis_off_hook]
right_hooks  = hooks + [viz.yaxis_off_hook]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig = left.overlayed_scatters.opts(hooks=left_hooks, xlabel="") \
                             .opts(hv.opts.Scatter(f"Scatter.{san_labels[-15]}", s=30)) \
                             .opts(hv.opts.Scatter(f"Scatter.{san_labels[-12]}", s=16)) \
      + middle.overlayed_scatters.opts(hooks=middle_hooks) \
                                 .opts(hv.opts.Scatter(f"Scatter.{san_labels[-9]}", s=18)) \
                                 .opts(hv.opts.Scatter(f"Scatter.{san_labels[-6]}", s=18)) \
                                 .opts(hv.opts.Scatter(f"Scatter.{san_labels[-3]}", s=24)) \
      + right.overlayed_scatters.opts(hooks=right_hooks, xlabel="") \
                                .opts(hv.opts.Scatter(f"Scatter.{san_labels[6]}", s=24)) \
                                .opts(hv.opts.Scatter(f"Scatter.{san_labels[3]}", s=14)) \
                                .opts(hv.opts.Scatter(f"Scatter.{san_labels[0]}", s=12))

fig.opts(*calibopts,
         hv.opts.Overlay(xlim=(-0.05, 1.05), # Make space so the dots at edges are fully plotted
                         legend_position="top_left",
                         legend_labels={lbl: f"$c=2^{{{p}}}$" for p, lbl in labels.items()}),
    )
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
hv.save(fig, config.paths.figures/"uv_calibration_effect-c_raw.svg")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
viz.glue("calib_Nsims_uv", N)
viz.glue("calib_num-bins_uv", len(calib_normal.calibration_curves.values()[0].data))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
emdcmp.utils.GitSHA(packages=["emdcmp"])
```

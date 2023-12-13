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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Pedagogical figures

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import Ex_UV
import emd_falsify as emd
#import matplotlib as mpl
import numpy as np
import holoviews as hv
from dataclasses import dataclass
from more_itertools import nth
#from myst_nb import glue
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
from config import config
from viz import glue
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Configuration

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
hv.extension("matplotlib", "bokeh")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
dims = viz.dims.matplotlib
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
@dataclass
class colors(viz.ColorScheme):
    mixed : str = config.figures.colors["bright"].green
    synth : str = config.figures.colors["bright"].purple
    δemd  : str = config.figures.colors["pale"].yellow
    qhat  : str = config.figures.colors["bright"].grey
    data  : str = config.figures.colors["bright"].grey
@dataclass
class color_labels:
    mixed : str = "green"
    synth : str = "red"
    δemd  : str = "yellow"
    qhat  : str = "grey"
    data  : str = "data"
colors
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## $δ^{\mathrm{EMD}}$ and path sampling

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
datasets = Ex_UV.panel_param_iterator(Ex_UV.L_med, "PPF sampling illustration", progbar=False)

D = nth(datasets, 5)
mixed_ppfs, synth_ppfs = Ex_UV.get_ppfs(D)

#phys_model = "Rayleigh-Jeans"
phys_model = "Planck"
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
c=Ex_UV.c_chosen
#c = .25
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
Φarr = np.linspace(0, 1, 192*8)  # *8 is to resolve the curve in the [0.97, 1] zoom
# mixed_ppf = mixed_ppfs["Planck"]
# synth_ppf = synth_ppfs["Planck"]
mixed_ppf = mixed_ppfs[phys_model]
synth_ppf = synth_ppfs[phys_model]
def δemd(Φarr): return abs(synth_ppf(Φarr) - mixed_ppf(Φarr))

mixed_curve = hv.Curve(zip(Φarr, mixed_ppf(Φarr)),
                       kdims=[dims.Φ], vdims=[dims.q], label="mixed PPF")
synth_curve = hv.Curve(zip(Φarr, synth_ppf(Φarr)),
                       kdims=[dims.Φ], vdims=[dims.q], label="synth PPF")
area = hv.Area((Φarr, mixed_ppf(Φarr) - c*δemd(Φarr), mixed_ppf(Φarr) + c*δemd(Φarr)),
               kdims=[dims.Φ], vdims=[dims.q, "q2"], label="δemd")

fig_onlycurves = area * mixed_curve * synth_curve
ticks = dict(xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[-6, -3, -0, 3, 6],
             yformatter=lambda y: str(y) if y in {-6, 6} else "",
             xformatter=lambda x: str(x) if x in {0, 1} else "")
hooks = [viz.despine_hook, viz.xlabel_shift_hook(3), viz.ylabel_shift_hook(2)]
fig_onlycurves.opts(
    hv.opts.Curve("Curve.Mixed_PPF", color=colors.mixed),
    hv.opts.Curve("Curve.Mixed_PPF", color=colors.mixed, backend="bokeh"),
    hv.opts.Curve("Curve.Synth_PPF", color=colors.synth),
    hv.opts.Curve("Curve.Synth_PPF", color=colors.synth, backend="bokeh"),
    hv.opts.Area(facecolor=colors.δemd, edgecolor="none", color="none", backend="matplotlib"),
    hv.opts.Area(fill_color=colors.δemd, line_color=None, backend="bokeh"),
    hv.opts.Overlay(legend_position="top_left", fontscale=1.3, backend="matplotlib"),
    hv.opts.Overlay(legend_position="top_left", backend="bokeh"),
    # NB: despine_hook looks at the ticks, so important to set the ticks of all elements
    hv.opts.Area(**ticks),
    hv.opts.Curve(backend="matplotlib", **ticks, hooks=hooks),
    hv.opts.Area(backend="matplotlib", hooks=[viz.despine_hook, viz.xlabel_shift_hook(), viz.ylabel_shift_hook(5)]),
)
fig_onlycurves.redim.range(q=(-7.1,6))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{note}
:class: margin
We normally don’t use a `res` greater than 8, because the increased precision in the integral decreases sharply beyond that point (and the computational cost *increases* sharply). However for purpose of illustration, since we zoom into the region $Φ \in [0.97, 1]$, we use `res=10` to better resolve the sampled paths.
:::

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
rng = utils.get_rng("pedag", "qpaths")
qpaths = emd.path_sampling.generate_quantile_paths(mixed_ppf, δemd, c=c, M=6, res=10, rng=rng)
qhat_curves = [hv.Curve(zip(Φhat, qhat), label="sampled PPF",
                        kdims=[dims.Φ], vdims=[dims.q])
               .opts(color=colors.lighten(rng.uniform(-0.2, +0.1)).qhat)
               .opts(color=colors.lighten(rng.uniform(-0.2, +0.1)).qhat, backend="bokeh")
               for Φhat, qhat in qpaths]
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig_with_qpaths = area * hv.Overlay(qhat_curves) * mixed_curve * synth_curve

ticks = dict(yticks=[-6, -5, -4, -3, -2, -1],
             yformatter=lambda y: str(y) if y in {-6, -1} else "")
fig_with_qpaths.opts(
    hv.opts.Curve(**ticks), hv.opts.Area(**ticks),
    hv.opts.Overlay(legend_opts={"loc": "upper left", "bbox_to_anchor":(0.05, 0.85)}),
    hv.opts.Overlay(legend_position="top_left", backend="bokeh")
)
fig_with_qpaths = fig_with_qpaths.redim.range(q=(-7,-1))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output, hide-input]
---
# Remove text descriptions for the published plot
fig = fig_with_qpaths
fig_with_qpaths = fig.redim(Φ=hv.Dimension('Φ', label='$Φ$', range=fig.get_dimension("Φ").range),
                            q=hv.Dimension('q', label='$q$', range=fig.get_dimension("q").range))

# Zoom near 0
ticks = dict(yticks=[-6.9, -6.8, -6.7, -6.6, -6.5], xticks=[0, 0.175, 0.35, 0.525, 0.7],
             yformatter = lambda y: str(y) if y in {-6.9, -6.5} else "",
             xformatter = lambda x: str(x) if x in {0, 0.7} else "")
hooks=[viz.despine_hook, viz.xlabel_shift_hook(), viz.ylabel_shift_hook(2)]
zoom_initial = fig_with_qpaths.clone().redim.range(q=(-6.97,-6.45), Φ=(0, 0.72))
zoom_initial.opts(hv.opts.Curve(**ticks, hooks=hooks),
                  hv.opts.Area (**ticks, hooks=hooks),
                  hv.opts.Overlay(show_legend=False), hv.opts.Overlay(show_legend=False, backend="bokeh"))

# Zoom near 1
ticks = dict(yticks=[-5, 0, 5, 10, 15, 20, 25], xticks=[0.97, 0.98, 0.99, 1],
             yformatter = lambda y: str(y) if y in {-5, 25} else "",
             xformatter = lambda x: str(x) if x in {0.97, 1} else "")
zoom_final = fig_with_qpaths.clone().redim.range(q=(-5, 25), Φ=(0.97, 1))
zoom_final.opts(hv.opts.Curve(**ticks),
                hv.opts.Area (**ticks),
                hv.opts.Overlay(show_legend=False), hv.opts.Overlay(show_legend=False, backend="bokeh"))

# Rectangles showing zooms in first panel
def get_rect_points(panel):
    x0, x1 = panel.get_dimension("Φ").range
    y0, y1 = panel.get_dimension("q").range
    return (x0, y0, x1, y1)
zoom_rects = hv.Rectangles([get_rect_points(zoom_initial), get_rect_points(zoom_final)])

panelA = fig_with_qpaths.redim.range(Φ=(-0.025,1.025))*zoom_rects
panelA.opts(legend_opts={"loc": "upper left", "bbox_to_anchor":(0.05, 0.85)})
layout = panelA + zoom_initial + zoom_final
# Set plot options
layout.opts(shared_axes=False, sublabel_format="({alpha})", sublabel_position=(0.4, 0.85))
layout.opts(shared_axes=False, backend="bokeh")
layout.opts(hv.opts.Layout(fontscale=1.3),
            hv.opts.Curve(fontscale=1.3),
            hv.opts.Area(fontscale=1.3),
            hv.opts.Rectangles(color="none", facecolor="none", edgecolor="#222222"),
            hv.opts.Rectangles(fill_color="none", line_color="#888888", backend="bokeh"))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input, full-width]
---
hv.output(layout.cols(3), backend="bokeh")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# HTML figure
viz.save(layout.opts(clone=True, fig_inches=4).cols(3), config.paths.figures/"pedag_qhat-paths.svg")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# For the pdf figure we compact it a bit further since paper space is precious
layout.cols(1).opts(
    hv.opts.Overlay(aspect=2.7),
    hv.opts.Layout(sublabel_position=(0.4, 0.8))
)
hooks = [viz.despine_hook, viz.ylabel_shift_hook(.9, .75)]
layout[0].opts(xlabel="").opts(hv.opts.Curve(hooks=hooks), hv.opts.Area(hooks=hooks))
hooks = [viz.despine_hook, viz.ylabel_shift_hook(1.7, .5)]
layout[1].opts(xlabel="").opts(hv.opts.Curve(hooks=hooks), hv.opts.Area(hooks=hooks))
layout

viz.save(layout.cols(1), config.paths.figures/"pedag_qhat-paths.pdf")
```

## Quantile axis flip

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
λ, B = D.data_model(30, rng=D.rng)
fitted_σT = Ex_UV.fit_gaussian_σT((λ,B))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
Q = Ex_UV.Q(phys_model, fitted_σT[phys_model].σ, Ex_UV.data_T)
qarr = np.sort(Q((λ,B)))
Φarr = np.arange(1, len(qarr)+1)/(len(qarr)+1)  # Remove end points to leave space for unseen data
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
empirical_cdf = hv.Scatter(zip(qarr, Φarr), kdims=[dims.q], vdims=[dims.Φ], label="empirical CDF")
empirical_ppf = hv.Scatter(zip(Φarr, qarr), kdims=[dims.Φ], vdims=[dims.q], label="empirical PPF")
#fitted_ppf = mixed_curve
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig = empirical_cdf.redim.range(q=(-7,5.5)) \
      + (mixed_curve.clone().redim.range(q=(-7,5.5))*empirical_ppf.redim.range(q=(-7,5.5))
        ).opts(title="empirical PPF\n(quantile function)")

qticks = [-6.6, -5, -3.3, -1.6, 0, 1.6, 3.3, 5]
qform = lambda x: str(x) if x in {-6.6, 5} else ""
Φticks = [0, 0.25, 0.5, 0.75, 1]
Φform = lambda x: str(x) if x in {0, 1} else ""
fig.opts(hv.opts.Scatter(color=colors.data, s=12),
         hv.opts.Scatter("Scatter.Empirical_CDF", xticks=qticks, xformatter=qform, yticks=Φticks, yformatter=Φform,
                         hooks=[viz.despine_hook, viz.xlabel_shift_hook(8), viz.ylabel_shift_hook(6)]),
         hv.opts.Scatter("Scatter.Empirical_PPF", xticks=Φticks, xformatter=Φform, yticks=qticks, yformatter=qform,
                         hooks=[viz.despine_hook, viz.xlabel_shift_hook(8), viz.ylabel_shift_hook(13,.5)]),
         hv.opts.Curve(alpha=0.25, linewidth=3,
                       xticks=Φticks, xformatter=Φform, yticks=qticks, yformatter=qform,
                       hooks=[viz.despine_hook]),
         hv.opts.Overlay(show_legend=False),
         hv.opts.Layout(fig_inches=config.figures.defaults.fig_inches/2, shared_axes=False,
                        sublabel_position=(0.4, 0.9), sublabel_format="")
        )
fig
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
viz.save(fig, config.paths.figures/"pedag_CDF_PPF.pdf")
viz.save(fig.clone().cols(1).opts(vspace=.75),
         config.paths.figures/"pedag_CDF_PPF.svg")  # Sized to insert in margin
```

## Exported notebook variables
These can be inserted into other pages

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
glue("color_mixed", color_labels.mixed, raw_myst=True, raw_latex=True)
glue("color_synth", color_labels.synth, raw_myst=True, raw_latex=True)
glue("color_deltaemd", color_labels.δemd, raw_myst=True, raw_latex=True)
glue("color_qhat", color_labels.qhat, raw_myst=True, raw_latex=True)
glue("color_data", color_labels.data, raw_myst=True, raw_latex=True)
glue("pedag_paths_c", c, raw_myst=True, raw_latex=True)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
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
